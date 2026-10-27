"""QLoRA training of a story-world adapter or a soul adapter on a 4-bit base model.

--world FILE : causal language modelling on the text of the novel up to chapter c-1.
--soul FILE  : the decision points of one character (build_training.py), loss on the answer
               only. With --world-adapter the soul is trained on top of the frozen story-world
               adapter: both adapters are active in the forward pass and only the soul weights
               are updated, so that in the simulation the soul adds to the story world.
The adapter, its training log (losses, tokens, time, peak GPU memory) and the arguments are
written to --out."""

import argparse
import json
import math
import os
import random
import shutil
import sys
import time

import bitsandbytes as bnb
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
VAL_EVERY = 20  # every 20th block of the novel is held out from story-world training


def world_data(tok, path, max_len):
    with open(path, encoding="utf-8") as f:
        ids = tok(f.read(), add_special_tokens=False)["input_ids"]
    blocks = [ids[i:i + max_len] for i in range(0, len(ids), max_len)]
    rows = [{"input_ids": b, "labels": list(b)} for b in blocks if len(b) >= 64]
    return ([r for i, r in enumerate(rows) if i % VAL_EVERY != VAL_EVERY - 1],
            [r for i, r in enumerate(rows) if i % VAL_EVERY == VAL_EVERY - 1])


def chat_prompt(tok, messages):
    return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, enable_thinking=False)


def soul_data(tok, path, max_len):
    train, val, dropped = [], [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            prompt = chat_prompt(tok, s["messages"][:-1])
            answer = tok(s["messages"][-1]["content"] + tok.eos_token, add_special_tokens=False)["input_ids"]
            if len(prompt) + len(answer) > max_len:
                dropped += 1
                continue
            row = {"input_ids": prompt + answer, "labels": [-100] * len(prompt) + answer}
            (val if s.get("split") == "val" else train).append(row)
    return train, val, dropped


def batches(rows, size, rng):
    """Shuffled batches of rows of similar length (sorted inside windows of 50 batches)."""
    order = list(range(len(rows)))
    rng.shuffle(order)
    out = []
    win = size * 50
    for i in range(0, len(order), win):
        chunk = sorted(order[i:i + win], key=lambda j: len(rows[j]["input_ids"]))
        out += [chunk[k:k + size] for k in range(0, len(chunk), size)]
    rng.shuffle(out)
    return out


def collate(rows, pad_id, device):
    n = max(len(r["input_ids"]) for r in rows)
    ids = torch.full((len(rows), n), pad_id, dtype=torch.long)
    labels = torch.full((len(rows), n), -100, dtype=torch.long)
    mask = torch.zeros((len(rows), n), dtype=torch.long)
    for i, r in enumerate(rows):
        k = len(r["input_ids"])
        ids[i, :k] = torch.tensor(r["input_ids"])
        labels[i, :k] = torch.tensor(r["labels"])
        mask[i, :k] = 1
    return ids.to(device), mask.to(device), labels.to(device)


@torch.no_grad()
def evaluate(net, rows, size, pad_id, device):
    net.eval()
    total, count = 0.0, 0
    for i in range(0, len(rows), size):
        ids, mask, labels = collate(rows[i:i + size], pad_id, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = net(input_ids=ids, attention_mask=mask, labels=labels).loss
        n = int((labels[:, 1:] != -100).sum())
        total, count = total + float(loss) * n, count + n
    net.train()
    return total / max(1, count)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="directory of a 4-bit base model")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--world")
    src.add_argument("--soul")
    ap.add_argument("--world-adapter", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=None, help="default 1e-4 for --world and 2e-4 for --soul")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=None, help="default 1024 for --world and 1280 for --soul")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    lr = args.lr or (1e-4 if args.world else 2e-4)
    max_len = args.max_len or (1024 if args.world else 1280)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda"

    tok = AutoTokenizer.from_pretrained(args.base)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if args.world:
        train, val = world_data(tok, args.world, max_len)
        dropped = 0
    else:
        train, val, dropped = soul_data(tok, args.soul, max_len)
    n_tokens = sum(len(r["input_ids"]) for r in train)
    print(f"train rows {len(train)} ({n_tokens} tokens), val rows {len(val)}, dropped {dropped}, eos {tok.eos_token!r}")

    model = AutoModelForCausalLM.from_pretrained(args.base, device_map={"": 0}, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    cfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout, target_modules=TARGETS,
                     bias="none", task_type="CAUSAL_LM")
    if args.world_adapter:
        model = PeftModel.from_pretrained(model, args.world_adapter, adapter_name="world", is_trainable=False)
        model.add_adapter("soul", cfg)
        model.base_model.set_adapter(["world", "soul"])
        for n, p in model.named_parameters():
            p.requires_grad = "lora_" in n and ".soul." in n
        for m in model.modules():  # the frozen story world behaves as in inference
            if hasattr(m, "lora_dropout") and "world" in m.lora_dropout:
                m.lora_dropout["world"] = torch.nn.Identity()
        saved = ["soul"]
    else:
        model = get_peft_model(model, cfg)
        saved = ["default"]
    # the transformers model with the LoRA layers injected; calling it directly keeps the
    # list of active adapters that PeftModel.forward does not accept
    net = model.base_model.model
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable parameters {sum(p.numel() for p in params)}")

    rng = random.Random(args.seed)
    steps = args.epochs * math.ceil(math.ceil(len(train) / args.batch) / args.accum)
    opt = bnb.optim.PagedAdamW8bit(params, lr=lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.05 * steps)), steps)
    log = {"args": vars(args), "lr": lr, "max_len": max_len, "train_rows": len(train), "val_rows": len(val),
           "dropped": dropped, "train_tokens_per_epoch": n_tokens, "optimizer_steps": steps, "epochs": [],
           "val_loss_before": evaluate(net, val, args.batch, pad_id, device) if val else None}
    print(f"val loss before training {log['val_loss_before']}")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    net.train()
    for epoch in range(args.epochs):
        run, n_run = 0.0, 0
        bl = batches(train, args.batch, rng)
        for i, b in enumerate(bl):
            ids, mask, labels = collate([train[j] for j in b], pad_id, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = net(input_ids=ids, attention_mask=mask, labels=labels).loss
            (loss / args.accum).backward()
            run, n_run = run + float(loss), n_run + 1
            if (i + 1) % args.accum == 0 or i + 1 == len(bl):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if (i + 1) % 50 == 0:
                print(f"  epoch {epoch + 1} batch {i + 1}/{len(bl)} loss {run / n_run:.4f} {time.time() - t0:.0f} s",
                      flush=True)
        vl = evaluate(net, val, args.batch, pad_id, device) if val else None
        log["epochs"].append({"train_loss": run / max(1, n_run), "val_loss": vl, "seconds": time.time() - t0})
        print(f"epoch {epoch + 1}: train loss {run / max(1, n_run):.4f}, val loss {vl}", flush=True)
    log["seconds"] = time.time() - t0
    log["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, selected_adapters=saved)
    if saved != ["default"]:
        sub = os.path.join(args.out, saved[0])
        for name in os.listdir(sub):
            shutil.move(os.path.join(sub, name), os.path.join(args.out, name))
        os.rmdir(sub)
    with open(os.path.join(args.out, "train_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=1)
    print(json.dumps({k: v for k, v in log.items() if k != "args"}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
