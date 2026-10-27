"""A base model with its story-world and soul adapters, for evaluation and simulation.

The base model is loaded once in 4 bits; the adapters are loaded by name and any set of them
can be made active. The intention of a character is read from the model as a distribution:
every answer prefix {"intend": "<x>" is scored as a whole string, and the urgency of the
character is the probability of not waiting."""

import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

INTENDS = ("speak", "act", "silent", "wait")


class AgentModel:
    def __init__(self, base, adapters=None):
        # a cap on the GPU memory of this process (CHARSIM_GPU_MIB), so that it runs out of
        # memory itself instead of the other trainings that share the GPU
        cap = os.environ.get("CHARSIM_GPU_MIB")
        if cap:
            torch.cuda.set_per_process_memory_fraction(
                min(1.0, int(cap) * 2**20 / torch.cuda.get_device_properties(0).total_memory))
        self.tok = AutoTokenizer.from_pretrained(base)
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(base, device_map={"": 0}, dtype=torch.bfloat16)
        self.peft = None
        for name, path in (adapters or {}).items():
            if self.peft is None:
                self.peft = PeftModel.from_pretrained(model, path, adapter_name=name)
            else:
                self.peft.load_adapter(path, adapter_name=name)
        self.net = self.peft.base_model.model if self.peft else model
        self.net.eval()
        self.cache_ok = os.environ.get("CHARSIM_NO_CACHE") != "1"
        self.active = list(adapters or {})
        self.activate([])

    def activate(self, names):
        """Make exactly these adapters active ([] = the base model alone)."""
        if self.peft is None:
            if names:
                raise ValueError("no adapters loaded")
            return
        if names:
            self.peft.base_model.enable_adapter_layers()
            self.peft.base_model.set_adapter(list(names))
        else:
            self.peft.base_model.disable_adapter_layers()
        self.active = list(names)

    def layers(self):
        return self.net.model.layers

    def set_steering(self, layer=None, vector=None):
        """Add a vector to the output of one decoder layer at every position (None removes it)."""
        if getattr(self, "_hook", None) is not None:
            self._hook.remove()
            self._hook = None
        if vector is None:
            return
        vec = vector.to(device="cuda", dtype=torch.bfloat16)

        def hook(module, inputs, output):
            if isinstance(output, tuple):
                return (output[0] + vec,) + tuple(output[1:])
            return output + vec

        self._hook = self.layers()[layer].register_forward_hook(hook)

    @torch.no_grad()
    def last_hidden(self, messages, layer):
        """Hidden state of the last prompt token at the output of a decoder layer."""
        ids = torch.tensor([self.prompt_ids(messages)]).cuda()
        out = self.net(input_ids=ids, output_hidden_states=True, logits_to_keep=1)
        return out.hidden_states[layer + 1][0, -1].float().cpu()

    def prompt_ids(self, messages):
        return self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                            enable_thinking=False)

    @torch.no_grad()
    def _logprob(self, prompt, continuations):
        """Sum of the log-probabilities of every continuation (token lists) after the prompt.
        The prompt is encoded once and its cache is repeated for the continuations."""
        if not self.cache_ok:
            return self._logprob_batch(prompt, continuations)
        out = self.net(input_ids=torch.tensor([prompt]).cuda(), use_cache=True, logits_to_keep=1)
        first = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
        past, n, k = out.past_key_values, len(continuations), max(map(len, continuations))
        try:
            past.batch_repeat_interleave(n)
        except AttributeError:
            self.cache_ok = False
            return self._logprob_batch(prompt, continuations)
        ids = torch.full((n, k), self.pad_id, dtype=torch.long)
        mask = torch.zeros((n, len(prompt) + k), dtype=torch.long)
        mask[:, :len(prompt)] = 1
        for i, c in enumerate(continuations):
            ids[i, :len(c)] = torch.tensor(c)
            mask[i, len(prompt):len(prompt) + len(c)] = 1
        pos = torch.arange(len(prompt), len(prompt) + k).expand(n, k)
        logits = self.net(input_ids=ids.cuda(), attention_mask=mask.cuda(), position_ids=pos.cuda(),
                          past_key_values=past).logits
        logp = torch.log_softmax(logits.float(), dim=-1)
        return [float(first[c[0]]) + sum(float(logp[i, j - 1, c[j]]) for j in range(1, len(c)))
                for i, c in enumerate(continuations)]

    @torch.no_grad()
    def _logprob_batch(self, prompt, continuations):
        """The same without a cache: the sequences are padded on the left so that only the
        logits of the last positions are computed."""
        seqs = [prompt + c for c in continuations]
        n, k = max(map(len, seqs)), max(map(len, continuations))
        ids = torch.full((len(seqs), n), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), n), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, n - len(s):] = torch.tensor(s)
            mask[i, n - len(s):] = 1
        ids, mask = ids.cuda(), mask.cuda()
        pos = (mask.cumsum(1) - 1).clamp(min=0)
        logits = self.net(input_ids=ids, attention_mask=mask, position_ids=pos, logits_to_keep=k + 1).logits
        logp = torch.log_softmax(logits.float(), dim=-1)
        out = []
        for i, c in enumerate(continuations):
            # the token at position p is predicted by the kept logits at index p - n + k
            out.append(sum(float(logp[i, p - n + k, ids[i, p]]) for p in range(n - len(c), n)))
        return out

    def _intend_tokens(self):
        """For every intention the first tokens of an answer in the training format, up to the
        delimiter after the intention (the closing quote is merged with it by the tokenizer)."""
        if not hasattr(self, "_conts"):
            self._conts = []
            for x in INTENDS:
                s = json.dumps({"intend": x} if x == "wait" else {"intend": x, "to": []})
                cut = s.index('"' + x + '"') + len(x) + 3
                ids = self.tok(s, add_special_tokens=False)["input_ids"]
                k = next(k for k in range(1, len(ids) + 1) if len(self.tok.decode(ids[:k])) >= cut)
                self._conts.append(ids[:k])
        return self._conts

    def intend_probs(self, messages):
        """{intend: probability} of the next move given the system and user messages."""
        prompt = self.prompt_ids(messages)
        lp = torch.tensor(self._logprob(prompt, self._intend_tokens()))
        p = torch.softmax(lp, dim=0)
        return dict(zip(INTENDS, map(float, p)))

    def choice_probs(self, messages, n):
        """Probabilities of the answers "1" to "n" (n < 10) given the messages."""
        prompt = self.prompt_ids(messages)
        lp = torch.tensor(self._logprob(prompt, [self.tok(str(i), add_special_tokens=False)["input_ids"]
                                                 for i in range(1, n + 1)]))
        return [float(x) for x in torch.softmax(lp, dim=0)]

    def answer_nll(self, messages, answer):
        """Mean negative log-likelihood per token of an answer string."""
        prompt = self.prompt_ids(messages)
        cont = self.tok(answer + self.tok.eos_token, add_special_tokens=False)["input_ids"]
        return -self._logprob(prompt, [cont])[0] / len(cont), len(cont)

    @torch.no_grad()
    def generate(self, messages, max_new_tokens=120, temperature=0.0, seed=0, prefix=""):
        """Text of the answer (prefix included), decoded greedily or by sampling."""
        prompt = self.prompt_ids(messages)
        pre = self.tok(prefix, add_special_tokens=False)["input_ids"] if prefix else []
        ids = torch.tensor([prompt + pre]).cuda()
        torch.manual_seed(seed)
        kw = {"do_sample": True, "temperature": temperature, "top_p": 0.95} if temperature > 0 else {"do_sample": False}
        out = self.net.generate(ids, max_new_tokens=max_new_tokens, pad_token_id=self.pad_id, **kw)
        return prefix + self.tok.decode(out[0, len(prompt) + len(pre):], skip_special_tokens=True)


def parse_move(text):
    """The first JSON object of an answer if it is a move, or None."""
    s = text.find("{")
    if s == -1:
        return None
    try:
        move, _ = json.JSONDecoder().raw_decode(text[s:])
    except ValueError:
        return None
    return move if isinstance(move, dict) and move.get("intend") in INTENDS else None
