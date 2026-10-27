"""Gradio interface of the simulation.

Simulate: choose the base model, the adapter set, the scene, the mode, the memory tools and the
character tools (T4 plan, T5 state, T6 scene check, T7 consistency check, T8 choice, T9
turn-taking, with the T5 texts of config/state_texts2.json as in the simulations); edit the
card of every character, choose a card edit or a steering coefficient; run the scene turn by
turn, run it to the end, or fork it from any turn. For every turn the urgencies of the
characters are shown, and for a chosen character what it perceives: the T1 relations (graph),
the T2 memories and the exact input of its model. A session can be exported as JSON.
Results: browse the simulated sessions next to the original scene, with the verdicts of the
judges. Run on the GPU machine and open it through an SSH tunnel:
    python scripts/app.py --port 7860   then   ssh -L 7860:127.0.0.1:7860 <gpu machine>"""

import argparse
import copy
import glob
import html
import json
import math
import os
import sys
import tempfile
import time

import gradio as gr
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from charsim import pseudonyms  # noqa: E402
from charsim.agent import AgentModel  # noqa: E402
from charsim.cast import Cast  # noqa: E402
from charsim.samples import SYSTEM, build_tools, card_text  # noqa: E402
from charsim.scene import SceneRun  # noqa: E402
from charsim.transcript import judge_line, original_lines, scene_items  # noqa: E402

MODELS = os.environ.get("MODELS_DIR", "/media/w/DATA1/models")
BASES = {"llama": f"{MODELS}/Meta-Llama-3.1-8B-Instruct-bnb-4bit", "qwen": f"{MODELS}/Qwen3-8B-bnb-4bit"}
RUNS = ["llama", "qwen", "qwen_pseudo"]
CHAR_TOOLS = ["T4 plan", "T5 state", "T6 scene check", "T7 consistency check", "T8 choice", "T9 turn-taking"]
with open("config/state_texts2.json", encoding="utf-8") as f:
    STATE_TEXTS = json.load(f)
COLORS = ["#2563eb", "#c2410c", "#15803d", "#7c3aed"]
MAX_CHARS = 3

CAST = Cast("config/cast.json")
with open("config/cards.json", encoding="utf-8") as f:
    CARDS = json.load(f)
with open("config/scenes.json", encoding="utf-8") as f:
    SCENES = {s["id"]: s for s in json.load(f)["scenes"]}
TABLE = pseudonyms.load("config/pseudonyms.json")
LOADED = {"key": None, "agent": None, "souls": [], "tools": {}}


def load(run, scene_id):
    sc = SCENES[scene_id]
    key = (run, sc["chapter"])
    if LOADED["key"] != key:
        LOADED["agent"] = None
        torch.cuda.empty_cache()
        adir = f"adapters/{run}/c{sc['chapter']}"
        souls = sorted(os.path.basename(p)[5:] for p in glob.glob(f"{adir}/soul_*")
                       if os.path.isfile(f"{p}/train_log.json"))
        LOADED.update(key=key, souls=souls, agent=AgentModel(
            BASES[run.split("_")[0]], {"world": f"{adir}/world"} | {f"soul_{s}": f"{adir}/soul_{s}" for s in souls}))
    if sc["chapter"] not in LOADED["tools"]:
        LOADED["tools"][sc["chapter"]] = build_tools("data", CAST, sc["chapter"], 1)
    return LOADED["agent"], LOADED["souls"], LOADED["tools"][sc["chapter"]]


def scene_cast(scene_id):
    return SCENES[scene_id]["cast"]


def default_cards(scene_id):
    sc = SCENES[scene_id]
    cards = [card_text(CARDS, sc["chapter"], cid) for cid in sc["cast"]]
    return cards + [""] * (MAX_CHARS - len(cards))


def trait_choices(scene_id, i):
    sc = SCENES[scene_id]
    if i >= len(sc["cast"]):
        return ["none"]
    cid = sc["cast"][i]
    return ["none"] + [f"{e['trait']}:{d}" for e in sc["trait_edits"] if e["character"] == cid for d in ("up", "down")]


def steer_choices(scene_id, i):
    sc = SCENES[scene_id]
    if i >= len(sc["cast"]):
        return ["none"]
    return ["none"] + [e["trait"] for e in sc["trait_edits"] if e["character"] == sc["cast"][i]]


# rendering -----------------------------------------------------------------------------

def transcript_html(sess):
    if not sess:
        return "<p>Start a scene.</p>"
    run = sess["run"]
    color = {cid: COLORS[i % len(COLORS)] for i, cid in enumerate(run.sc["cast"])}
    out = ["<div style='font-family:Georgia,serif;font-size:15px;line-height:1.5'>"]
    for t in run.turns:
        it = t["item"]
        line = judge_line(it, CAST)
        if run.table:
            line = pseudonyms.apply(line, run.table)
        bars = " ".join(
            f"<span title='{CAST.name(c)}' style='display:inline-block;width:{max(2, int(60 * u))}px;height:8px;"
            f"background:{color[c]};opacity:{1 if c == t['actor'] else 0.35};margin-right:2px'></span>"
            for c, u in t["urgency"].items())
        change = f" <b>state: {html.escape(json.dumps(t['state_change']))}</b>" if t["state_change"] else ""
        stage = f" <i>[{t['stage']}]</i>" if t.get("stage") else ""
        out.append(f"<div style='margin:4px 0'><span style='color:#888;font-size:12px'>{t['turn'] + 1:2d}</span> "
                   f"{bars} <span style='color:{color[t['actor']]}'>{html.escape(line)}</span>{change}{stage}</div>")
    out.append("</div>")
    return "".join(out)


def urgency_rows(sess):
    if not sess:
        return {"data": [], "headers": ["turn"]}
    run = sess["run"]
    return {"headers": ["turn", *[CAST.name(c) for c in run.sc["cast"]], "takes the floor", "intention"],
            # with T9 only the characters addressed by the last move have an urgency
            "data": [[t["turn"] + 1, *[round(t["urgency"][c], 3) if c in t["urgency"] else "" for c in run.sc["cast"]],
                      CAST.name(t["actor"]),
                      t["intend"]] for t in run.turns]}


def relation_svg(tools, viewer, scene_id, size=420):
    others = {}
    for r in tools.relations:
        if viewer in r["known_by"]:
            for c in (r["source"], r["target"]):
                if c != viewer:
                    others[c] = others.get(c, 0) + 1
    nodes = [viewer] + [c for c, _ in sorted(others.items(), key=lambda x: -x[1])[:8]]
    for c in scene_cast(scene_id):
        if c not in nodes:
            nodes.append(c)
    pos = {}
    for i, c in enumerate(nodes):
        a = 2 * math.pi * i / len(nodes)
        pos[c] = (size / 2 + 0.38 * size * math.cos(a), size / 2 + 0.38 * size * math.sin(a))
    edges = []
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            # the latest relation between a and b that the viewer knows
            rels = [r for r in tools.relations if {r["source"], r["target"]} == {a, b} and viewer in r["known_by"]]
            if rels:
                r = max(rels, key=lambda x: x["chapter"])
                col = "#15803d" if r["polarity"] > 0 else "#b91c1c" if r["polarity"] < 0 else "#6b7280"
                (x1, y1), (x2, y2) = pos[a], pos[b]
                edges.append(f"<line x1='{x1:.0f}' y1='{y1:.0f}' x2='{x2:.0f}' y2='{y2:.0f}' stroke='{col}' "
                             f"stroke-width='2'><title>{html.escape(r['relation'])} (ch. {r['chapter']})</title></line>"
                             f"<text x='{(x1 + x2) / 2:.0f}' y='{(y1 + y2) / 2:.0f}' font-size='10' fill='{col}'>"
                             f"{html.escape(r['relation'])}</text>")
    labels = [f"<circle cx='{x:.0f}' cy='{y:.0f}' r='6' fill='{'#111' if c == viewer else '#9ca3af'}'/>"
              f"<text x='{x + 8:.0f}' y='{y + 4:.0f}' font-size='12'>{html.escape(CAST.name(c))}</text>"
              for c, (x, y) in pos.items()]
    return (f"<svg width='{size + 120}' height='{size}' style='background:#fafafa;border:1px solid #ddd'>"
            + "".join(edges) + "".join(labels) + "</svg>")


# actions -------------------------------------------------------------------------------

def start(run_name, scene_id, mode, tools_mode, t3, char_tools, seed, temperature, *per_char):
    agent, souls, tools = load(run_name, scene_id)
    sc = SCENES[scene_id]
    flags = {f"t{k}": any(t.startswith(f"T{k} ") for t in char_tools or []) for k in range(4, 10)}
    if flags["t5"]:
        sc = sc | {"state_texts": STATE_TEXTS[scene_id]}
    cards_in, edits_in, steer_in, alpha_in = (per_char[i * MAX_CHARS:(i + 1) * MAX_CHARS] for i in range(4))
    edits, steer = {}, {}
    for i, cid in enumerate(sc["cast"]):
        if edits_in[i] and edits_in[i] != "none":
            trait, direction = edits_in[i].rsplit(":", 1)
            edits[cid] = [trait, direction]
        if steer_in[i] and steer_in[i] != "none" and alpha_in[i]:
            path = f"results/steer/{run_name}/{scene_id}/{cid}__{steer_in[i].replace(' ', '_')}.pt"
            if os.path.exists(path):
                steer[cid] = {"path": path, "alpha": float(alpha_in[i])}
    cfg = {"mode": mode, "tools": tools_mode, "t3_to_agents": bool(t3), "edits": edits, "steer": steer,
           "souls": souls, "max_turns": 20, "temperature": float(temperature)} | flags
    table = TABLE if run_name.endswith("_pseudo") else None
    run = SceneRun(sc, CAST, CARDS, agent, tools, cfg, int(seed), table, scene_items("data", sc)[0])
    for i, cid in enumerate(sc["cast"]):
        if cards_in[i] and cards_in[i].strip() != card_text(CARDS, sc["chapter"], cid, tuple(edits[cid]) if cid in edits else None):
            text = SYSTEM.format(card=cards_in[i].strip())
            run.system[cid] = pseudonyms.apply(text, table) if table else text
    sess = {"run": run, "snapshots": [snapshot(run)], "times": []}
    return sess, transcript_html(sess), urgency_rows(sess), status(sess)


def snapshot(run):
    return copy.deepcopy({"items": run.items, "turns": run.turns, "state": run.state, "stage": run.stage,
                          "rng": run.rng.getstate()})


def status(sess):
    if not sess:
        return ""
    run = sess["run"]
    t = sess["times"]
    return (f"turn {len(run.turns)} / {run.cfg['max_turns']}, state {json.dumps(run.state)}, "
            f"adapters {LOADED['souls']}, mean {sum(t) / len(t) if t else 0:.1f} s per turn, "
            f"peak GPU {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB")


def step(sess):
    if not sess:
        return sess, transcript_html(sess), urgency_rows(sess), "Start a scene first."
    run = sess["run"]
    if len(run.turns) < run.cfg["max_turns"]:
        t0 = time.time()
        run.step(len(run.turns))
        sess["times"].append(time.time() - t0)
        sess["snapshots"].append(snapshot(run))
    return sess, transcript_html(sess), urgency_rows(sess), status(sess)


def run_all(sess):
    if not sess:
        yield sess, transcript_html(sess), urgency_rows(sess), "Start a scene first."
        return
    while len(sess["run"].turns) < sess["run"].cfg["max_turns"]:
        yield step(sess)


def fork(sess, k):
    """Go back to the state after turn k; the next steps draw a new continuation."""
    if not sess:
        return sess, transcript_html(sess), urgency_rows(sess), "Start a scene first."
    k = max(0, min(int(k), len(sess["snapshots"]) - 1))
    snap = copy.deepcopy(sess["snapshots"][k])
    run = sess["run"]
    run.items, run.turns, run.state, run.stage = snap["items"], snap["turns"], snap["state"], snap["stage"]
    run.rng.setstate(snap["rng"])
    run.seed += 1000  # a new continuation of the generation
    sess["snapshots"] = sess["snapshots"][:k + 1]
    return sess, transcript_html(sess), urgency_rows(sess), status(sess)


def perceive(sess, viewer_name, scene_id):
    sc = SCENES[scene_id]
    ids = {CAST.name(c): c for c in sc["cast"]}
    viewer = ids.get(viewer_name, sc["cast"][0])
    tools = LOADED["tools"].get(sc["chapter"]) or build_tools("data", CAST, sc["chapter"], 1)
    events = [e for e in tools.events if viewer in e["witnesses"]]
    rows = [[e["chapter"], e["summary"], ", ".join(CAST.name(p) for p in e["participants"])] for e in events[-40:]]
    prompt = ""
    if sess:
        msgs = sess["run"].messages(viewer)
        prompt = f"SYSTEM:\n{msgs[0]['content']}\n\nUSER:\n{msgs[1]['content']}"
    return relation_svg(tools, viewer, scene_id), rows, prompt


def export(sess):
    if not sess:
        return None
    run = sess["run"]
    lines = [ln for ln in (judge_line(t["item"], CAST) for t in run.turns) if ln]
    res = {"scene": run.sc["id"], "seed": run.seed, "config": run.cfg, "turns": run.turns, "final_state": run.state,
           "transcript": lines, "system": run.system}
    path = os.path.join(tempfile.mkdtemp(), f"{run.sc['id']}_seed{run.seed}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    return path


# results browser -----------------------------------------------------------------------

def result_files():
    return sorted(glob.glob("results/sim/*/*/*/seed*.json"))


def load_judged():
    out = {}
    for path in glob.glob("results/judge/*.jsonl"):
        with open(path, encoding="utf-8") as f:
            for rec in map(json.loads, f):
                if rec.get("ok"):
                    out.setdefault(rec["key"], {})[rec["judge"]] = rec
    return out


def show_result(path):
    if not path:
        return "", "", ""
    with open(path, encoding="utf-8") as f:
        res = json.load(f)
    sc = SCENES[res["scene"]]
    lines = [ln for ln in (judge_line(t["item"], CAST) for t in res["turns"]) if ln]
    key = os.path.relpath(path, "results/sim").replace(os.sep, "/")
    verdicts = load_judged().get(key, {})
    v = "<br>".join(f"<b>{j}</b>: ending {r['ending']} ({html.escape(sc['ending_classes'][r['ending']])}), "
                    f"propositions {r['propositions']}, ratings {r['character']}/{r['coherence']}/{r['plausibility']}"
                    for j, r in verdicts.items()) or "not judged yet"
    left = "<br>".join(html.escape(ln) for ln in lines)
    right = "<br>".join(html.escape(ln) for ln in original_lines("data", sc, CAST))
    return (f"<div style='font-family:Georgia,serif'>{left}</div>", f"<div style='font-family:Georgia,serif'>{right}</div>",
            f"final state {json.dumps(res['final_state'])}<br>{v}")


# layout --------------------------------------------------------------------------------

def build():
    first = "pearls"
    with gr.Blocks(title="CharSim") as demo:
        sess = gr.State(None)
        with gr.Tab("Simulate"):
            with gr.Row():
                run_name = gr.Dropdown(RUNS, value="llama", label="Base model and adapters")
                scene_id = gr.Dropdown(list(SCENES), value=first, label="Scene")
                mode = gr.Dropdown(["soul", "world", "prompted"], value="soul", label="Mode")
                tools_mode = gr.Dropdown(["on", "off", "t1", "t2"], value="on", label="Tools (T1 relations, T2 memory)")
                t3 = gr.Checkbox(False, label="Show the stage (T3) to the characters")
                seed = gr.Number(0, precision=0, label="Seed")
                temperature = gr.Slider(0.1, 1.5, 0.8, step=0.05, label="Temperature")
            char_tools = gr.CheckboxGroup(CHAR_TOOLS, value=CHAR_TOOLS, label="Character tools")
            cards, edits, steers, alphas = [], [], [], []
            with gr.Row():
                for i in range(MAX_CHARS):
                    with gr.Column():
                        cid = scene_cast(first)[i] if i < len(scene_cast(first)) else None
                        cards.append(gr.Textbox(default_cards(first)[i], lines=6,
                                                label=f"Card of {CAST.name(cid)}" if cid else "(no character)"))
                        edits.append(gr.Dropdown(trait_choices(first, i), value="none", label="Card edit"))
                        steers.append(gr.Dropdown(steer_choices(first, i), value="none", label="Steering trait"))
                        alphas.append(gr.Slider(-2, 2, 0, step=0.25, label="Steering coefficient"))
            with gr.Row():
                b_start, b_step, b_all = gr.Button("Start", variant="primary"), gr.Button("Next turn"), gr.Button("Run to the end")
                fork_k = gr.Number(0, precision=0, label="Fork after turn")
                b_fork, b_export = gr.Button("Fork"), gr.Button("Export JSON")
            stat = gr.Textbox(label="Status", interactive=False)
            with gr.Row():
                with gr.Column(scale=3):
                    stage = gr.HTML(label="Scene")
                with gr.Column(scale=2):
                    urg = gr.Dataframe(label="Urgency per turn", interactive=False)
            file_out = gr.File(label="Session")
            with gr.Accordion("What a character perceives", open=False):
                with gr.Row():
                    viewer = gr.Dropdown([CAST.name(c) for c in scene_cast(first)], label="Character")
                    b_view = gr.Button("Show")
                graph = gr.HTML(label="T1 relations")
                memory = gr.Dataframe(headers=["chapter", "event", "participants"], label="T2 memories")
                model_input = gr.Textbox(label="Input of the character's model", lines=16)

            def on_scene(s):
                ups = []
                for i in range(MAX_CHARS):
                    cid = scene_cast(s)[i] if i < len(scene_cast(s)) else None
                    ups.append(gr.update(value=default_cards(s)[i], label=f"Card of {CAST.name(cid)}" if cid else "(no character)"))
                return [*ups, *[gr.update(choices=trait_choices(s, i), value="none") for i in range(MAX_CHARS)],
                        *[gr.update(choices=steer_choices(s, i), value="none") for i in range(MAX_CHARS)],
                        gr.update(choices=[CAST.name(c) for c in scene_cast(s)], value=None)]

            scene_id.change(on_scene, scene_id, [*cards, *edits, *steers, viewer])
            outs = [sess, stage, urg, stat]
            b_start.click(start, [run_name, scene_id, mode, tools_mode, t3, char_tools, seed, temperature,
                                  *cards, *edits, *steers, *alphas], outs)
            b_step.click(step, sess, outs)
            b_all.click(run_all, sess, outs)
            b_fork.click(fork, [sess, fork_k], outs)
            b_export.click(export, sess, file_out)
            b_view.click(perceive, [sess, viewer, scene_id], [graph, memory, model_input])
        with gr.Tab("Results"):
            files = gr.Dropdown(result_files(), label="Session")
            b_refresh = gr.Button("Refresh the list")
            verdict = gr.HTML()
            with gr.Row():
                sim_text = gr.HTML(label="Simulated scene")
                orig_text = gr.HTML(label="Original scene")
            b_refresh.click(lambda: gr.update(choices=result_files()), None, files)
            files.change(show_result, files, [sim_text, orig_text, verdict])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    build().queue().launch(server_name="127.0.0.1", server_port=args.port)
