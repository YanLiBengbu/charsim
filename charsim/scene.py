"""Director-free simulation of a test scene.

Every turn each character of the scene reads the scene so far as it perceives it, together
with the T1 and T2 results that the framework queries for it, and the model of the character
gives the probability of each intention. The urgency of a character is the probability of not
waiting, and the character who takes the floor is drawn in proportion to the urgencies. The
chosen character draws an intention from its distribution without "wait" and generates the
rest of its move. As in training, the scene continues the timeline of its chapter: the lines of
the chapter before the start anchor of the scene open the history. The opening circumstances of
a character are its own first thought, pinned at the top of the scene it perceives (thoughts
are shown only to the thinker, as in training).
A rule-based state machine follows the objects of the scene. Every scene
runs for the same number of turns. In the ablation T3 the structure tool (the base model
without adapters, prompted) observes the stage of the scene at every turn and the stage is
shown to the characters. A character without a soul adapter of its own plays with the shared
supporting-cast soul when there is one.

Two baselines of the follow-up (2026-10-26) replace the urgency draw. With "director" the base
model without adapters knows the cards, the opening circumstances and the premise of the scene
(the question it must decide, not the answer) and at every turn chooses the character who moves
and gives it a direction, shown to that character only as its thought for this move; it may
not choose the character of the last two moves (the urgency draw among the others then
chooses, without a direction), and a direction it has given before is not shown again. The
director prompt is the fourth of six versions tried in smoke tests (results/sim_smoke*). With "coser" (the given
circumstances of CoSER, Wang et al., 2025) every character knows the plot of the original scene
from its system prompt and a next-speaker predictor that knows the plot chooses who moves.
When the answer of the director or predictor names no character of the scene, the urgency draw
chooses. As in the main conditions, every scene runs for max_turns moves: after a decision the
director often raises the question again. Two ways to let the base model end the scene were
tried on 2026-10-26 and dropped: an END answer of the director or predictor (smoke test
results/sim_smoke7: the director never gave it) and a separate yes/no question after every
move (checked on saved sessions: "decided" at the first allowed move in nearly every session,
also where nothing had been decided).

Two tools of a second follow-up (2026-10-26, asked for by the authors after the first one) let
the characters pursue what they want without a director. With T4 (plan) the base model without
adapters thinks as the character, from its card, its opening circumstances, its T1 and T2
results and the scene as the character perceives it, about what the character wants from the
others now and what it will say or do next; the plan is shown to the character as its last
thought. Every character plans before the first move, and the character who takes the floor
plans again before each later move of its own. The planner does not see its earlier plans
(a director shown its earlier directions copied them). With T5 (state) every character sees
the place of the scene and who holds the object of the scene, updated after every move by the
rules of charsim/state.py. The plan prompt was revised once after a smoke test
(results/sim_smoke_t45): the first plans were long, mixed several wishes and planned to go
elsewhere ("then return to the tea house"), and the characters followed them away; the plan is
now one short sentence about the people here and now. With T6 (scene check) a move that is
addressed to a character who is not in the scene, or an action that takes its character away
from the place or time of the scene (charsim/state.py, the rule of the descriptive measure
"away") or names a character who is not in the scene (other than thinking of one: "Wang Lung
buys a gold pin for Lotus", added after the second smoke test, results/sim_smoke_t456, where
the other rules caught nothing of this kind) is drawn again, at most twice; the third draw is
kept. On the saved soul sessions the check would have drawn again 8% of the moves.

Two more tools of a third follow-up (2026-10-26, after the sessions of the second were read)
let the characters act on the state and on their plans. With T7 (consistency check) an action
that is impossible in the state shown by T5 (charsim/state.py: buying something, since no scene
is at a market, or hiding, keeping or wearing the pearls while another character has them) is
drawn again like a move of T6, with the same limit. With T8 (choice) the character draws three
moves that pass the checks (at most five draws, the last one kept when no move passed) and the
base model without adapters, thinking as the character with its card, what it perceives and its
plan, chooses the one it would really do: the move whose number is the most probable answer.
The soul still writes every candidate; the choice uses nothing the character does not know. In
the sessions of the second follow-up the plans often asked for the pearls while the moves did
something else ("Wang Lung buys red paper"), and the characters kept hiding the pearls after
they had changed hands. With T9 (turn-taking) the characters spoken to by the last move (speech
or its gist, addressed to characters of the scene) answer: the urgency draw chooses among them
only, as the current speaker selects the next one in conversation (Sacks, Schegloff and
Jefferson, 1974); the one chosen may still keep silent. In the director-free sessions saved
before, a character spoken to moved next after 27-52% of such moves, against 52-80% with the
director or the next-speaker predictor of CoSER.

T7 was extended after the first session of the third follow-up (Qwen, pearls, seed 100), where
Wang Lung "takes the jewels from O-lan" without the state of T5 following, O-lan hid the pearls in
eight of her ten moves and a purchase was kept as the last of five draws, and the second (seed
101), where Wang Lung gave O-lan the pearls she had. With T7 the state of T5 also follows "the
jewels" (the descriptive measures keep their rule), giving or showing the pearls that another
character has is impossible like hiding them, a move that repeats an
earlier move of the same kind of its character (words without function words and names, cut to
four letters, Jaccard overlap at least 0.7: 9% of the saved moves; of 40 random hits read, at
most one doubtful) is drawn again, and a character whose draws all fail the checks says nothing."""

import json
import random
import re

import torch

from . import pseudonyms
from .agent import INTENDS, parse_move
from .samples import HISTORY, QUERY_LINES, SYSTEM, USER, card_text, render
from .state import AWAY, JEWEL, PEARL, after_move, impossible, repeats

STAGES = ("setup", "conflict", "climax", "resolution")
# an action that only thinks of an absent character stays in the scene (T6)
THINK = re.compile(r"\b(thinks?|thought|remembers?|dreams?|longs?|wish(es)?|imagines?)\b", re.I)

T3_PROMPT = """You observe a scene of a novel as it is being written. The scene so far:
{lines}

At which stage is the scene now: "setup", "conflict", "climax" or "resolution"? Has the scene reached a natural end? Answer only with JSON of the form {{"stage": "...", "ended": true or false}}."""

DIRECTOR_PROMPT = """You direct a scene of a novel. The characters of the scene:
{cards}

What the scene is about: {premise}

The scene so far:
{lines}

The last moves were made by: {last}.

The scene has {left} more moves, all in this place. Choose the character who moves next: the one who would naturally move now, often the one just spoken to, and never the one who made the last two moves. Give that character a direction for this move: one short sentence of at most 15 words in the second person that says what the character wants, feels or means to ask or press for now, so that the question of the scene is raised and pressed until the characters decide it. The direction never decides the answer to the question of the scene for anyone, never sends a character away from this place, and never asks a character for what its card says the character would not do. Answer only with JSON of the form {{"next": "<name>", "direction": "..."}}, with one of these names: {names}."""

PREDICTOR_PROMPT = """You follow a scene of a novel as it is acted. The characters of the scene: {names}.

The plot of the scene: {plot}

The scene so far:
{lines}

Who moves next? Answer only with JSON of the form {{"next": "<name>"}}, with one of these names: {names}."""

PLAN_PROMPT = """A character of a novel: {card}

{user}

Think privately as {name}. Only the people named above are here, and the scene stays in this place and at this moment. What do you want from them now, and what will you say or do to them next to get it? Answer with one sentence of at most 25 words in the first person, without plans for later or for another place."""

CHOICE_PROMPT = """A character of a novel: {card}

{user}

Think privately as {name}.{want} You could do one of these next:
{options}

Which one would you really do now? Answer only with its number."""

# state rules: (scene, actor, pattern of a move text, state key, new value)
TRANSFER = {
    "pearls": [("o_lan", re.compile(r"\b(give|gives|hand|hands|holds? out|offers?|puts?|places?|drops?|lays?)\b"
                                    r"[^.]*\bpearls?\b|\bpearls?\b[^.]*\b(to him|to wang lung|into his)", re.I),
                "pearls", "wang_lung"),
               ("wang_lung", re.compile(r"\b(takes?|snatch(es)?|grabs?|seizes?)\b[^.]*\bpearls?\b", re.I),
                "pearls", "wang_lung")],
}


class SceneRun:
    def __init__(self, scene, cast, cards, agent, tools, config, seed, table=None, opening=None):
        """config: {"mode": prompted | world | soul, "tools": on | off | t1 | t2, "t3_to_agents": bool,
        "edits": {cid: [trait, up | down]}, "steer": {cid: {"path": persona vector, "alpha": float}},
        "souls": [ids with a soul adapter, "cast" for the shared one], "max_turns": int,
        "temperature": float, "baseline": None | director | coser (the scene then has "premise" or "plot"),
        "t4": bool, "t5": bool (the scene then has "state_texts"), "t6": bool, "t7": bool, "t8": bool, "t9": bool}"""
        self.sc, self.cast, self.agent, self.tools = scene, cast, agent, tools
        self.cfg = {"mode": "soul", "tools": "on", "t3_to_agents": False, "edits": {}, "steer": {}, "souls": [],
                    "max_turns": 20, "temperature": 0.8, "baseline": None, "t4": False, "t5": False,
                    "t6": False, "t7": False, "t8": False, "t9": False} | config
        self.seed, self.rng, self.table = seed, random.Random(seed), table
        self.state = dict(scene["state"])
        self.plans, self.shown = {}, dict(scene["state"])  # the current plans (T4) and the state shown (T5)
        self.items, self.turns, self.stage = list(opening or []), [], None
        self.n_open = len(self.items)
        c = scene["chapter"]
        self.system, self.situation, self.cards = {}, {}, {}
        for cid in scene["cast"]:
            edit = self.cfg["edits"].get(cid)
            card = self.cards[cid] = card_text(cards, c, cid, tuple(edit) if edit else None)
            if self.cfg["baseline"] == "coser":
                card += f"\n\nThe plot of the scene you act: {scene['plot']}"
            self.system[cid] = self._name(SYSTEM.format(card=card))
            self.situation[cid] = " ".join(x for x in (scene["circumstances"].get(cid, ""),
                                                       scene.get("private", {}).get(cid, "")) if x)
        names = {cast.name(cid): cid for cid in cast.characters}
        if table:
            names |= {pseudonyms.apply(n, table): cid for n, cid in list(names.items())}
        self.ids = {n.lower(): cid for n, cid in names.items()}
        # the words of the names of the scene's characters do not tell two moves apart (T7)
        self.name_words = {w for n, cid in names.items() if cid in scene["cast"] for w in re.findall(r"[a-z]+", n.lower())}
        self.steer = {}
        for cid, spec in self.cfg.get("steer", {}).items():
            v = torch.load(spec["path"])
            self.steer[cid] = (v["layer"], spec["alpha"] * v["vector"])

    def use(self, cid):
        """Activate the adapters (and the steering vector) of a character; None = T3 observer."""
        self.agent.activate(self.adapters(cid) if cid else [])
        layer, vec = self.steer.get(cid, (None, None))
        self.agent.set_steering(layer, vec)

    def _name(self, text):
        return pseudonyms.apply(text, self.table) if self.table else text

    def adapters(self, cid):
        mode = self.cfg["mode"]
        if mode == "prompted":
            return []
        if mode == "soul" and cid in self.cfg["souls"]:
            return ["world", f"soul_{cid}"]
        if mode == "soul" and "cast" in self.cfg["souls"]:
            return ["world", "soul_cast"]
        return ["world"]

    def messages(self, cid, note=None):
        """note: the direction of the director, the last thought of the character for this move."""
        return [{"role": "system", "content": self.system[cid]},
                {"role": "user", "content": self._name(self._user(cid, note))}]

    def _user(self, cid, note=None, plan=True):
        others = [x for x in self.sc["cast"] if x != cid]
        lines = [ln for ln in (render(it, cid, self.cast) for it in self.items) if ln][-(HISTORY - 1):]
        # the memory query names the objects of the scene state (the pearls, the land)
        query = " ".join([self.situation[cid], *self.state, *lines[-QUERY_LINES:]])
        thoughts = [x for x in (self.plans.get(cid) if plan else None, note) if x]
        lines = [f"(your thought) {self.situation[cid]}", *lines] + [f"(your thought) {x}" for x in thoughts]
        known = ""
        tools = self.cfg["tools"]  # on, off, t1 (relations only) or t2 (memories only)
        if tools != "off" and self.tools:
            known = self.tools.describe(cid, others if tools != "t2" else [], query, k=0 if tools == "t1" else 3)
        if self.cfg["t3_to_agents"] and self.stage:
            known = (known + "\n" if known else "") + f"Stage of the scene: {self.stage}"
        if self.cfg["t5"]:
            texts = self.sc["state_texts"]
            where = " ".join(texts["state"][k][v] for k, v in self.shown.items())
            known = (known + "\n" if known else "") + f"Where you are: {texts['place']}\nWhat is where: {where}"
        return USER.format(present=", ".join(self.cast.name(x) for x in [cid, *others]),
                           tools=known + "\n" if known else "",
                           lines="\n".join(lines))

    def _private(self, cid):
        """The user message of the character without its last plan and question (T4, T8)."""
        return self._user(cid, plan=False).rsplit("\n\nWhat do you do now?", 1)[0]

    def _plan(self, cid, seed):
        """T4: what the character wants now and what it will do next, thought by the base model as
        the character."""
        name = self.cast.name(cid)
        self.use(None)
        raw = self.agent.generate([{"role": "user", "content": self._name(PLAN_PROMPT.format(
            name=name, card=self.cards[cid], user=self._private(cid)))}], max_new_tokens=50,
            temperature=self.cfg["temperature"], seed=seed)
        text = re.sub(r"\s+", " ", raw).strip().strip('"').strip()
        if text and text[-1] not in ".!?" and re.search(r"[.!?]", text):
            text = text[:max(text.rfind(x) for x in ".!?") + 1]  # cut off by the token limit
        if text:
            self.plans[cid] = text
        return {"raw": raw, "plan": text or None}

    def _off_scene(self, move, intend):
        """T6: the move is addressed to a character who is not in the scene, or it is an action
        that takes its character away from the place or time of the scene or names a character
        who is not in the scene without only thinking of it."""
        to = move.get("to") if isinstance(move.get("to"), list) else []
        absent = [n for n in to if self.ids.get(str(n).lower().strip(), "") not in ("", *self.sc["cast"])]
        text = str(move.get("action") or "") if intend == "act" else ""
        if text and not THINK.search(text):
            absent += [c for _, _, c in self.cast.mentions(text, self.sc["chapter"]) if c not in self.sc["cast"]]
        return bool(AWAY.search(text) or absent)

    def _rejected(self, actor, move, intend):
        """The scene check (T6) or the consistency check (T7) draws the move again."""
        if self.cfg["t6"] and self._off_scene(move, intend):
            return True
        if not self.cfg["t7"] or intend == "silent":
            return False
        if intend == "act" and impossible(self.sc["id"], self.shown, actor, self.cast.name(actor), "act",
                                          str(move.get("action") or ""), JEWEL):
            return True
        kinds = ("act",) if intend == "act" else ("speak", "gist")
        text = move.get("action") if intend == "act" else move.get("speech") or move.get("gist")
        earlier = [t["item"]["text"] for t in self.turns if t["actor"] == actor and t["item"]["kind"] in kinds]
        return repeats(str(text or ""), earlier, self.name_words)

    @staticmethod
    def _option(move, intend):
        to = move.get("to") if isinstance(move.get("to"), list) else []
        to = f" to {', '.join(map(str, to))}" if to else ""
        if intend == "act":
            return f"(do{to}) {move.get('action')}"
        if str(move.get("speech") or "").strip():
            return f'(say{to}) "{move["speech"]}"'
        return f"(say{to}) {move.get('gist')}"

    def _choose(self, actor, moves, intend):
        """T8: the index of the move the character chooses, and the probability of every number."""
        plan = self.plans.get(actor)
        options = "\n".join(f"{i + 1}. {self._option(m, intend)}" for i, m in enumerate(moves))
        self.use(None)
        p = self.agent.choice_probs([{"role": "user", "content": self._name(CHOICE_PROMPT.format(
            card=self.cards[actor], user=self._private(actor), name=self.cast.name(actor),
            want=f" What you want now: {plan}" if plan else "", options=options))}], len(moves))
        return max(range(len(moves)), key=lambda i: p[i]), p

    def _to_ids(self, names):
        out = []
        for n in names if isinstance(names, list) else []:
            cid = self.ids.get(str(n).lower().strip())
            if cid and cid in self.sc["cast"]:
                out.append(cid)
        return out

    def _apply_rules(self, cid, text):
        for actor, pat, key, value in TRANSFER.get(self.sc["id"], []):
            if cid == actor and pat.search(text) and self.state.get(key) != value:
                self.state[key] = value
                return {key: value}
        return {}

    def _observe(self):
        lines = [ln for ln in (render(it, None, self.cast) for it in self.items[self.n_open:]) if ln]
        self.use(None)
        text = self.agent.generate([{"role": "user", "content": self._name(T3_PROMPT.format(lines="\n".join(lines)))}],
                                   max_new_tokens=40)
        try:
            obs = json.loads(text[text.find("{"):text.rfind("}") + 1])
        except ValueError:
            obs = {}
        stage = obs.get("stage") if obs.get("stage") in STAGES else self.stage
        return stage, bool(obs.get("ended"))

    def _direct(self, turn):
        """(character, direction or None, record) of the director or the next-speaker predictor;
        the character is None when the answer names no character of the scene."""
        lines = "\n".join([ln for ln in (render(it, None, self.cast) for it in self.items) if ln][-HISTORY:])
        names = ", ".join(self.cast.name(c) for c in self.sc["cast"])
        last = [t["actor"] for t in self.turns[-2:]]
        if self.cfg["baseline"] == "director":
            # the director knows every card and what every character has in mind at the start
            prompt = DIRECTOR_PROMPT.format(
                cards="\n".join(f"{self.cast.name(c)}: {self.cards[c]} At the start of the scene: {self.situation[c]}"
                                for c in self.sc["cast"]),
                premise=self.sc["premise"], lines=lines, left=self.cfg["max_turns"] - turn, names=names,
                last=", ".join(self.cast.name(c) for c in last) or "nobody yet")
        else:
            prompt = PREDICTOR_PROMPT.format(names=names, plot=self.sc["plot"], lines=lines)
        self.use(None)
        # sampled like the moves of the characters (greedy decoding repeated the same direction)
        text = self.agent.generate([{"role": "user", "content": self._name(prompt)}], max_new_tokens=80,
                                   temperature=self.cfg["temperature"], seed=self.seed * 1000 + turn * 10 + 7)
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[text.find("{"):])
        except ValueError:
            obj = {}
        obj = obj if isinstance(obj, dict) else {}
        # a name is matched without its article, and the character id is accepted as well
        key = re.sub(r"^the\s+", "", str(obj.get("next") or "").lower().replace("_", " ").strip())
        actor = next((c for c in self.sc["cast"] if key in (c.replace("_", " "),
                      re.sub(r"^the\s+", "", self.cast.name(c).lower()))), None)
        rec = {"raw": text, "actor": actor}
        if self.cfg["baseline"] == "director" and actor and len(last) == 2 and last == [actor, actor]:
            rec["rejected"], actor = actor, None  # the director chose the character of the last two moves
        note = None
        if actor and self.cfg["baseline"] == "director":
            note = str(obj.get("direction") or "").strip() or None
        # a direction given before word for word is not shown again (the director repeated itself
        # in the smoke tests; showing it its last directions made it copy them)
        if note and note in {t["director"].get("direction") for t in self.turns}:
            rec["repeated"], note = note, None
        return actor, note, rec | {"actor": actor, "direction": note}

    def step(self, turn):
        pick, note, directed, plan = None, None, None, None
        if self.cfg["t4"] and turn == 0:  # every character plans before the first move
            self.plans0 = [self._plan(cid, self.seed * 1000 + 900 + i) for i, cid in enumerate(self.sc["cast"])]
        if self.cfg["baseline"]:
            pick, note, directed = self._direct(turn)
        # without a choice of the director the urgency draw chooses, among the characters other
        # than the one the director was not allowed to choose
        cands = [pick] if pick else [c for c in self.sc["cast"] if c != (directed or {}).get("rejected")]
        if self.cfg["t9"] and not pick and self.turns:  # the characters spoken to answer (T9)
            last = self.turns[-1]["item"]
            to = [c for c in last["to"] if c != last["actor"] and c in cands]
            if last["kind"] in ("speak", "gist") and to:
                cands = to
        probs = {}
        for cid in cands:
            self.use(cid)
            probs[cid] = self.agent.intend_probs(self.messages(cid, note))
        urg = {cid: 1 - p["wait"] for cid, p in probs.items()}
        total = sum(urg.values())
        r, acc, actor = self.rng.random() * total, 0.0, cands[-1]
        for cid in [] if pick else cands:
            acc += urg[cid]
            if r < acc:
                actor = cid
                break
        p = probs[actor]
        if self.cfg["t4"] and turn > 0:  # the character who takes the floor plans again
            plan = self._plan(actor, self.seed * 1000 + turn * 10 + 8)
            self.use(actor)
            p = self.agent.intend_probs(self.messages(actor, note))
        p = {k: v for k, v in p.items() if k != "wait"}
        z = sum(p.values())
        r, acc, intend = self.rng.random() * z, 0.0, "speak"
        for k in ("speak", "act", "silent"):
            acc += p[k]
            if r < acc:
                intend = k
                break
        self.use(actor)
        msgs, move, raw, off, cands, choice = self.messages(actor, note), None, "", [], [], None
        want = 3 if self.cfg["t8"] and intend != "silent" else 1  # candidates of the choice (T8)
        draws = 5 if want > 1 else 3
        for attempt in range(draws):
            raw = self.agent.generate(msgs, max_new_tokens=120, temperature=self.cfg["temperature"],
                                      seed=self.seed * 1000 + turn * 10 + attempt, prefix='{"intend": "' + intend + '"')
            move = parse_move(raw)
            fields = {"speak": ("speech", "gist"), "act": ("action",), "silent": ()}[intend]
            if move and move["intend"] == intend and \
                    (not fields or any(str(move.get(k) or "").strip() for k in fields)):
                # the last draw is kept when no move passed the checks (with T7 the character then
                # says nothing)
                if (attempt < draws - 1 or cands or self.cfg["t7"]) and self._rejected(actor, move, intend):
                    off.append(raw)  # drawn again
                    continue
                cands.append((move, raw))
                if len(cands) == want:
                    break
        move = None
        if len(cands) > 1:
            i, chance = self._choose(actor, [m for m, _ in cands], intend)
            choice = {"candidates": [r for _, r in cands], "probs": chance, "chosen": i}
            move, raw = cands[i]
        elif cands:
            move, raw = cands[0]
        dropped = intend if self.cfg["t7"] and not move and off else None
        if not move:
            move, intend = {"intend": "silent", "to": []}, "silent"
        to = self._to_ids(move.get("to"))
        if intend == "speak" and move.get("speech"):
            item = {"kind": "speak", "text": str(move["speech"])}
        elif intend == "speak":
            item = {"kind": "gist", "text": str(move.get("gist") or move.get("speech") or "")}
        elif intend == "act":
            item = {"kind": "act", "text": str(move.get("action") or "")}
        else:
            item = {"kind": "silent", "text": f"{self.cast.name(actor)} says nothing."}
        item |= {"actor": actor, "chorus": None, "to": to}
        self.items.append(item)
        change = self._apply_rules(actor, item["text"])
        rec = {"turn": turn, "probs": probs, "urgency": urg, "actor": actor, "intend": intend, "move": move,
               "raw": raw, "item": item, "state": dict(self.state), "state_change": change}
        if directed:
            rec["director"] = directed
        if off:
            rec["off_scene"] = off
        if dropped:
            rec["dropped"] = dropped  # every draw failed the checks
        if choice:
            rec["choice"] = choice
        if self.cfg["t4"]:
            rec["plan"] = (plan or {}) | {"shown": self.plans.get(actor), "intend_probs": p}
        if self.cfg["t5"]:
            self.shown = after_move(self.sc["id"], self.shown, actor, item["kind"], item["text"],
                                    JEWEL if self.cfg["t7"] else PEARL)
            rec["shown_state"] = dict(self.shown)
        if self.cfg["t3_to_agents"]:
            self.stage, ended = self._observe()
            rec |= {"stage": self.stage, "ended": ended}
        self.turns.append(rec)
        return rec

    def run(self):
        for turn in range(self.cfg["max_turns"]):
            self.step(turn)
        lines = [ln for ln in (render(it, None, self.cast) for it in self.items[self.n_open:]) if ln]
        out = {"scene": self.sc["id"], "seed": self.seed, "config": self.cfg, "turns": self.turns,
               "opening_lines": self.n_open, "final_state": self.state,
               "transcript": [self._name(ln) for ln in lines]}
        if self.cfg["t4"]:
            out["first_plans"] = dict(zip(self.sc["cast"], self.plans0))
        return out


__all__ = ["SceneRun", "INTENDS", "STAGES"]
