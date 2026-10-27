"""Rule-based outcome of the scene objects, recomputed from the moves of a session.

Only actions count, sentence by sentence. O-lan gives the pearls when a sentence has a verb
of giving, the pearls and Wang Lung as the receiver ("places the pearls in his hand", "holds
them out to Wang Lung", "offers Wang Lung her pearls"). Wang Lung takes the pearls when a
sentence of his has a verb of taking and the pearls and does not have O-lan or "she" as its
subject. A sentence of O-lan that keeps the pearls (puts them back, hides, clutches, shields,
refuses) and a sentence of Wang Lung that gives them back never count. The land of chapter 34 is not sold within the scene, so it has no rule."""

import re

SENT = re.compile(r"(?<=[.!?;])\s+")
PEARL = re.compile(r"\bpearls?\b", re.I)
# putting the pearls back, keeping or hiding them (O-lan); giving them back (Wang Lung)
PUT_BACK = r"\b(puts?|places?|placed|tucks?|tucked|slips?|slipped|returns?|returned|wraps?|wrapped)\b[^.]*\bback\b" \
           r"|\bback (into|in|between|inside|under)\b"
KEEP = {"o_lan": re.compile(PUT_BACK + r"|\b(refuses?|refused|hides?|hid|hidden|keeps?|kept|clutch(es|ed|ing)?|"
                            r"shield(s|ed|ing)?|covers?|covering|away from|tight(ly|ens|ening)?)\b", re.I),
        "wang_lung": re.compile(PUT_BACK + r"|\b(refuses?|refused|gives? (\w+ )?back|lets? her keep|hesitat\w*)\b"
                                r"|\b(hands?|gives?|returns?|pass(es)?) (\w+ ){0,3}to (o-lan|su-mei|her)\b"
                                r"|\b(reaches|reach|tries|try|attempts?|wants?) (out )?to take\b", re.I)}
GIVE = re.compile(r"\b(give|gives|gave|giving|hands|handed|handing|pass(es|ed)?|places?|placed|placing|puts?|"
                  r"putting|press(es|ed)?|lays?|laid|drops?|dropped|offers?|offered|offering|thrusts?|"
                  r"holds? (\w+ )?out|held (\w+ )?out|holding (\w+ )?out)\b", re.I)
TO_HIM = re.compile(r"\b(to (him|wang lung|her husband|chen fu)|in(to)? (his|wang lung's|chen fu's|her husband's) "
                    r"(\w+ )?(hand|hands|palm|lap)|(him|wang lung|chen fu|her husband) (the |her |two |those |both |"
                    r"small |white |little )*pearls)\b", re.I)
TAKE = re.compile(r"\b(takes?|took|taking|snatch(es|ed)?|grabs?|grabbed|seizes?|seized|plucks?|plucked|"
                  r"accepts?|accepted|receives?|received|pockets?|pocketed)\b", re.I)
OTHER_SUBJECT = re.compile(r"^\s*(o-lan|su-mei|she|her)\b", re.I)
# a sentence of Wang Lung about pearls he already has, or about taking O-lan somewhere: not a
# transfer in the measure (strict rule, 2026-10-27, after the counted moves of the third follow-up
# were read: "takes a pearl from his sleeve", "takes out the pearls and shows them to O-lan",
# "takes O-lan to their hut and gives her the pearls"); the state tool T5 used the rule without
# it during the runs
HIS_OWN = re.compile(r"\b(from|out of) his (own )?(sleeve|pocket|robe|coat|girdle|belt|bosom|breast|purse|pouch|"
                     r"bag|clothes|garment|shirt|jacket|sash)s?\b"
                     r"|\b(takes?|took|taking) out\b(?![^.]*\b(from|out of) (her|o-lan)\b)"
                     r"|\b(takes?|took|taking|leads?|led) (o-lan|her|su-mei) (by the hand )?"
                     r"(to|into|home|away|back|inside|out)\b"
                     r"|\b(gives?|gave|hands?|handed|offers?|offered|shows?|showed|presents?|presented) "
                     r"(o-lan|her|su-mei) (\w+ ){0,3}(pearls?|jewels?)\b"
                     r"|\b(shows?|showed|presents?|presented|offers?|offered) (\w+ ){0,3}to (o-lan|her|su-mei)\b", re.I)
# a sentence of O-lan in which the giving is only feared, intended or nearly done ("You feel the
# weight of the pearls but fear giving them to Wang Lung"): not a transfer in the strict rule
NOT_DONE = re.compile(r"\b(fears?|feared|afraid|about to|almost|nearly|thinks? (of|about)|considers?|wonders?|"
                      r"wants? to|wishes|would|could|might|ready to|tempted)\b[^.]*\b(giv|hand|pass|offer|plac|put|"
                      r"press|lay|laid|drop|thrust|hold|held)", re.I)


def pearls_given(actor, kind, text, obj=PEARL, strict=False):
    if kind != "act" or actor not in KEEP:
        return False
    for s in SENT.split(text or ""):
        if not obj.search(s) or KEEP[actor].search(s):
            continue
        if actor == "o_lan" and GIVE.search(s) and TO_HIM.search(s) and not (strict and NOT_DONE.search(s)):
            return True
        if actor == "wang_lung" and TAKE.search(s) and not OTHER_SUBJECT.search(s) and \
                not (strict and HIS_OWN.search(s)):
            return True
    return False


def outcome(scene_id, turns, strict=True):
    """{"changed": bool, "turn": first turn of the change or None} for the scene's object.
    strict=False is the rule before 2026-10-27 (without HIS_OWN)."""
    if scene_id != "pearls":
        return {"changed": None, "turn": None}
    for t in turns:
        it = t["item"]
        if pearls_given(it["actor"], it["kind"], it["text"], strict=strict):
            return {"changed": True, "turn": t["turn"]}
    return {"changed": False, "turn": None}


WEAR = re.compile(r"\b(wears?|wore|wearing)\b", re.I)
GIVE_BACK = re.compile(r"\bgives? (\w+ ){0,3}back\b|\blets? her keep\b"
                       r"|\b(hands?|gives?|returns?|pass(es)?) (\w+ ){0,3}to (o-lan|her)\b", re.I)


def undone(scene_id, turns, strict=True):
    """After the first transfer of the pearls, does a later action undo or contradict it (O-lan
    keeps, hides or wears the pearls, or Wang Lung gives them back)? None without a transfer.
    Descriptive, added on 2026-10-26 after the director sessions were read."""
    first = outcome(scene_id, turns, strict)["turn"]
    if first is None:
        return None
    for t in turns:
        it = t["item"]
        if t["turn"] <= first or it["kind"] != "act":
            continue
        for s in SENT.split(it["text"] or ""):
            if not PEARL.search(s):
                continue
            if it["actor"] == "o_lan" and (KEEP["o_lan"].search(s) or WEAR.search(s)):
                return True
            if it["actor"] == "wang_lung" and GIVE_BACK.search(s):
                return True
    return False


AWAY = re.compile(r"\b(leaves?|walks? (away|off|out|home|back)|goes (away|home|out|back|off|in(to)?|to the)|"
                  r"returns? (home|to (the|his|her) (house|town|tea|room|courts?))|enters? (the|his|her)|"
                  r"falls? asleep|sleeps?|lies? down|goes to bed|the next (day|morning)|that night)\b", re.I)


def away(turns):
    """Does an action take a character away from the place of the scene (leaving, going or
    returning elsewhere, entering a room) or the scene to another time (sleep, the next day)?
    Descriptive, added on 2026-10-26 after the sessions were read (45 random hits of all
    sessions read: at most one doubtful). Reported speech does not count ("O-lan threatens to
    leave Wang Lung")."""
    return any(t["item"]["kind"] == "act" and AWAY.search(t["item"]["text"] or "") for t in turns)


# verbs of an action that buys something, or that hides, keeps, holds or wears an object (T7)
BUY = r"(buys?|bought|buying|purchases?|purchased)"
HOLD = r"(hides?|hid|keeps?|kept|holds?|held|wears?|wore|clutch(es|ed)?|tucks?|tucked|wraps?|wrapped|puts?|" \
       r"removes?|removed|slips?|slipped|buries|buried|fingers|strokes|stroked|" \
       r"gives?|gave|hands?|handed|offers?|offered|returns?|returned|shows?|showed|presents?|presented|passes|passed)"
# the pearls as the characters also call them ("These jewels you have kept and hidden", "Wang Lung
# takes the jewels from O-lan"): the state of T5 follows both words when T7 is on (the first
# session of the third follow-up missed the transfer); the descriptive measures keep PEARL
JEWEL = re.compile(r"\b(pearls?|jewels?)\b", re.I)


def impossible(scene_id, state, actor, name, kind, text, obj=PEARL):
    """Is an action impossible for its character in the state shown by T5 (the consistency check
    T7, third follow-up of 2026-10-26)? A sentence whose own verb (with the name of the character
    or no subject before it) buys something, since no scene takes place at a market
    ("Wang Lung buys a gold pin", "buys coffin"), or hides, keeps, holds, wears, gives or shows the
    pearls while the state shows them with another character ("O-lan hides the pearls once more"
    after she gave them; "Wang Lung gives two pearls to O-lan" while she has them, added with the
    other changes of T7 on 2026-10-27). Added after the sessions of the second follow-up were read."""
    if kind != "act":
        return False
    subject = rf"^\s*({re.escape(name)}\s+)?(\w+ly\s+|again\s+|then\s+)*"
    for s in SENT.split(text or ""):
        if re.search(subject + BUY + r"\b", s, re.I):
            return True
        if scene_id == "pearls" and state.get("pearls") != actor and obj.search(s) and \
                re.search(subject + HOLD + r"\b", s, re.I):
            return True
    return False


def after_move(scene_id, state, actor, kind, text, obj=PEARL):
    """The state after one move, as the state tool T5 shows it to the characters (follow-up of
    2026-10-26): the pearls pass to Wang Lung by the rule of outcome() before 2026-10-27 (strict=False,
    kept so that the runs can be reproduced), and back to O-lan when an action of his gives them back."""
    if scene_id != "pearls":
        return state
    if state["pearls"] == "o_lan" and pearls_given(actor, kind, text, obj):
        return state | {"pearls": "wang_lung"}
    if state["pearls"] == "wang_lung" and actor == "wang_lung" and kind == "act" and \
            any(obj.search(s) and GIVE_BACK.search(s) for s in SENT.split(text or "")):
        return state | {"pearls": "o_lan"}
    return state


# words that do not tell two moves apart (T7 repetition): function words, "again" and the like
STOP = set("a an the and or but of to in on at for from with into onto upon by as is are was were be been am "
           "do does did has have had will would shall should can could may might must not no nor so then than "
           "this that these those there here it its he him his she her hers they them their i me my mine you "
           "your we us our again once more still now just only very too also all what who whom which "
           "between under over inside behind through about around before after against among".split())


def words(text, names):
    """The words of a move that tell it apart: lower case, without function words and the words of
    the characters' names, each cut to its first four letters (hides, hide -> hide; pearls -> pear)."""
    out = set()
    for w in re.findall(r"[a-z]+", (text or "").lower()):
        if w not in STOP and w not in names:
            out.add(w[:4])
    return out


def repeats(text, earlier, names, limit=0.7):
    """Does a move repeat one of the earlier moves of the same kind of its character (T7)? Its
    words overlap those of an earlier move by at least `limit` (Jaccard), as "O-lan hides the pearls
    once again" after "O-lan hides the pearls" (but not after "O-lan hides the pearls between her
    breasts", nor "gives the pearls" after "almost gives the pearls"). Added after the first session of the
    third follow-up, where O-lan hid the pearls in eight of her ten moves."""
    new = words(text, names)
    if not new:
        return False
    for e in earlier:
        old = words(e, names)
        if old and len(new & old) / len(new | old) >= limit:
            return True
    return False
