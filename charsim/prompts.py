"""Prompt templates of the corpus and tool extraction and of the memorization probe."""

RELATIONS = ("parent, child, spouse, concubine, sibling, kin, master, servant, employer, employee, "
             "neighbour, creditor, debtor, affection, desire, trust, distrust, resentment, fear, "
             "contempt, gratitude, duty, rivalry")

ATTRIBUTE = """You are annotating the quotations of a novel. Below is a passage from chapter {chapter}. Every quotation to annotate is marked by an id in double square brackets, such as [[q3]], placed right before its opening quotation mark. Unmarked quotations are context only.

Characters who may appear in this chapter (id: name; description; how the text refers to them):
{cast}

For every marked quotation, give the id of the character who speaks it and the ids of the characters it is addressed to. Use a list of ids as speaker when several characters speak it together, "other" for a speaker who is not in the list, and "none" when the quoted words are not spoken by anyone in the scene (a written text, a thought, a word that is only heard or imagined).
Return only JSON with one entry for every marked id ({ids}), of the form {{"q3": {{"speaker": "...", "to": ["..."]}}}}.

Passage:
{passage}"""

BEATS = """You are annotating a passage from chapter {chapter} of a novel for a simulation of its characters.

Characters (id: name; description; how the text refers to them):
{cast}

List, in order of appearance, every beat of these characters that is not a quotation. The beat types are
"action" (a character does something that others can see, such as giving, taking, working, leaving, weeping),
"reported_speech" (the narrator reports what a character says without quoting the words),
"silence" (a character is addressed or expected to answer but says nothing, or the text says the character is silent),
"thought" (the narrator renders what a character thinks or feels).
For every beat give "type", "character" (an id), "to" (ids of the characters it is directed at, possibly empty), "evidence" (an exact substring of the passage of at most 25 words, copied character for character) and "summary" (at most 15 words, present tense).
Return only JSON of the form {{"beats": [...]}}.

Passage:
{passage}"""

EVENTS = """You are building a database of the narrative events of a novel. A narrative event is something that happens in the story world and changes the situation of at least one character. Below is a passage from chapter {chapter}.

Characters (id: name; description; how the text refers to them):
{cast}

List the narrative events of the passage in the order in which they happen. For every event give "summary" (at most 20 words, past tense), "participants" (ids), "witnesses" (ids of the characters who are present and perceive the event, including the participants), "place" (a short phrase, or an empty string), "time" (a time expression of the text, or an empty string), "objects" (important objects such as land, silver or pearls), "causes" (positions, counted from 0, of earlier events in your list that cause this event) and "evidence" (an exact substring of the passage of at most 30 words, copied character for character).
Return only JSON of the form {{"events": [...]}}.

Passage:
{passage}"""

RELATIONS_PROMPT = """You are building a graph of the relationships between the characters of a novel. Below is a passage from chapter {chapter}.

Characters (id: name; description; how the text refers to them):
{cast}

List every relationship between two of these characters that the passage states or clearly shows. For every relationship give "source" and "target" (ids), "relation" (one of: {relations}), "polarity" (an integer from -2, very negative, to 2, very positive, seen from the source), "description" (at most 15 words) and "evidence" (an exact substring of the passage of at most 30 words, copied character for character). Include only relationships that this passage supports.
Return only JSON of the form {{"relations": [...]}}.

Passage:
{passage}"""

PROBE_MC = """{intro}{setup}

What happens next? Choose the option that describes it.
{options}
Return only JSON of the form {{"answer": "A"}}."""

PROBE_CONTEXT_MC = """Here is a passage from {source}.

{passage}

What happens next in the {kind}? Choose the option that describes it.
{options}
Return only JSON of the form {{"answer": "A"}}."""

PROBE_VERBATIM = """Continue the following passage from Pearl S. Buck's novel The Good Earth with its exact original words. Write only the continuation, about forty words.

{prefix}"""

PROBE_WHO = "Who is {name} in Pearl S. Buck's novel The Good Earth? Answer in one sentence."


def cast_lines(cast, cids, chapter):
    lines = []
    for cid in cids:
        c = cast.characters[cid]
        refs = sorted({a for a, lo, hi in c["aliases"] if lo <= chapter <= hi})
        lines.append(f"- {cid}: {c['name']}; {c['description']}; " + ", ".join(refs))
    return "\n".join(lines)
