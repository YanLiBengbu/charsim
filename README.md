# CharSim

Archived on Zenodo: https://doi.org/10.5281/zenodo.23696765 (release v1.0.0)

Code, prompts, simulated scenes and analysis scripts for the paper

> Lili Zhao and Hui Shi. *Character determines incident: director-free language model agents for testing fictional characters.* Manuscript submitted to PeerJ Computer Science.

CharSim plays a scene of a novel with one fine-tuned language model per main character and without a director. Every character model is trained only on the chapters before the scene. The character with the highest urgency to act takes the floor. Six character tools let a character plan what it wants (T4), see the place and the holder of the scene object (T5), stay in the scene (T6), avoid impossible or repeated moves (T7), choose among candidate moves (T8) and answer when it is addressed (T9). The study simulates two scenes of Pearl S. Buck's *The Good Earth* (chapter 19, the pearls; chapter 34, the land) and compares the endings with the chapters.

## Repository layout

```
charsim/                  framework package
  agent.py                4-bit base model with a story-world adapter and soul adapters; intention scores and urgency
  scene.py                director-free scene, director and known-plot baselines, character tools T4 to T9
  state.py                rules of the scene object (transfer of the pearls), away, consistency check, repetition
  tools.py                relation tool T1 and event tool T2 at a chapter cutoff, from one character's view
  samples.py prompts.py   decision points and prompts of the character agents
  extract.py grounding.py LLM annotation of the novel and the evidence check of every annotated item
  stats.py                Krippendorff's alpha, bootstrap intervals, permutation tests, Holm correction
scripts/
  prepare_text.py         PDF copy of the novel -> data/text/chapters.json
  extract_quotes.py extract_llm.py merge_annotations.py   speakers, beats, events and relations
  build_training.py       training and held-out data of every scene; train_lora.py trains the adapters
  eval_fidelity.py        fidelity on the held-out chapters
  probe_memory.py         memorization probe of the base models
  persona_vectors.py      persona vectors of the personality edits (steering)
  simulate.py             simulated sessions of a scene under a grid of conditions
  judge.py                three local judges: ending class, outcome propositions, ratings
  analyze.py              summary of all sessions and the planned and exploratory tests
  personality.py decided_tests.py addressee_next.py   further analyses reported in the paper
  make_figures.py make_tables.py make_diagrams.py     figures and tables of the paper
  app.py screenshot_ui.py writer interface (Gradio) and its screenshot
  novel_overlap.py        finds and replaces copied passages of the novel in released files
  run_step1.sh ... run_step7.sh   the pipeline as run for the paper (gpu_gate.sh shares the GPU with other jobs)
config/                   cast, character cards, rich cards, scenes with the pre-registered ending classes,
                          outcome propositions and predicted effects of the personality edits
                          (frozen: scenes.frozen.sha256), written control endings, baselines, texts of T5
results/
  sim/                    all simulated sessions: results/sim/<base>/<scene>/<condition>/seed<k>.json
  judge/                  the answers of the three judges for every session and control
  summary/                sessions.csv (one row per session), summary.json, summary_print.txt (all shares
                          and tests), personality.json, decided_tests.json, addressee_next.json, and the
                          summary with the transfer rule without its two exclusions
  fidelity/ probe/ annotation/ training/ steer/   fidelity, memorization probe, annotator agreement,
                          training logs and adapter configurations, persona vectors
data/checksums.json       SHA-256 of every data file derived from the novel (not distributed, see below)
```

## Requirements

* Python 3.11 with the packages in `requirements.txt` (simulation, judges, analysis) and `requirements-train.txt` (training). The paper used PyTorch 2.8.0, Transformers 4.57.1, PEFT 0.17.1, bitsandbytes 0.47.0 and Gradio 5.50.0.
* The 4-bit base models `Meta-Llama-3.1-8B-Instruct-bnb-4bit` and `Qwen3-8B-bnb-4bit` (bitsandbytes checkpoints of Llama 3.1 8B Instruct and Qwen3 8B) in `$MODELS_DIR`.
* [Ollama](https://ollama.com) 0.40.3 with `qwen2.5:14b`, `gemma3:12b` and `phi4:14b` (annotation), `phi4:14b`, `gemma3:12b` and `mistral-nemo:12b` (judges), and `llama3.1:8b` and `qwen3:8b` (memorization probe).
* One GPU with 16 GB of memory. The paper used an NVIDIA GeForce RTX 4070 Ti SUPER under Ubuntu 24.04.

## Reproducing the paper

The analyses, figures and tables can be recomputed from the released results without any model:

```
python scripts/analyze.py                 # results/summary/*, prints all shares and tests
python scripts/analyze.py --old-transfer-rule --out results/summary_rule_without_exclusions
python scripts/personality.py
python scripts/decided_tests.py
python scripts/make_figures.py --out paper && python scripts/make_tables.py --out paper
python scripts/make_diagrams.py --out paper
```

`analyze.py` reads the judges' answers in `results/judge/` and the sessions in `results/sim/`. `addressee_next.py` also reads the sessions.

The full pipeline needs a copy of the novel (see below) and runs in seven step scripts:

1. `run_step1.sh` annotates the novel with two local models, merges the annotations into the tools T1 and T2, and runs the memorization probe.
2. `run_step2.sh` builds the training data, trains the story-world and soul adapters of both base models (and of Qwen3 8B on the pseudonymized text) and evaluates the fidelity.
3. `run_step3.sh` runs the main design and the pseudonym control.
4. `run_step4.sh` runs the judges and their controls, the persona vectors, the personality edits and the ablations of the memory and structure tools.
5. `run_step5.sh` runs the director, known-plot and rich-card baselines.
6. `run_step6.sh` and `run_step7.sh` run the character tools and their ablations.

Every language-model call is seeded. The sessions use the seeds 100 to 119.

## Data and copyright

*The Good Earth* (1931) is still protected by copyright in the United States until the end of 2026. The repository therefore contains neither its text nor the files derived from it: the parsed chapters, the annotations with their evidence quotations, the event and relation stores, the training and held-out data, the response caches of the annotators and the judges, and the trained adapters. `data/checksums.json` gives the SHA-256 of each of these files. With a PDF copy of the novel at `data/source/the_good_earth.pdf`, step 1 and step 2 rebuild them, and the checksums show whether a rebuilt file matches the one used in the paper.

The simulated sessions are released in full, except for passages of 50 or more characters that a character model reproduced verbatim from the novel. These passages are replaced by `[GE-OMITTED:<hash>]`, where the hash is the start of the SHA-256 of the passage. `results/sim_omitted_passages.json` lists every replaced passage with its file, hash and length. The analyses in `results/summary/` were computed before the replacement. Short quotations of the novel remain in the configuration as anchors of the scenes and in the written control endings.

## License

The code is released under the MIT License (see `LICENSE`).

## Citation

See `CITATION.cff`.
