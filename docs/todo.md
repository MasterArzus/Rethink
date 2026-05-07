# Revision To-Do

## Experiment Code

- [x] Build staged K=8 dataset generator.
- [x] Implement stage-aware prompts.
- [x] Implement checker for base, taboo, JSON, tone proxy, and dynamic constraints.
- [x] Implement `reflexion.py`.
- [x] Implement `autoLR.py`.
- [x] Implement `constraint_decoding.py`.
- [x] Implement `chat.py`.
- [x] Implement `steer.py`.
- [x] Implement `steer_lite.py`.
- [x] Implement unified `run_exp.sh`.
- [x] Implement `score.py`.

## Next Experimental Pass

- [ ] Expand `data/staged_cases.json` from the current pilot-sized set to the final paper-sized set.
- [ ] Decide whether K=4 dynamic checks should use heuristic-only, LLM judge, or both.
- [ ] Run all methods on the target model list.
- [ ] Aggregate outputs into paper tables.
- [ ] Compare chat, steer, and steer-lite under the same simulated inspection-time assumptions.

## App

- [x] Add fast generation path to avoid full hidden-state/SOS precompute on every generation.
- [x] Add explicit model unload and CUDA cache cleanup.
- [ ] Add a dedicated Revision Staged dataset loader in the UI if human-study tasks move fully to the new dataset.

