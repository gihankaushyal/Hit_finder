---
name: pipeline-check
description: Run full preprocessing pipeline smoke tests on all four detectors (AGIPD, JUNGFRAU, ePix10k, Eiger4M) to catch regressions after any preprocessing change
disable-model-invocation: true
---

Run the following commands in sequence and report pass/fail per detector:

```bash
pytest tests/test_pipeline.py tests/test_normalize.py tests/test_hitfinders.py tests/test_augmentation.py tests/test_geometry_assembly.py -v 2>&1 | tail -30
```

Then run the shape smoke test:
```bash
python scripts/smoke_test_detector_shapes.py
```

Report results as:
- AGIPD: PASS / FAIL
- JUNGFRAU 4M: PASS / FAIL
- ePix10k: PASS / FAIL
- Eiger4M: PASS / FAIL

If any detector fails:
1. Extract the traceback from the pytest output
2. Invoke `/superpowers:systematic-debugging` with the failing test name and traceback
3. Check CLAUDE.md confirmed preprocessing parameters: lcn_window=9, lcn_eps=1e-2 variance-form, EDGE_EROSION_PX=2, GCN→crop→rot90/flip→LCN→cutout order
