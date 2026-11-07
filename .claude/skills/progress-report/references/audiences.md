# The three audiences

Each report renders the same verified facts for three readers. Defaults below; `reports/INSTRUCTIONS.md` (written in the first-run interview) overrides them with real names and preferences.

| | A: domain PI | B: domain expert, moderate AI | C: AI expert, light domain |
|---|---|---|---|
| Knows | Why hit finding is hard: sparse hits, detector differences, beamtime cost | The problem deeply; concepts of supervised vs self-supervised learning, train/test shift | Modern ML deeply; what a hit is and why detectors differ only roughly |
| Wants to know | Does it work on the detectors we use? What does it change at the beamline? What do you need from me? | Why this approach, what the numbers mean, where it fails and why | Exact setup, metrics, ablations, seeds, confounds, what is not yet shown |
| Lead with | The outcome in beamtime terms (fewer missed hits, which detector is ready) | Method idea + result in plain metrics | The experiment, the controls and the effect size |
| Avoid | Architecture names, loss functions, "AP" without a gloss | Unexplained ML jargon (logits, layer decay); unexplained crystallography is fine | Re-explaining ML basics; long domain tutorials |
| Figures | One picture of the problem; one bar per detector (works / does not yet) | Per-detector results with plain axis labels; a method sketch | Tables with mean +- sample std, per-fold values, curves, ablation grids |
| Length | 6-8 slides + 1 page | 8-10 slides + 1-2 pages | 10-14 slides + 2 pages, appendix allowed |

## Rules common to all
- Same numbers everywhere. If A shows "0.97", C's table contains 0.97 with its seed count and fold.
- Explain a term the first time it appears using `glossary.md` wording, then keep to it.
- State what is established, what is suggestive, and what is untested. Use the confidence tags from `facts.md`.
- End with decisions or help needed from that audience specifically.

## Project vocabulary (starting glossary, plain language)
- **Hit / non-hit:** a detector image that contains diffraction from a crystal vs one that does not.
- **Detector-agnostic:** one model that works on AGIPD, JUNGFRAU, ePix10k and Eiger4M without retraining per detector.
- **Leave-one-detector-out:** train on three detectors, test on the fourth, so the test detector was never seen.
- **AP (average precision):** how well hits rank above non-hits, 1.0 perfect; threshold-free.
- **Track 1 / Track 2:** supervised ResNet18 baseline / model pretrained on unlabeled frames (self-supervised ViT), then fine-tuned.
