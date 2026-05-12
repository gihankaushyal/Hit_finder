# README Design Spec — Detector-Agnostic SFX Hitfinder
**Date:** 2026-05-12  
**Status:** Approved

---

## Goals

Produce a GitHub README that serves two audiences simultaneously:
- **Research peers** (crystallography, XFEL, ML-for-science communities)
- **Broader ML community** (engineers, students, open-source contributors)

Tone: professional/technical with a narrative opening — tells the scientific story before diving into code.

---

## Structure & Sections

### 1. Hero Banner
- SVG embedded directly in the README (no external image host dependency)
- Dark background, concentric diffraction rings, Bragg spots in blue (#58a6ff) and green (#3fb950)
- Beamstop shadow at centre
- Title and subtitle overlaid: "Detector-Agnostic SFX Hitfinder" / "Serial Femtosecond X-ray Crystallography · Fromme Lab · Arizona State University"

### 2. Badge Strip
Five shields.io badges, left-to-right:
- Python 3.11 (blue)
- PyTorch (orange/red)
- License: MIT (green)
- Phase: 1 — Proposal (purple)
- Institution: ASU Fromme Lab (maroon #8C1D40)

### 3. Hook Paragraph
One pull-quote style paragraph (left blue border) opening with:
> "Every pulse of an X-ray free-electron laser lasts just femtoseconds..."

Introduces: femtosecond pulses → protein crystal diffraction → <5% hit rate → hitfinding as the first bottleneck.

### 4. The Challenge
~2 sentences explaining detector heterogeneity: per-detector calibration, silent failure on cross-detector deployment, and the goal of a single detector-agnostic classifier.

### 5. The Approach
Two subsections:

**5a. Shared Preprocessing Pipeline**  
Inline ASCII-style pipeline diagram (rendered as styled HTML in mockup, plain text in actual README):
```
HDF5/CXI → Detector ID (metadata) → Reborn Geometry → GCN → LCN → 224×224
```
Key constraint noted: normalization (GCN→LCN) always precedes resize.

**5b. Two-Track Modeling**  
Side-by-side comparison table or diagram:
- Track 1: ResNet18→ResNet50, supervised, timm pretrained weights
- Track 2: ViT MAE encoder, self-supervised pretraining on unlabeled frames, classification head fine-tuned
- Shared pipeline feeds both tracks identically — the Track 1 vs Track 2 comparison is itself a scientific contribution

### 6. Target Detectors
Markdown table with 4 rows:

| Detector | Facility | Raw Dimensions | Layout |
|---|---|---|---|
| AGIPD | EuXFEL | 16×512×128 px | 16 modules |
| JUNGFRAU 4M | LCLS CXI | 8×512×1024 px | 8 modules |
| ePix10k | LCLS | varies | multiple configs |
| Eiger4M | Synchrotron/SSX | 2068×2162 px | monolithic |

### 7. Project Status
4-cell phase grid (Phases 1, 2, 3, 4–6) with Phase 1 highlighted as CURRENT.

### 8. Setup
Placeholder block:
> "⚠ Environment setup and training instructions are coming in Phase 2. The conda environment definition (environment.yml) and SLURM scripts will be finalized once the Sol HPC data pipeline is established."

### 9. Citation
BibTeX block:
```bibtex
@misc{ketawala2025sfxhitfinder,
  author      = {Ketawala, Gihan},
  title       = {Detector-Agnostic Hitfinder for SFX},
  year        = {2026},
  institution = {Arizona State University, Fromme Lab},
  url         = {https://github.com/gihankaushyal/Hit_finder}
}
```

### 10. Acknowledgments
One sentence: Fromme Lab, Biodesign Institute, ASU, Prof. Petra Fromme, Sol HPC.

---

## Graphics Strategy

All graphics are embedded directly in the README as either:
- **SVG inline** (hero banner, pipeline diagram, two-track diagram) — no external hosting required, renders on GitHub
- **Markdown tables** (detector overview, phase status)
- **shields.io badges** (standard GitHub badge service)

No PNG/JPG files committed to repo — keeps the repo lean and graphics version-controlled as text.

---

## Implementation Notes

- File: `README.md` at project root
- Hero SVG: diffraction rings at radii 30, 55, 80, 108, 138, 170 px; Bragg spots on first two rings; beamstop shadow at centre
- Pipeline diagram: rendered as an SVG flowchart embedded in README
- Two-track diagram: rendered as an SVG side-by-side comparison
- Badges: shields.io static badges (no CI integration needed yet)
- Phase grid: markdown table with emoji indicators
- `.superpowers/` added to `.gitignore`

---

## Out of Scope

- Installation walkthrough (Phase 2)
- Results / benchmarks (Phase 4–6)
- Demo / Colab notebook link (Phase 6+)
- License file creation (separate task)
