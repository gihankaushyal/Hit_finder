# Data Specification

Detector formats, label conventions, and data paths on Sol.

---

## Image Formats

| Extension | Library | Notes |
|-----------|---------|-------|
| `.h5` | h5py | HDF5 — multi-panel detectors; Reborn geometry assembly required |
| `.cxi` | h5py | CXI (HDF5 schema) — same as .h5 |
| `.img` | fabio | ADSC/MAR — already assembled; skip geometry step |

All formats return a 2D float32 array `(H, W)` from `read_image()`.

---

## Labels Format

Labels are stored in a single JSON file mapping **absolute path strings** to integer labels:

```json
{
  "/data/raw/agipd/run042/frame_001.h5": 1,
  "/data/raw/agipd/run042/frame_002.h5": 0,
  "/data/raw/agipd/run042/frame_003.h5": 1
}
```

- `1` = hit (crystal diffraction pattern present)
- `0` = non-hit (blank or background frame)
- Keys must be **absolute paths** matching exactly the paths in split `.txt` files
- One labels file can cover multiple splits (train/val/test all reference the same file)

### Generating labels.json

Labels typically come from CrystFEL's `indexamajig` or `process_hkl` output. Convert to the JSON format with:

```python
import json
from pathlib import Path

# Example: build from a CrystFEL stream file hit list
hits = set(...)      # absolute paths of hit frames
all_frames = list(Path("/data/raw").glob("**/*.h5"))

labels = {str(p): int(p in hits) for p in all_frames}
Path("labels.json").write_text(json.dumps(labels, indent=2))
```

---

## Split Files

Plaintext `.txt` files listing one absolute image path per line:

```
/data/raw/agipd/run042/frame_001.h5
/data/raw/agipd/run042/frame_002.h5
/data/raw/jungfrau/run011/frame_007.h5
```

Blank lines are ignored. Splits are stored under `data/splits/`:

```
data/splits/
├── train.txt    # training set (cross-detector, leave-one-out)
├── val.txt      # validation set
└── test.txt     # held-out test set (one detector left out per fold)
```

---

## HDF5 Key Convention

Current default key: `entry/data/data`

This must be confirmed against real detector files before Phase 3. Known variations:

| Detector | Facility | Likely key |
|----------|----------|------------|
| AGIPD | EuXFEL | `entry_1/data_1/data` |
| JUNGFRAU 4M | LCLS | `entry/data/data` |
| ePix10k | LCLS | `entry/data/data` |
| Eiger4M | Synchrotron | `entry/data/data` |

Update `read_image()` in `src/preprocessing/io.py` once confirmed.

---

## Data Paths on Sol

```
data/
├── raw/        → symlink to actual storage
├── processed/  → symlink to preprocessed tensor cache
└── splits/     → train.txt, val.txt, test.txt
```

`labels.json` lives alongside the split files or in `data/`.

---

## Confirmed HDF5/CXI Keys (verified 2026-05-21)

Probed against real detector files in `/Users/gketawal/Desktop/detector-images/`.

| Detector     | File            | Format | Confirmed key                              | Raw shape        |
|--------------|-----------------|--------|--------------------------------------------|-----------------|
| AGIPD        | AGIPD.cxi       | CXI    | `entry_1/data_1/data`                      | (N, 5632, 384)  |
| JUNGFRAU 4M  | JUNFRAU.h5      | HDF5   | `entry_0000/instrument/Simulator/data`     | (N, 2164, 2068) |
| ePix10k      | epix10k.cxi     | CXI    | `entry_1/data_1/data`                      | (N, 5632, 384)  |
| Eiger4M      | Eiger4M.h5      | HDF5   | `entry/data/data`                          | (N, 2167, 2070) |

**Notes:**
- All files store N frames with a leading batch dimension — `io.py` returns `data[0]` (first frame).
- AGIPD and ePix10k CXI files follow CrystFEL output convention (`entry_1/data_1/data`).
- Data in these files is already assembled (2D per frame), not raw multi-panel.
- `io.py` tries keys in this priority order: `entry/data/data` → `entry_1/data_1/data` → `entry_0000/instrument/Simulator/data`.
- The JUNGFRAU file (`JUNFRAU.h5`) has a filename typo — missing one 'G'. This is the file as received.
- The JUNGFRAU file also contains embedded hit labels at `entry_0000/processing/peakfinder/isHit` (shape: N×uint8) — useful for Phase 4 supervised training.
