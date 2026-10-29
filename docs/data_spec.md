# Data Specification

Detector formats, label conventions, and data paths on Sol.

---

## Image Formats

| Extension | Library | Notes |
|-----------|---------|-------|
| `.h5` | h5py | HDF5 — multi-panel detectors; Reborn geometry assembly required |
| `.cxi` | h5py | CXI (HDF5 schema) — same as .h5 |
| `.img` | fabio | ADSC/MAR — already assembled; skip geometry step |

All formats return a 2D float32 array `(H, W)` from `read_image()`. Multi-frame files are read one frame at a time with `read_frame(path, frame_idx)`; `count_frames(path)` gives the frame count.

---

## Current Data Layout (LODO pipeline)

The Track 1 and Track 2 pipelines read **multi-frame CXI files with embedded labels** from the Resonet production dataset, one directory per detector:

| Detector | Directory (`lodo.detector_dirs`) | Files | Frames/file | Total | Stored array (`entry_1/data_1/data`) | Assembled |
|----------|----------------------------------|-------|-------------|-------|--------------------------------------|-----------|
| AGIPD | `Resonet/production/agipd_20k` | 5 | 4000 | 20,000 | `(N, 16, 512, 128)` | 1273 × 1273 |
| JUNGFRAU_4M | `Resonet/production/jungfrau_20k` | 10 | 2000 | 20,000 | `(N, 2164, 2068)` | 2173 × 2173 |
| ePix10k | `Resonet/production/epix10k_20k` | 5 | 4000 | 20,000 | `(N, 5632, 384)` | 1667 × 1667 |
| Eiger4M | `Resonet/production/eiger4m_20k` | 5 | 4000 | 20,000 | `(N, 5632, 384)` | 1687 × 1687 |

(Shapes read from the production files and from assembling a zero frame through each loader in `src/preprocessing/geometry.py`, 2026-10-22. Directory root is `/data/bioxfel/user/gihan/`.)

- **Labels:** `entry_1/labels/hit` (`io._DEFAULT_LABEL_KEY`), one value per frame, `1` = hit. The asymmetric dataset uses them for metadata gating; no external label file is involved.
- **Sessions:** `build_sessions()` (`src/training/lodo.py`) globs `compressed*.cxi` (the `lodo.cxi_pattern` default) under each detector directory. Each file is one session, named `<detector>_<file stem>` (e.g. `AGIPD_compressed0`), with its frame count read from the label array. Splitting is done at the session level, never within a file.
- **Split artifacts:** each fold's train / val / in-domain-test / cross-detector assignment is written to `checkpoints/asymmetric_splits/fold_<N>.json` when a run starts. They are not committed.
- **Detector identity** comes from `entry_1/instrument_1/detector_1/description` (see below), never from the image content.

The sections *Labels Format* and *Split Files* below describe the original one-file-per-frame convention. **No current code path reads `labels.json` or `data/splits/*.txt`** (`data/splits/` is empty); they are kept for reference and for `.img`/single-frame workflows.

---

## Labels Format (legacy, one file per frame)

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

## Split Files (legacy)

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

`src/preprocessing/io.py` (`_HDF5_CANDIDATE_KEYS`) opens the first of these keys that exists in the file, in this order:

1. `entry/data/data`
2. `entry_1/instrument_1/detector_1/detector_corrected/data`
3. `entry_1/data_1/data`
4. `entry_0000/instrument/Simulator/data`

Keys seen per source (confirmed against real files, see the tables below):

| Detector | Source | Key |
|----------|--------|-----|
| AGIPD | development sample (EuXFEL) | `entry_1/instrument_1/detector_1/detector_corrected/data` |
| JUNGFRAU 4M | development sample (LCLS) | `entry_0000/instrument/Simulator/data` |
| ePix10k | development sample (LCLS) | `entry_1/data_1/data` |
| Eiger4M | development sample (synchrotron) | `entry/data/data` |
| all four | **Resonet production CXI** | `entry_1/data_1/data` |

Embedded labels, where present, are read from `entry_1/labels/hit` (`read_embedded_labels`).

### Detector names

Detector type is read from `entry_1/instrument_1/detector_1/description` (`read_detector_description`). The CXI description strings and the short names used in code are:

| CXI description | Short name (`DETECTORS`, `DETECTOR_LOADERS`) |
|-----------------|-----------------------------------------------|
| `AGIPD 1M` | `AGIPD` |
| `Jungfrau 4M` | `JUNGFRAU_4M` |
| `ePix10k 2.2M` | `ePix10k` |
| `EIGER 4M` | `Eiger4M` (the Resonet CrystFEL layout is also available as `EigerRESoNeT`) |

Geometry is built in `src/preprocessing/geometry.py`. Local geometry files in `src/preprocessing/data/` are `agipd.geom`, `epix10k.geom` and `eiger_resonet.geom`; Jungfrau 4M and Eiger4M use Reborn's bundled loaders (`jungfrau_8_pad_geometry_list`, `eiger4m_64_pad_geometry_list`).

---

## Data Paths on Sol

```
data/
├── raw/        → symlink to actual storage
├── processed/  → symlink to preprocessed tensor cache
├── splits/     → empty (.gitkeep); legacy location for train/val/test.txt
└── synthetic/
```

The production CXI files live under `/data/bioxfel/user/gihan/Resonet/production/<detector>_20k/` (referenced from `lodo.detector_dirs` in the configs, not through `data/`). The frame cache root is `cache.root` in `configs/base.yaml` (`/data/bioxfel/user/gihan/Hit_finder_cache`), with an optional per-node NVMe copy at `cache.nvme_root`.

---

## Resonet Multi-Frame CXI Format

CXI files generated by the Resonet simulation pipeline differ from the one-file-per-frame convention used elsewhere. Key properties (all four production detectors):

| Property | Value |
|----------|-------|
| HDF5 data key | `entry_1/data_1/data` |
| Frame shape | per detector, see *Current Data Layout* (e.g. `(N, 5632, 384)` for ePix10k and Eiger4M) — N frames per file |
| Label key | `entry_1/labels/hit` |
| Label encoding | `float32`: `1.0` = hit (class 1), `0.0` = non-hit (class 0) |
| Detector metadata | `entry_1/instrument_1/detector_1/description`, e.g. `"EIGER 4M"` |
| Compression | gzip level 4, chunked per frame (e.g. `(1, 5632, 384)`) |

**Geometry:** The `(5632, 384)` layout is 64 stacked panels of 176 × 192 pixels, not the 2167 × 2070 monolithic Eiger 4M image (that shape belongs to the separate development sample file below). It is assembled with `extract_panels_from_canvas()` followed by Reborn's `PADAssembler` using `eiger4m_64_pad_geometry_list()` (effective distance 0.1139775 m; checked against the former local CrystFEL file by a one-off script before that file was removed in PR #43), giving a 1687 × 1687 frame. `eiger_resonet_pad_geometry_list()` builds the same layout from `eiger_resonet.geom`. The earlier note that panel assembly was "deferred" (with geometry bypassed during normalization) no longer applies.

**Dataset classes** (`src/data/dataset.py`): `AsymmetricCXIDataset` (Track 1 training, hitfinder-guided crops), `SSLPretrainCXIDataset` (Track 2 pretraining, multiple random valid-region crops per frame), and `UnlabeledDataset`. `MultiFrameCXIDataset` is the older flat `(file_path, frame_idx)` dataset that applies an optional `preprocess_fn` to each raw frame; it is marked DEPRECATED in favor of `AsymmetricCXIDataset`.

---

## Confirmed HDF5/CXI Keys — development sample files (verified 2026-05-22, updated AGIPD file)

Probed against the single-detector development sample files (originally kept on a laptop, `detector-images/`). These are **not** the Resonet production files; shapes differ for AGIPD (stored as `(8192, 128)` here, `(16, 512, 128)` in production) and Eiger4M (a true 2167 × 2070 monolithic image here, 64 stacked panels `(5632, 384)` in production).

| Detector     | File            | Format | Confirmed key                                                   | Raw shape        |
|--------------|-----------------|--------|-----------------------------------------------------------------|-----------------|
| AGIPD        | AGIPD.cxi       | CXI    | `entry_1/instrument_1/detector_1/detector_corrected/data`       | (N, 8192, 128)  |
| JUNGFRAU 4M  | Jungfrau.h5     | HDF5   | `entry_0000/instrument/Simulator/data`                          | (N, 2164, 2068) |
| ePix10k      | epix10k.cxi     | CXI    | `entry_1/data_1/data`                                           | (N, 5632, 384)  |
| Eiger4M      | Eiger4M.h5      | HDF5   | `entry/data/data`                                               | (N, 2167, 2070) |

**Notes:**
- All files store N frames with a leading batch dimension — `io.py` returns `data[0]` (first frame).
- AGIPD key updated 2026-05-22: original file (`entry_1/data_1/data`, shape N×5632×384) was replaced with a corrected file using `entry_1/instrument_1/detector_1/detector_corrected/data` (shape N×8192×128).
- `io.py` tries keys in this priority order: `entry/data/data` → `entry_1/instrument_1/detector_1/detector_corrected/data` → `entry_1/data_1/data` → `entry_0000/instrument/Simulator/data`.
- The JUNGFRAU file was renamed from `JUNFRAU.h5` to `Jungfrau.h5` when it was replaced.
- The JUNGFRAU file also contains embedded hit labels at `entry_0000/processing/peakfinder/isHit` (shape: N×uint8) — useful for Phase 4 supervised training.

---

## Confirmed CXI Geometry Keys (verified 2026-08-04, Resonet production files)

Probed against all four detector CXI files in
`/data/bioxfel/user/gihan/Resonet/production/`.

All four detectors share identical HDF5 paths:

| Parameter | HDF5 key | Units |
|-----------|----------|-------|
| Detector distance | `entry_1/instrument_1/detector_1/distance` | metres |
| X-ray wavelength | `entry_1/instrument_1/source_1/wavelength` | metres |
| Pixel size (x) | `entry_1/instrument_1/detector_1/x_pixel_size` | metres |
| Pixel size (y) | `entry_1/instrument_1/detector_1/y_pixel_size` | metres |

Beam centre is **not stored** in any CXI file.
`gpu_pf8.py` computes it as `(frame_rows / 2) × pixel_size` and
`(frame_cols / 2) × pixel_size`. All four detectors have equal x/y pixel sizes.

Observed pixel sizes per detector:

| Detector | Description in CXI | Pixel size |
|----------|---------------------|-----------|
| ePix10k 2.2M | `b'ePix10k 2.2M'` | 100 µm |
| AGIPD 1M | `b'AGIPD 1M'` | 200 µm |
| EIGER 4M | `b'EIGER 4M'` | 100 µm |
| Jungfrau 4M | `b'Jungfrau 4M'` | 75 µm |

`AsymmetricCXIDataset` reads `x_pixel_size`, `distance`, and `wavelength` at
`__init__` time and passes them to `hitfinder.set_geometry()` before each frame's
`find_peaks` call (short-circuited when the CXI path has not changed).

---

## From file to model input

For reference, the shared preprocessing applied to every frame (both tracks), in fixed order: read frame → assemble with Reborn (native resolution) → hitfinder (metadata-gated, on the raw assembled frame) → GCN over the full assembled frame, invalid pixels filled with 0 → crop to 224 × 224 (training: hitfinder-guided; evaluation: stride-224 patch grid) → `random_rot90` → `random_flip` (training) → masked LCN (window 9, `ε = 1e-2` variance form, valid-pixel mask eroded by `EDGE_EROSION_PX = 2`) → peak-aware cutout (training). There is no resize step. The read → GCN prefix can be served from the frame cache. See [architecture.md](architecture.md) and [codebase/data-flow.md](codebase/data-flow.md).
