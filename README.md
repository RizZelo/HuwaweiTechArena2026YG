# RoomSense: counting people (0-4) from WiFi CSI

Code, experiments and conclusions from the Huawei TechArena France 2026 "Future Network" hackathon (RoomSense).

**Task.** Predict how many people (0 to 4) are in a room from 3 seconds of WiFi channel measurements (CSI):
`frames (600, 1, 3, 1024, 2)` (600 time steps, 3 receive antennas, 1024 subcarriers, real and imaginary part) and
`intervals (600,)` (time between frames, in microseconds). The output is 5 scores. The model is one ONNX file
(under 100 MB, under 0.7 s per sample on CPU) and it is scored on **rooms never seen in training**, so the whole
project is about generalising from one room to another.

**Data.** 48 labelled recordings (only two rooms contain 1 to 4 people) and 150 unlabelled recordings from six other rooms.

## Results and conclusions

Every upload and its lesson is in [`submissions/log.md`](submissions/log.md). Public leaderboard accuracy (chance is 0.20;
the test is balanced at about 20% per class):

| Model | Score |
|---|---|
| CNN pre-trained with SimCLR on unlabelled windows, then fine-tuned on labels | 0.39 (the same recipe re-run gave about 0.34) |
| Same CNN with cleaning (despike + smoothing) | 0.35 |
| "Image" CNN (time x subcarrier, weights shared over subcarriers) | 0.28 |
| Same-room contrastive pre-training | 0.34 |
| Frame timing (ATI) features only | 0.22 |
| 15 physics features + logistic regression / small MLP | 0.33 / 0.31 |
| Same, with per-room normalisation learned on labels | 0.21 |
| Still/active gate + ranking by an activity score (no training) | 0.41 |
| Teammate's pipeline (presence features, Doppler bandwidth cue, small networks, per-room ranking) | 0.46-0.47 |
| That pipeline with room recognition switched off / made strict | 0.45 / 0.45 |
| That pipeline with a probe that answers 1-4 at random (measures presence alone) | 0.36 |

What the experiments showed:

- **Networks trained on the labelled rooms learn the room or the recording session, not the headcount.** In one labelled room
  more people means less motion, in another it is flat, so supervised models and room-alignment tricks did not transfer.
- **The unlabelled rooms are the closest thing to the test rooms.** Everything that worked uses them only for statistics
  (a room's usual level, quartiles), never for labels.
- **Empty rooms are the easy part, counting is the hard part.** A probe upload measured empty-vs-occupied at about 84% right
  and counting 1-4 at about 39% right (chance 25%). Presence is worth +0.03 per +10 points, counting +0.08.
- **Ranking inside a room beats absolute values.** Scores are turned into ranks inside the recognised room and cut into
  quarters. This depends on knowing the room: with the room removed the score drops by about 0.02, and an unseen room
  is often mistaken for a wrong known room (a strict recognition gate removes that risk).
- **Changes that did not help:** cleaning (neutral), class balancing, moving the class cut points, reweighting the fused
  components, frame timing (only 0.2% of frames are delayed; it correlates with nothing else), averaging spectra to steady a cue,
  a student model trained to predict a recording's average score.
- **Run-to-run luck is about +-0.05** for trained networks, larger than most effects measured, so one upload only detects big changes.
  Label-free checks (steadiness of a score inside a recording, which has a fixed headcount) were the most reliable guide.

The 0.46-0.47 pipeline was built by a teammate. This repository contains the analysis, tests, fixes and probes we ran on it;
the scripts that work on his ONNX model (`fix_ahmed_cuts.py` and everything after it in the list below) need that file,
which is **not included**.

## Get the project back

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # for an NVIDIA GPU install the CUDA build of PyTorch first (see requirements.txt)
```

## Get the data

The data is not in this repository (about 10 GB). Download both archives from the organisers:

| Dataset | Link |
|---|---|
| Labelled training data | https://dl.r0h.eu/techarena2026-paris/FutureNetwork-data-labeled.zip |
| Unlabelled training data | https://dl.r0h.eu/techarena2026-paris/FutureNetwork-data-unlabeled.zip |

SHA-256 checksums:

```text
1f60df2d4c1459c307e444741056e17a5ec537614cf577331ed04eb94d0765c1  FutureNetwork-data-labeled.zip
4b4adf44827d2f62b87c8ce5dae805562964304f73ab40609b5a9efee83c373a  FutureNetwork-data-unlabeled.zip
```

Extract each archive into its own folder so that the code finds this layout (`src/data.py` expects it):

```
Data/FutureNetwork-data-labeled/labeled/      samples.csv + CSI_*.b2 + ATI_*.b2
Data/FutureNetwork-data-unlabeled/unlabeled/  samples.csv + CSI_*.b2 + ATI_*.b2
```

```bash
mkdir -p Data/FutureNetwork-data-labeled Data/FutureNetwork-data-unlabeled
unzip FutureNetwork-data-labeled.zip   -d Data/FutureNetwork-data-labeled
unzip FutureNetwork-data-unlabeled.zip -d Data/FutureNetwork-data-unlabeled
```

(PowerShell: `Expand-Archive FutureNetwork-data-labeled.zip -DestinationPath Data/FutureNetwork-data-labeled`, same for the other archive.)
The problem statement and the rules came from the organisers and are not redistributed here.

## Folders

| Folder | What is in it |
|---|---|
| `src/` | All the code. Notebooks import from here. |
| `notebooks/` | Exploration and plots (`01`-`03`), and `old/` with the first notebooks (feature baseline, CNN, SSL CNN). |
| `submissions/log.md` | The record of every upload: what changed, how it was checked, the score and the lesson. |
| `Data/`, `cache/` | Not in the repository. `Data/` is the dataset, `cache/` is everything the scripts generate (cut windows, trained weights, score tables). |
| `submissions/*.onnx` | Not in the repository. Rebuild a model with `src/export.py` or the scripts below. |

## `src/` modules

Data and signal processing (NumPy, pandas):
- `data.py` - paths, `load_index`, `load_trace`, `windows` (600-frame cuts like the test set), `to_model_input` (the exact tensors the platform gives the model).
- `dsp.py` - guard-subcarrier removal, gain normalisation, cleaning, static/dynamic split, CSI ratio, Doppler spectrum, covariance eigenvalues, delay profile, room fingerprint.
- `activity.py` - the activity score (motion power divided by the window's own 57-100 Hz noise floor) in NumPy.
- `ati.py` - frame-timing features. `physics.py` - 15 physics features and the room fingerprint, in PyTorch.

Networks and training (PyTorch, GPU when available, otherwise CPU):
- `models.py` - the CNN, the torch twin of the cleaning, the `Submission` wrapper that goes into ONNX, SSL helpers.
- `prepare.py` - builds the training caches. `train_ssl_cnn.py` (the 0.39 recipe), `train_image_cnn.py`, `train_same_room.py`,
  `train_ati.py`, `train_physics.py`, `train_physics_mlp.py`, `train_room_norm.py`.

ONNX building blocks and rule models (no training):
- `export.py` - writes the ONNX file and checks it (shape, size, ONNX == PyTorch, CPU time).
- `rule_model.py` (activity rule), `gated_v2.py`, `constant_model.py`, `combine_onnx.py` (average several ONNX models).

Work on the teammate's ONNX model (needs his file, not included; each script prints its own checks):
- `balance_team_model.py`, `fix_ahmed_cuts.py` (finds and repairs the per-room class cut points), `add_ati_component.py`,
  `reweight_components.py`, `presence_probe.py` (measures presence accuracy alone), and the label-free checks
  `check_motion_ati.py`, `check_flaws.py` (unseen-room stress test), `check_presence_threshold.py`,
  `component_steadiness.py`, `steady_count.py`, plus `phase1_checks.py` and `score_table.py`.

## Typical runs (from the project folder, after downloading the data)

```bash
python src/prepare.py --clean
python src/train_ssl_cnn.py --tag clean
python src/export.py --weights cache/cnn_ssl_clean.pt --clean --output submissions/v3_ssl_cnn_clean.onnx
```

```bash
python src/rule_model.py --power-weight 0.5 --output submissions/vR_act_minus_power.onnx   # activity rule, no training
python src/export.py --help                                                                # other architectures and options
```

PyTorch is needed to train a network and for the final ONNX export: the ONNX file must contain the preprocessing too, so the
`dsp.py` steps that are kept are rewritten in torch at export time.

## Shapes

```
csi  (T, 3, 1024) complex64   time x Rx antenna x subcarrier   (Tx axis dropped)
ati  (T,)         float32     microseconds between frames
model input: frames (600, 1, 3, 1024, 2) float32, intervals (600,) float32 -> logits (5,)
```

Notebooks start with `sys.path.insert(0, "../src")`; run them from `notebooks/` with the project environment.
