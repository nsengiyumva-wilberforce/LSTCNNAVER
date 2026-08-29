# Lightweight Spatio-Temporal CNN for Audio-Visual Emotion Recognition

PyTorch implementation of Ding, Tang, and Lu, *Lightweight Spatio-Temporal Convolutional Neural Network for Audio-Visual Emotion Recognition*, IEEE Transactions on Affective Computing, 16(4):2721–2734, 2025. [DOI: 10.1109/TAFFC.2025.3566773](https://doi.org/10.1109/TAFFC.2025.3566773)

## Architecture (paper)

| Item | Paper |
| --- | --- |
| Spatial 2D CNN | **16 → 32 → 64**, grayscale **64×64**, 3 layers |
| Audio 1D CNN | **16 → 32**, **40 MFCCs**, 2 layers |
| Alignment | each clip → **6 video frames** + **6 audio segments** |
| Size | **0.06M** parameters, **0.014 GFLOPs** |
| Deploy | quantization-aware training → **TFLite** |
| Accuracy | SAVEE **97.57%**, RAVDESS **95.89%**, MEAD **98.57%** |

Six grayscale frames are stacked as a 6-channel 64×64 input (one spatial forward). Six equal-length audio chunks are encoded with a shared 1D CNN so the two streams stay time-aligned.

## Setup

```bash
cd /home/computergeek/lightweight_SER
source lightweight_SER/bin/activate
pip install -r requirements.txt
```

## Train on RAVDESS

Use the **audio-visual videos**, not the audio-only zip. Official pack: [Zenodo RAVDESS](https://zenodo.org/records/1188976) (`Video_Speech_Actor_*.zip`). Decoding soundtrack from `.mp4` needs **ffmpeg** (`sudo apt install ffmpeg`).

Expected layout (nested `Video_Speech_Actor_XX` folders are fine):

```text
data/raw/ravdess/Actor_01/01-01-01-01-01-01-01.mp4
data/raw/ravdess/Actor_02/...
```

Train (8 classes, speech-only, speaker split, 50 epochs, batch 16, early stopping). Checkpoint: `outputs/ravdess/best.pt`.

```bash
python train.py \
  --dataset ravdess \
  --data-root data/raw/ravdess \
  --out-dir outputs/ravdess
```

Evaluate / infer:

```bash
python evaluate.py \
  --checkpoint outputs/ravdess/best.pt \
  --split test \
  --dataset ravdess \
  --data-root data/raw/ravdess

python infer.py \
  --checkpoint outputs/ravdess/best.pt \
  --video data/raw/ravdess/Actor_01/01-01-01-01-01-01-01.mp4
```

QAT + TFLite after the videos are in place. This trains a Keras copy from the dataset; it does not load `best.pt`.

```bash
python qat_tflite.py --dataset ravdess --out-dir outputs/ravdess/tflite
```

## Smoke test (no corpus)

```bash
python train.py --dataset synthetic --epochs 5 --out-dir outputs/synthetic
python evaluate.py --checkpoint outputs/synthetic/best.pt --split test
python qat_tflite.py --dataset synthetic --out-dir outputs/tflite
```

```bash
PYTHONPATH=src python -m pytest tests/test_model.py -q
```
