# Lightweight Spatio-Temporal CNN for Audio-Visual Emotion Recognition

PyTorch implementation of Ding, Tang, and Lu, *Lightweight Spatio-Temporal Convolutional Neural Network for Audio-Visual Emotion Recognition*, IEEE Transactions on Affective Computing, 16(4):2721–2734, 2025. [DOI: 10.1109/TAFFC.2025.3566773](https://doi.org/10.1109/TAFFC.2025.3566773)

## Architecture (paper)

| Item | Paper |
| --- | --- |
| Spatial 2D CNN | one **64×64×1** face, valid **3×3**, **16→32→64**, pool 2, flatten **2304** |
| Audio 1D CNN | mean **40** MFCCs, valid **5×1**, **16→32**, pool 2, flatten **224** |
| Fusion | concat → dense **16** (SAVEE **14**) → dropout → dense **8** (SAVEE **7**) |
| Alignment | clip → **6** independent (face, MFCC) windows at one-sixth intervals |
| Regularization | dropout **0.4** RAVDESS, **0.5** SAVEE, **0.3** MEAD |
| Train | batch **128**, Adam 0.001, early stop Δval_loss **< 0.001** for **30** epochs |
| Size | **0.06M** parameters, **0.014 GFLOPs** |
| Deploy | quantization-aware training → **TFLite** |
| Accuracy | SAVEE **97.57%**, RAVDESS **95.89%**, MEAD **98.57%** |

Each video is split into six non-overlapping windows. Every window is one training example: one Haar-cropped grayscale 64×64 face plus a 40-d mean-MFCC vector. Both branches are flattened (Fig. 3), concatenated, and classified. Training uses a class-stratified 80/20 then 80/20 split, batch size 128, and early stopping on validation loss.

## Setup

```bash
cd /home/computergeek/lightweight_SER
source lightweight_SER/bin/activate
pip install -r requirements.txt
```

## Train on RAVDESS

Use the **Video_Speech** mp4s for faces. Optional but better: also unzip **Audio_Speech_Actors_01-24.zip** into the same root so MFCCs come from official 48 kHz wavs (same takes as the video soundtrack, no ffmpeg). Zenodo: [RAVDESS](https://zenodo.org/records/1188976).

```text
/media/bitwire/SER-datasets/Ravdess/
  Video_Speech_Actor_01/Actor_01/01-01-01-01-01-01-01.mp4
  Audio_Speech_Actors/Actor_01/03-01-01-01-01-01-01.wav
```

Train (8 classes, speech-only, stratified 80/20 then 80/20, batch 128, early stopping on val loss). Checkpoint: `outputs/ravdess/best.pt`. Use `split: speaker` in the config for a speaker-independent holdout.

The first run decodes every video once and writes Haar faces + MFCCs under `data/cache/ravdess`. Later epochs only read those files. Use `--num-workers 8` if the machine has spare cores.

```bash
python train.py \
  --dataset ravdess \
  --data-root /media/bitwire/SER-datasets/Ravdess \
  --out-dir outputs/ravdess \
  --num-workers 8
```

Evaluate / infer:

```bash
python evaluate.py \
  --checkpoint outputs/ravdess/best.pt \
  --split test \
  --dataset ravdess \
  --data-root /media/bitwire/SER-datasets/Ravdess

python infer.py \
  --checkpoint outputs/ravdess/best.pt \
  --video /media/bitwire/SER-datasets/Ravdess/Actor_01/01-01-01-01-01-01-01.mp4
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
