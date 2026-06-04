# Speech Emotion Recognition

A machine learning system for classifying emotional content in voice clips, supporting **happy, sad, angry, neutral, fearful, disgusted, and surprised** emotion categories.

## Project Structure

```
speech_emotion/
├── data/
│   ├── download_datasets.py     # Scripts to download RAVDESS, TESS, EmoDB
│   └── dataset_loader.py        # Unified dataset loading & splitting
├── features/
│   ├── handcrafted.py           # MFCC, RMSE, chroma, ZCR, spectral features
│   └── spectrogram.py           # Mel-spectrogram & log-spectrogram generation
├── models/
│   ├── baseline.py              # Logistic Regression, SVM baselines
│   ├── cnn.py                   # CNN on spectrograms
│   ├── lstm.py                  # LSTM / Bi-LSTM on feature sequences
│   ├── cnn_lstm.py              # Hybrid CNN + LSTM architecture
│   └── transformer.py           # Transformer-based encoder
├── evaluation/
│   ├── metrics.py               # Accuracy, precision, recall, F1, confusion matrix
│   └── cross_validation.py      # Speaker-independent cross-validation
├── utils/
│   ├── config.py                # Hyperparameters & paths
│   ├── audio_utils.py           # Audio loading, augmentation, normalization
│   └── visualization.py         # Plots for features, confusion matrices, training curves
├── train.py                     # Main training script
├── evaluate.py                  # Evaluation & report generation
├── predict.py                   # Inference on a single audio file
├── experiments.py               # Automated experiment comparison runner
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Download Datasets
```bash
python data/download_datasets.py --datasets ravdess tess emodb --output_dir ./data/raw
```

### 2. Training Models
To visualize the data and train different models, run the notebook.

## Datasets

| Dataset | Languages | Emotions | Actors |
|---------|-----------|----------|--------|
| [RAVDESS](https://zenodo.org/record/1188976) | English | 8 | 24 |
| [TESS](https://tspace.library.utoronto.ca/handle/1807/24487) | English | 7 | 2 |
| [EmoDB](http://emodb.bilderbar.info/download/) | German | 7 | 10 |

## Emotion Labels

`angry, disgust, fear, happy, neutral, sad, surprised`

## Models

| Model | Features | Notes |
|-------|----------|-------|
| Logistic Regression | Handcrafted | Baseline |
| SVM (RBF) | Handcrafted | Strong baseline |
| CNN | Log-Mel Spectrogram | Image-like classification |
| CNN + Bi-LSTM | MFCC sequences | Temporal modeling |

## Evaluation

All models are evaluated with:
- Accuracy, weighted/macro Precision, Recall, F1
- Per-class confusion matrix
- Speaker-independent cross-validation (leave-one-speaker-out)
