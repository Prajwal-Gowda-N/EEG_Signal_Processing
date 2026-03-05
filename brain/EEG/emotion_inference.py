"""
Emotion Inference Script
=========================
Called by Streamlit UI to predict Valence / Arousal / Dominance
from a raw EEG input.

Usage
──────
  from emotion_inference import EmotionPredictor

  predictor = EmotionPredictor(model_dir="models")

  # Option A — numpy array  (samples, 14)  already preprocessed
  result = predictor.predict(eeg_array)

  # Option B — CSV file path (dreamer_raw_samples format, single trial)
  result = predictor.predict_from_csv("my_trial.csv")

  print(result)
  # {
  #   "valence":   {"label": "High", "confidence": 0.82},
  #   "arousal":   {"label": "Low",  "confidence": 0.75},
  #   "dominance": {"label": "High", "confidence": 0.69},
  #   "summary":   "High Valence · Low Arousal · High Dominance",
  #   "emotion":   "Relaxed / Content",
  #   "raw_scores": {"valence": 1, "arousal": 0, "dominance": 1}
  # }
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt
from torch.amp import autocast

# ─────────────────────────────────────────────
# EEGNet  (must match training definition exactly)
# ─────────────────────────────────────────────

SFREQ = 128   # used inside EEGNet conv kernel sizes

class EEGNet(nn.Module):
    def __init__(self, n_channels=14, n_classes=2, T=384,
                 F1=8, D=2, F2=16, dropout=0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, SFREQ//2),
                      padding=(0, SFREQ//4), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1*D, kernel_size=(n_channels, 1),
                      groups=F1, bias=False),
            nn.BatchNorm2d(F1*D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1*D, F1*D, kernel_size=(1, 16),
                      padding=(0, 8), groups=F1*D, bias=False),
            nn.Conv2d(F1*D, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )
        flat = self._get_flatten_size(n_channels, T)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(flat, n_classes))

    def _get_flatten_size(self, n_channels, T):
        with torch.no_grad():
            x = torch.zeros(1, 1, n_channels, T)
            return self.block2(self.block1(x)).numel()

    def forward(self, x):
        return self.classifier(self.block2(self.block1(x)))


# ─────────────────────────────────────────────
# Emotion mapping  (Valence × Arousal quadrants)
# ─────────────────────────────────────────────

EMOTION_MAP = {
    # (valence, arousal, dominance) → emotion label
    (1, 1, 1): "Relaxed / Content",
    (1, 1, 0): "Calm / Serene",
    (1, 0, 1): "Happy / Excited",
    (1, 0, 0): "Pleased / Joyful",
    (0, 1, 1): "Stressed / Tense",
    (0, 1, 0): "Anxious / Nervous",
    (0, 0, 1): "Sad / Depressed",
    (0, 0, 0): "Bored / Fatigued",
}

# ─────────────────────────────────────────────
# Preprocessing  (must match training pipeline)
# ─────────────────────────────────────────────

def _bandpass(data, lowcut=1.0, highcut=45.0, fs=128, order=4):
    nyq  = fs / 2.0
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, data, axis=0)

def _notch(data, notch=50.0, fs=128, order=4):
    nyq  = fs / 2.0
    low  = max((notch-1)/nyq, 0.001)
    high = min((notch+1)/nyq, 0.999)
    b, a = butter(order, [low, high], btype='bandstop')
    return filtfilt(b, a, data, axis=0)

def preprocess_eeg(eeg: np.ndarray) -> np.ndarray:
    """
    Apply same pipeline as training:
      notch 50 Hz → bandpass 1–45 Hz
    Input : (samples, 14)  float
    Output: (samples, 14)  float32
    """
    eeg = eeg.astype(np.float64)
    eeg = _notch(eeg)
    eeg = _bandpass(eeg)
    return eeg.astype(np.float32)


def to_tensor(eeg: np.ndarray, T: int = 384) -> torch.Tensor:
    """
    Crop / pad EEG to fixed length and reshape to EEGNet input.
    Input : (samples, 14)
    Output: (1, 1, 14, T)  torch.float32
    """
    if eeg.shape[0] >= T:
        eeg = eeg[:T, :]
    else:
        pad = np.zeros((T - eeg.shape[0], eeg.shape[1]), dtype=np.float32)
        eeg = np.vstack([eeg, pad])
    # (14, T) → (1, 1, 14, T)
    tensor = torch.tensor(eeg.T, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor


# ─────────────────────────────────────────────
# EmotionPredictor
# ─────────────────────────────────────────────

class EmotionPredictor:
    """
    Loads the three saved EEGNet models and runs inference.
    """

    LABEL_NAMES = ['valence', 'arousal', 'dominance']

    def __init__(self, model_dir: str = "models", device: str = None):
        self.model_dir = model_dir
        self.device    = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.use_gpu   = self.device.type == 'cuda'

        print(f"[Predictor] Device : {self.device}")

        # Load training metadata
        meta_path = os.path.join(model_dir, 'training_results.json')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.meta = json.load(f)
            self.T        = self.meta['T']
            self.channels = self.meta['channels']
        else:
            self.T        = 384
            self.channels = ['AF3','F7','F3','FC5','T7','P7',
                             'O1','O2','P8','T8','FC6','F4','F8','AF4']
            self.meta     = {}
            print("[Predictor] Warning: training_results.json not found, using defaults.")

        # Load all three models
        self.models = {}
        for label in self.LABEL_NAMES:
            path = os.path.join(model_dir, f"eegnet_{label}.pt")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Model file not found: {path}\n"
                    f"Run emotion_classification.py first to train and save models.")
            self.models[label] = self._load_model(path)

        print(f"[Predictor] Loaded models: {list(self.models.keys())}")

    def _load_model(self, path: str) -> nn.Module:
        ckpt   = torch.load(path, map_location=self.device)
        config = ckpt['config']

        model = EEGNet(
            n_channels = config['n_channels'],
            n_classes  = config['n_classes'],
            T          = config['T'],
            F1         = config['F1'],
            D          = config['D'],
            F2         = config['F2'],
            dropout    = config['dropout'],
        ).to(self.device)

        model.load_state_dict(
            {k: v.to(self.device) for k, v in ckpt['state_dict'].items()})
        model.eval()
        return model

    def _run_model(self, model: nn.Module, tensor: torch.Tensor):
        tensor = tensor.to(self.device, non_blocking=True)

        with torch.no_grad():
            with autocast(device_type=self.device.type,
                          dtype=torch.float16, enabled=self.use_gpu):
                logits = model(tensor)

        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])
        return pred_class, confidence, probs

    def predict(self, eeg: np.ndarray, preprocess: bool = True) -> dict:
        if eeg.ndim != 2 or eeg.shape[1] != 14:
            raise ValueError(
                f"Expected EEG shape (samples, 14), got {eeg.shape}")

        if preprocess:
            eeg = preprocess_eeg(eeg)

        tensor = to_tensor(eeg, T=self.T)

        raw_scores = {}
        output     = {}

        for label in self.LABEL_NAMES:
            pred_class, conf, probs = self._run_model(self.models[label], tensor)
            
            # ──── UPDATED VALENCE LOGIC ────
            high_prob = float(probs[1])  # probability of class 1 ('High')
            label_str = 'High' if high_prob > 0.35 else 'Low'  # ← tunable threshold
            confidence = round(high_prob, 4)

            raw_scores[label] = pred_class
            output[label]     = {
                'label'     : label_str,
                'confidence': confidence,
            }

            # Debug print for valence
            if label == 'valence':
                print(f"[VALENCE DEBUG] Raw probs: {probs.tolist()}, "
                      f"Class: {pred_class}, High prob: {high_prob:.3f} → "
                      f"Label: {label_str}, Conf: {confidence:.3f}")
        # ─────────────────────────────────

        # Summary string
        output['summary'] = (
            f"{output['valence']['label']} Valence · "
            f"{output['arousal']['label']} Arousal · "
            f"{output['dominance']['label']} Dominance"
        )

        # Emotion label from quadrant map
        key = (1 if output['valence']['label'] == 'High' else 0,
               1 if output['arousal']['label'] == 'High' else 0,
               1 if output['dominance']['label'] == 'High' else 0)
        output['emotion']    = EMOTION_MAP.get(key, "Unknown")
        output['raw_scores'] = raw_scores

        return output

    def predict_from_csv(self, csv_path: str, preprocess: bool = True) -> dict:
        df  = pd.read_csv(csv_path)
        eeg = df[self.channels].values.astype(np.float32)
        return self.predict(eeg, preprocess=preprocess)

    def get_model_info(self) -> dict:
        return {
            'model'     : 'EEGNet',
            'device'    : str(self.device),
            'T'         : self.T,
            'sfreq'     : 128,
            'n_channels': 14,
            'channels'  : self.channels,
            'results'   : self.meta.get('results', {}),
        }


# ─────────────────────────────────────────────
# Quick test  (run directly to verify)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Emotion Inference  —  Quick Test")
    print("=" * 55)

    predictor = EmotionPredictor(model_dir="models")

    # Simulate a 5-second EEG trial (random noise as placeholder)
    dummy_eeg = np.random.randn(640, 14).astype(np.float32)

    result = predictor.predict(dummy_eeg, preprocess=True)

    print("\n  Prediction Result:")
    print(f"  Valence   : {result['valence']['label']:<5}  "
          f"(confidence: {result['valence']['confidence']:.2%})")
    print(f"  Arousal   : {result['arousal']['label']:<5}  "
          f"(confidence: {result['arousal']['confidence']:.2%})")
    print(f"  Dominance : {result['dominance']['label']:<5}  "
          f"(confidence: {result['dominance']['confidence']:.2%})")
    print(f"\n  Summary   : {result['summary']}")
    print(f"  Emotion   : {result['emotion']}")
    print("=" * 55)

    info = predictor.get_model_info()
    print(f"\n  Model Info : {info}")