"""
DREAMER Dataset Preprocessing Script
======================================
Input  : DREAMER.mat (single file)
Outputs:
  1. dreamer_raw_samples.csv    → one row per EEG sample  (Deep Learning)
  2. dreamer_trial_features.csv → one row per trial       (Classical ML)

Confirmed struct layout:
  DREAMER.Data[p]
    ├── EEG.stimuli[t]    (25472, 14)  float64
    ├── EEG.baseline[t]   (7808,  14)  float64
    ├── ScoreValence[t]   uint8  (1–5)
    ├── ScoreArousal[t]   uint8  (1–5)
    └── ScoreDominance[t] uint8  (1–5)

  DREAMER.EEG_SamplingRate  = 128
  DREAMER.noOfSubjects      = 23
  DREAMER.noOfVideoSequences= 18
"""

import os
import time
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import butter, filtfilt

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

MAT_FILE     = "C:/Users/Prajwal/Documents/Brain-Signaling/brain/data/emotion/DREAMER.mat"
OUT_RAW      = "C:/Users/Prajwal/Documents/Brain-Signaling/brain/data/emotion/dreamer_raw_samples.csv"
OUT_FEATURES = "C:/Users/Prajwal/Documents/Brain-Signaling/brain/data/emotion/dreamer_trial_features.csv"

SFREQ          = 128
N_PARTICIPANTS = 23
N_TRIALS       = 18

CHANNELS = ['AF3','F7','F3','FC5','T7','P7',
            'O1','O2','P8','T8','FC6','F4','F8','AF4']

BANDS = {
    'delta': (1,   4),
    'theta': (4,   8),
    'alpha': (8,  13),
    'beta' : (13, 30),
    'gamma': (30, 45),
}

# ─────────────────────────────────────────────
# DATA ACCESSORS
# ─────────────────────────────────────────────

def get_participant(dreamer, idx):
    return dreamer.Data[idx]


def get_scores(participant, trial_idx):
    v = int(participant.ScoreValence[trial_idx])
    a = int(participant.ScoreArousal[trial_idx])
    d = int(participant.ScoreDominance[trial_idx])
    return v, a, d


def get_eeg(participant, trial_idx):
    eeg_stim = np.array(participant.EEG.stimuli[trial_idx],  dtype=np.float64)
    eeg_base = np.array(participant.EEG.baseline[trial_idx], dtype=np.float64)
    return eeg_stim, eeg_base

# ─────────────────────────────────────────────
# SIGNAL PROCESSING
# ─────────────────────────────────────────────

def bandpass_filter(data, lowcut=1.0, highcut=45.0, fs=128, order=4):
    """1–45 Hz bandpass filter."""
    nyq  = fs / 2.0
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data, axis=0)


def notch_filter(data, notch=50.0, fs=128, order=4):
    """50 Hz powerline notch filter."""
    nyq  = fs / 2.0
    low  = max((notch - 1.0) / nyq, 0.001)
    high = min((notch + 1.0) / nyq, 0.999)
    b, a = butter(order, [low, high], btype='bandstop')
    return filtfilt(b, a, data, axis=0)


def baseline_correct(eeg_stim, eeg_base):
    """Subtract mean of baseline from stimulus EEG."""
    return eeg_stim - np.mean(eeg_base, axis=0)


def preprocess_trial(eeg_stim, eeg_base):
    """
    Full preprocessing pipeline per trial:
      1. Baseline correction
      2. Notch filter  (50 Hz)
      3. Bandpass filter (1–45 Hz)
    Returns: float64 NumPy array (samples × 14)
    """
    eeg = baseline_correct(eeg_stim, eeg_base)
    eeg = notch_filter(eeg,    notch=50.0, fs=SFREQ)
    eeg = bandpass_filter(eeg, lowcut=1.0, highcut=45.0, fs=SFREQ)
    return eeg


def extract_band_power(eeg):
    """
    Mean band power per channel for 5 frequency bands.
    Returns flat dict of 70 features (5 bands × 14 channels).
    """
    features = {}
    nyq = SFREQ / 2.0

    for band_name, (lo, hi) in BANDS.items():
        b, a     = butter(4, [lo / nyq, hi / nyq], btype='band')
        filtered = filtfilt(b, a, eeg, axis=0)
        power    = np.mean(filtered ** 2, axis=0)   # mean power per channel
        for i, ch in enumerate(CHANNELS):
            features[f'{ch}_{band_name}'] = round(float(power[i]), 6)

    return features


def binarize(score):
    """1–5 scale → binary: >3 = High (1), ≤3 = Low (0)."""
    return 1 if score > 3 else 0

# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────

def load_dreamer(path):
    print(f"[INFO] Loading {path} ...")
    mat     = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    dreamer = mat['DREAMER']
    print(f"[INFO] OK  |  Subjects={int(dreamer.noOfSubjects)}"
          f"  Videos={int(dreamer.noOfVideoSequences)}"
          f"  EEG_fs={int(dreamer.EEG_SamplingRate)} Hz")
    return dreamer

# ─────────────────────────────────────────────
# OUTPUT 1 : RAW SAMPLES CSV
# ─────────────────────────────────────────────

def export_raw_samples(dreamer):
    """
    One row per EEG sample.
    Columns : AF3…AF4, participant, trial, sample_idx,
              valence, arousal, dominance,
              valence_bin, arousal_bin, dominance_bin
    Use for : LSTM, EEGNet, 1D-CNN, Transformer
    """
    print("\n[INFO] ── Generating dreamer_raw_samples.csv ──")
    t0   = time.time()
    rows = []

    for p in range(N_PARTICIPANTS):
        participant = get_participant(dreamer, p)

        for t in range(N_TRIALS):
            eeg_stim, eeg_base = get_eeg(participant, t)
            v, a, d            = get_scores(participant, t)
            eeg                = preprocess_trial(eeg_stim, eeg_base)

            for s_idx, sample in enumerate(eeg):
                row = {ch: round(float(sample[i]), 6) for i, ch in enumerate(CHANNELS)}
                row.update({
                    'participant'  : p + 1,
                    'trial'        : t + 1,
                    'sample_idx'   : s_idx,
                    'valence'      : v,
                    'arousal'      : a,
                    'dominance'    : d,
                    'valence_bin'  : binarize(v),
                    'arousal_bin'  : binarize(a),
                    'dominance_bin': binarize(d),
                })
                rows.append(row)

        elapsed = time.time() - t0
        print(f"  P{p+1:02d}/23  |  {elapsed:.1f}s  |  rows so far: {len(rows):,}")

    col_order = (CHANNELS +
                 ['participant', 'trial', 'sample_idx',
                  'valence', 'arousal', 'dominance',
                  'valence_bin', 'arousal_bin', 'dominance_bin'])

    df = pd.DataFrame(rows)[col_order]
    df.to_csv(OUT_RAW, index=False)

    total = time.time() - t0
    print(f"\n[DONE] {OUT_RAW}")
    print(f"       Shape : {df.shape}  |  Time : {total:.1f}s")
    return df

# ─────────────────────────────────────────────
# OUTPUT 2 : TRIAL FEATURES CSV
# ─────────────────────────────────────────────

def export_trial_features(dreamer):
    """
    One row per trial (414 total).
    Columns : 70 band-power features + 8 label/meta cols
    Use for : SVM, Random Forest, XGBoost
    """
    print("\n[INFO] ── Generating dreamer_trial_features.csv ──")
    t0   = time.time()
    rows = []

    for p in range(N_PARTICIPANTS):
        participant = get_participant(dreamer, p)

        for t in range(N_TRIALS):
            eeg_stim, eeg_base = get_eeg(participant, t)
            v, a, d            = get_scores(participant, t)
            eeg                = preprocess_trial(eeg_stim, eeg_base)
            feats              = extract_band_power(eeg)

            feats.update({
                'participant'  : p + 1,
                'trial'        : t + 1,
                'valence'      : v,
                'arousal'      : a,
                'dominance'    : d,
                'valence_bin'  : binarize(v),
                'arousal_bin'  : binarize(a),
                'dominance_bin': binarize(d),
            })
            rows.append(feats)

        elapsed = time.time() - t0
        print(f"  P{p+1:02d}/23  |  {elapsed:.1f}s")

    feature_cols = [f'{ch}_{b}' for ch in CHANNELS for b in BANDS]
    label_cols   = ['participant', 'trial',
                    'valence', 'arousal', 'dominance',
                    'valence_bin', 'arousal_bin', 'dominance_bin']

    df = pd.DataFrame(rows)[feature_cols + label_cols]
    df.to_csv(OUT_FEATURES, index=False)

    total = time.time() - t0
    print(f"\n[DONE] {OUT_FEATURES}")
    print(f"       Shape : {df.shape}  |  Features : 70 (5 bands × 14 ch)  |  Time : {total:.1f}s")
    return df

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    if not os.path.exists(MAT_FILE):
        print(f"\n[ERROR] '{MAT_FILE}' not found.")
        print("        Place DREAMER.mat in the same folder as this script,")
        print("        OR update MAT_FILE at the top.")
        exit(1)

    print("=" * 55)
    print("  DREAMER PREPROCESSING")
    print("=" * 55)
    print(f"  Input   : {MAT_FILE}")
    print(f"  Outputs : {OUT_RAW}")
    print(f"            {OUT_FEATURES}")
    print("=" * 55)

    dreamer = load_dreamer(MAT_FILE)

    start   = time.time()
    df_raw  = export_raw_samples(dreamer)
    df_feat = export_trial_features(dreamer)
    total   = time.time() - start

    print("\n" + "=" * 55)
    print("  ALL DONE")
    print("=" * 55)
    print(f"  dreamer_raw_samples.csv     shape={df_raw.shape}")
    print(f"  dreamer_trial_features.csv  shape={df_feat.shape}")
    print(f"  Total time : {total:.1f}s")
    print("=" * 55)