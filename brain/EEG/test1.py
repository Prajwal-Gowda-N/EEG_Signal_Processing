# QUICK PYTHON SCRIPT - Run this to generate perfect test file:
import numpy as np

# Generate REALISTIC EEG: 384 samples × 14 channels (exactly what EEGNet expects)
np.random.seed(42)  # reproducible
eeg_realistic = np.random.randn(384, 14).astype(np.float32) * 50  # scale like DREAMER data

# Save as CSV (exactly 14 columns, 384 rows)
np.savetxt("dreamer_eegnet_test.csv", eeg_realistic, delimiter=',', fmt='%.6f')

print("✅ dreamer_eegnet_test.csv created (384 rows × 14 channels)")
print("   Perfect for your EEGNet (T=384, 128Hz, 3sec window)")
