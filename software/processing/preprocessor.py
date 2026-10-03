"""ECG filtering and channel-wise normalization."""

import numpy as np
from scipy import signal

import config

class ECGPreprocessor:
    """Bandpass-filter, notch-filter, and normalize multichannel ECG signals."""

    def __init__(self, sample_rate=config.SAMPLE_RATE):
        """Create filters for the specified sample rate."""
        self.sample_rate = sample_rate
        self.n_channels = config.NUM_CHANNELS

        # Butterworth bandpass filter.
        self.sos_band = signal.butter(
            config.FILTER_ORDER,
            [config.BANDPASS_LOW_HZ, config.BANDPASS_HIGH_HZ],
            btype='bandpass', fs=self.sample_rate, output='sos',
        )
        # Notch filter.
        self.b_notch, self.a_notch = signal.iirnotch(
            config.NOTCH_FREQ, config.NOTCH_Q, self.sample_rate,
        )

    def process(self, signal_data):
        """Return filtered and channel-wise normalized ECG arrays."""
        if signal_data is None or signal_data.shape[0] == 0:
            return None

        filtered = np.zeros_like(signal_data)

        for ch in range(self.n_channels):
            bp = signal.sosfiltfilt(self.sos_band, signal_data[:, ch])
            notch = signal.filtfilt(self.b_notch, self.a_notch, bp)
            filtered[:, ch] = notch

        normalized = (filtered - np.mean(filtered, axis=0)) / (np.std(filtered, axis=0) + 1e-6)
        
        return filtered, normalized
