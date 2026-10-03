import os
import warnings

import numpy as np
import neurokit2 as nk
import torch
from scipy.ndimage import label

import config
from processing.unet1d import UNet1D


class PeakDetector:
    """Detect R-peaks in a four-second multichannel ECG window.

    Candidate peaks identify the most informative adjacent three-channel set.
    A trained 1D U-Net then produces a probability mask, and each valid region
    is refined to the sample with the largest absolute ECG amplitude.
    """

    _MODEL_PATH = os.path.join(os.path.dirname(__file__), "peak_unet1d.pt")

    def __init__(self, sample_rate: float = config.SAMPLE_RATE):
        """Initialize detection parameters and load the trained U-Net."""
        self.sample_rate = sample_rate

        self.num_chn              = config.NUM_CHANNELS
        self.prev_chn             = None        # stability bracket carried across windows
        self.peak_match_tolerance = 8
        self.min_amplitude        = 1
        self.min_snr_db           = 6.0
        self.sig_half_win         = int(0.05 * self.sample_rate)   # ±0.05 s signal window
        self.noise_win            = int(0.2  * self.sample_rate)   # 0.2 s noise window each side

        # Load the trained U-Net using parameters stored in the checkpoint.
        ckpt = torch.load(self._MODEL_PATH, map_location="cpu")
        self.model = UNet1D(**ckpt["hparams"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        self._model_channels: int = ckpt["hparams"]["in_channels"]

    def detect_peaks_per_channel(self, data, proximity=8):
        """Detect R-peaks across all channels:

        - Per-channel detection with NeuroKit2 (positive + negative polarity)
        - SNR filtering (per-peak SNR >= min_snr_db)
        - Group detections by temporal proximity
        - Keep one representative index per channel per group

        Args:
            data: np.ndarray of shape (n_samples, n_channels), already filtered.
            proximity: max sample distance to group detections as the same R-peak.

        Returns:
            peaks_per_channel: dict mapping channel -> sorted np.ndarray of peak indices.
        """
        num_channels = data.shape[1]
        detections = []

        # 1. Per-channel detection (NeuroKit + SNR filter)
        for ch in range(num_channels):
            raw_signal = data[:, ch]

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    _, pos_p = nk.ecg_peaks(raw_signal, sampling_rate=self.sample_rate)
                    _, neg_p = nk.ecg_peaks(-raw_signal, sampling_rate=self.sample_rate)

                all_peaks = np.concatenate([
                    pos_p['ECG_R_Peaks'],
                    neg_p['ECG_R_Peaks'],
                ]).astype(int)

                for p in all_peaks:
                    val = np.abs(raw_signal[int(p)])
                    if val < self.min_amplitude:
                        continue
                    snr_db = peak_snr_db(raw_signal, p, self.sig_half_win, self.noise_win)
                    if snr_db >= self.min_snr_db:
                        detections.append({'index': int(p), 'channel': ch})

            except Exception:
                continue

        peaks_per_channel = {ch: [] for ch in range(num_channels)}

        if not detections:
            return peaks_per_channel

        # 2. Sort and group by proximity
        detections.sort(key=lambda x: x['index'])
        groups = []
        current_group = [detections[0]]
        for i in range(1, len(detections)):
            if detections[i]['index'] - current_group[-1]['index'] <= proximity:
                current_group.append(detections[i])
            else:
                groups.append(current_group)
                current_group = [detections[i]]
        groups.append(current_group)

        # 3. Collect one representative peak index per channel per group — the
        #    largest-|amplitude| detection that channel contributed to the group.
        for group in groups:
            ch_detections = {}
            for det in group:
                ch = det['channel']
                idx = det['index']
                amp = np.abs(data[idx, ch])
                if ch not in ch_detections or amp > ch_detections[ch][1]:
                    ch_detections[ch] = (idx, amp)

            for ch, (idx, _) in ch_detections.items():
                peaks_per_channel[ch].append(idx)

        # Convert to sorted arrays
        for ch in range(num_channels):
            peaks_per_channel[ch] = np.array(sorted(peaks_per_channel[ch]), dtype=int)

        return peaks_per_channel

    def _best_adjacent_triplet(self, peaks_per_channel):
        """Return the contiguous triplet (in valid-channel space) with the
        highest summed peak count, restricted to triplets that are STABLE
        with the previous window's pick.

        Stable = the candidate triplet's [min, max] range touches or overlaps
        prev_chn's [min, max] range (gap ≤ 1). This prevents the picker from
        jumping across the array between windows.

        First window (prev_chn is None): is_stable() is True for every
        candidate and this returns the unrestricted best contiguous triplet —
        the data, not an orientation guess, anchors the first pick. detect()
        then seeds prev_chn from it.
        """
        valid  = list(range(self.num_chn))
        counts = [len(peaks_per_channel.get(c, [])) for c in valid]

        # Stability bracket from the previous window. None on the first
        # window (no seed) → is_stable() short-circuits to True below,
        # giving the unrestricted best triplet.
        prev = self.prev_chn
        if prev:
            prev_lo, prev_hi = min(prev), max(prev)
        else:
            prev_lo, prev_hi = None, None

        def is_stable(c_lo, c_hi):
            """Return whether a candidate touches or overlaps the previous triplet."""
            if prev_lo is None:
                return True
            # No gap on either side: candidate left edge can be at most one
            # past prev's right edge, and vice versa for the right side.
            return c_lo <= prev_hi + 1 and c_hi >= prev_lo - 1

        best_i, best_score = None, -1
        for i in range(len(valid) - 2):
            c_lo, c_hi = valid[i], valid[i + 2]
            if not is_stable(c_lo, c_hi):
                continue
            score = counts[i] + counts[i + 1] + counts[i + 2]
            if score > best_score:
                best_i, best_score = i, score

        return [valid[best_i], valid[best_i + 1], valid[best_i + 2]]

    def get_refined_peaks(self, prob_map, raw_window, threshold=0.5):
        """Refine R-peaks in a single four-second window.

        Args:
            prob_map:      1-D probability array of shape (500,) — model output for
                           the current window.
            raw_window:    Filtered ECG array of shape (500, 3) — the 3 selected
                           channels for the current window.
            threshold:     Probability threshold for ROI detection (default 0.5).

        Returns:
            Sorted list of R-peak sample indices within the 500-sample window.
        """
        detected_candidates = {}
        # 100 ms tolerance is standard for R-peak grouping at 125 Hz
        tolerance = int(0.1 * self.sample_rate)

        prob = prob_map.flatten()

        # 1 & 2. ROI detection via connected components
        roi_mask = prob > threshold
        labeled_regions, num_regions = label(roi_mask)

        # Filter out small regions (noise suppression, < 10 samples)
        if num_regions > 0:
            for r_id in range(1, num_regions + 1):
                if np.sum(labeled_regions == r_id) < 10:
                    labeled_regions[labeled_regions == r_id] = 0
            remaining_regions = np.unique(labeled_regions)
            remaining_regions = remaining_regions[remaining_regions > 0]
        else:
            remaining_regions = []

        # 3. Process each surviving ROI
        for r_id in remaining_regions:
            region_idx = np.where(labeled_regions == r_id)[0]

            # Find peak position as the sample with highest absolute amplitude
            # across all 3 channels within this ROI
            roi_signals        = np.abs(raw_window[region_idx, :])
            max_val_per_channel = np.max(roi_signals, axis=0)
            best_ch            = np.argmax(max_val_per_channel)

            local_max_idx_in_roi = np.argmax(roi_signals[:, best_ch])
            best_local_idx       = region_idx[local_max_idx_in_roi]
            best_amp             = roi_signals[local_max_idx_in_roi, best_ch]

            # 4. Duplicate / overlap resolution — keep the higher-amplitude detection
            is_duplicate = False
            for existing_idx in list(detected_candidates.keys()):
                if abs(existing_idx - best_local_idx) < tolerance:
                    if best_amp > detected_candidates[existing_idx]['amp']:
                        del detected_candidates[existing_idx]
                        detected_candidates[best_local_idx] = {'amp': best_amp}
                    is_duplicate = True
                    break

            if not is_duplicate:
                detected_candidates[best_local_idx] = {'amp': best_amp}

        return sorted(detected_candidates.keys())

    def detect(self, filtered_signal, normalized_signal) -> tuple[np.ndarray, list[int]]:
        """Detect R-peaks in a single four-second preprocessed ECG window.

        Args:
            filtered_signal:    Array of shape (500, 13) — bandpass-filtered and
                                notch-filtered. Used for channel selection and
                                peak refinement.
            normalized_signal:  Array of shape (500, 13) — z-score normalised.
                                Used as model input.

        Returns:
            A tuple containing the R-peak indices and selected channel indices.
        """
        # ── 1. Channel selection ──────────────────────────────────────────────
        peaks_per_channel = self.detect_peaks_per_channel(
            filtered_signal, proximity=self.peak_match_tolerance,
        )
        top_channels = self._best_adjacent_triplet(peaks_per_channel)

        # Carry the pick forward as the next window's stability bracket.
        if top_channels:
            self.prev_chn = top_channels

        # ── 2. Build (500, 3) windows for model input and peak refinement ────
        raw_window = filtered_signal[:, top_channels].astype(np.float32)    # (500, ≤3) for refinement
        norm_window = normalized_signal[:, top_channels].astype(np.float32) # (500, ≤3) for model

        # Pad to exactly _model_channels (3) with zero leads if the picker
        # found fewer (the model input is fixed-width).
        if norm_window.shape[1] < self._model_channels:
            pad = self._model_channels - norm_window.shape[1]
            norm_window = np.pad(norm_window, ((0, 0), (0, pad)))
            raw_window  = np.pad(raw_window,  ((0, 0), (0, pad)))

        # ── 3. Model inference (PyTorch UNet1D) ───────────────────────────────
        model_input = torch.from_numpy(norm_window.T).unsqueeze(0)         # (1, 3, 500)
        with torch.no_grad():
            logits = self.model(model_input)                              # (1, 500)
        prob_map = torch.sigmoid(logits).squeeze(0).numpy()               # (500,)

        # ── 4. Refine probabilities → peak indices ────────────────────────────
        peak_indices = self.get_refined_peaks(
            prob_map,
            raw_window,
        )

        return np.array(peak_indices, dtype=int), top_channels


def peak_snr_db(signal, peak_idx, sig_half, noise_w):
    """Per-peak SNR in dB: 20·log10(std(signal_region) / std(noise_region)).

    Signal region: [peak-sig_half, peak+sig_half+1).
    Noise region:  noise_w samples beyond each side of the signal region.
    Returns -inf if either region is empty or degenerate; the caller rejects
    invalid values.
    """
    # Use the full signal span.
    n_samples = len(signal)

    # signal region: a window centred on the peak
    sig_start = max(0, peak_idx - sig_half)
    sig_end   = min(n_samples, peak_idx + sig_half + 1)
    sig_region = signal[sig_start:sig_end]

    # noise region: noise_w samples just outside the signal region, each side
    noise_left  = signal[max(0, sig_start - noise_w):sig_start]
    noise_right = signal[sig_end:min(n_samples, sig_end + noise_w)]
    noise_region = np.concatenate([noise_left, noise_right])

    # empty region → undefined SNR → reject
    if len(sig_region) == 0 or len(noise_region) == 0:
        return -np.inf

    # degenerate (flat) region → undefined ratio → reject
    sig_std   = np.std(sig_region)
    noise_std = np.std(noise_region)
    if noise_std <= 1e-9 or sig_std <= 1e-9:
        return -np.inf

    # dB SNR
    return 20.0 * np.log10(sig_std / noise_std)
