import numpy as np
from scipy.ndimage import gaussian_filter1d
import config


class HeartRateCalculator:
    """Estimate and smooth heart rate from R-peaks across overlapping windows."""

    def __init__(self, sample_rate=config.SAMPLE_RATE):
        """Initialize HR limits and rolling real-time state."""
        self.sample_rate = sample_rate

        # ── Persistent real-time state ────────────────────────────────────────
        self._current_segment = []          # Global peak indices of ongoing valid segment
        self._recent_rr_intervals = []      # Recent valid RR intervals for adaptive threshold
        self._global_sample_offset = 0      # Buffer start in global samples; +sample_rate each call
        self._last_processed_global = -1    # Last global peak index fed into the state machine
        self._window_all_accepted = []      # Peaks accepted in the current window

        # HR validity limits in samples (derived from config)
        self.min_rr = int(60.0 * self.sample_rate / config.HR_MAX_BPM)
        self.max_rr = int(60.0 * self.sample_rate / config.HR_MIN_BPM)

        # ── Gaussian smoothing buffer ─────────────────────────────────────────
        self._bpm_history = []

        # ── Last computed valid BPM (repeated when no new peaks) ──────────────
        self._last_computed_bpm = None

    # ── Public API ────────────────────────────────────────────────────────────

    def compute(self, peak_indices):
        """
        Public entry point called each loop iteration (once per second).

        Uses the last 4 peaks (3 RR intervals) to compute BPM.

        Args:
            peak_indices: 1-D int array of local R-peak sample indices (0-499)

        Returns:
            dict with keys:
              "bpm"       - float or None
              "timestamp" - float seconds or None
              "status"    - "ok" | "invalid" | "filtered" | "insufficient_data"
              "peaks"     - 1-D int array of newly accepted global peak indices
                            in _current_segment from this window
              "all_peaks" - 1-D int array of all peaks accepted in this window
        """
        # Track peaks newly appended while updating the rolling peak state.
        seg_len_before = len(self._current_segment)
        self._update_peak_state(peak_indices)
        new_segment_peaks = self._current_segment[seg_len_before:]

        # Compute HR, limit sudden changes, and apply Gaussian smoothing.
        result = self._compute_bpm(has_new_peaks=len(new_segment_peaks) > 0)
        result = self._filter_sudden_change(result)
        result = self._smooth_bpm(result)
        result["peaks"] = np.array(new_segment_peaks, dtype=int)
        result["all_peaks"] = np.array(self._window_all_accepted, dtype=int)

        # Bound rolling histories during long sessions.
        if len(self._current_segment) > 10:
            self._current_segment = self._current_segment[-10:]
        if len(self._bpm_history) > 30:
            self._bpm_history = self._bpm_history[-30:]

        self._global_sample_offset += int(self.sample_rate)
        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _update_peak_state(self, local_peak_indices,
                           adaptive_window=5,
                           threshold_multiplier=1.5):
        """Update peak and RR state from one overlapping signal window.

        Convert local peak indices to global positions, remove previously
        processed peaks, and apply the adaptive short-RR and gap rules.

        Updates self._current_segment, self._recent_rr_intervals, and
        self._window_all_accepted in place.
        """
        # Convert window-local indices to global positions and remove overlap duplicates.
        window_start = self._global_sample_offset
        global_peaks = sorted(int(p) + window_start for p in local_peak_indices)
        new_peaks    = [p for p in global_peaks if p > self._last_processed_global]

        # Track all peaks that pass validity checks.
        self._window_all_accepted = []

        for new_peak in new_peaks:
            self._last_processed_global = new_peak

            if not self._current_segment:
                self._current_segment.append(new_peak)
                self._window_all_accepted.append(new_peak)
                continue

            last_peak = self._current_segment[-1]
            rr_to_new = new_peak - last_peak

            # ── Warm-up phase ──────────────────────────────────────────
            if len(self._recent_rr_intervals) < adaptive_window:
                if len(self._recent_rr_intervals) >= 2:
                    expected_rr = np.median(self._recent_rr_intervals)

                    if rr_to_new < expected_rr * 0.5:
                        continue
                    if rr_to_new > expected_rr * threshold_multiplier:
                        # Start a new segment while retaining the recent RR history.
                        self._current_segment = [new_peak]
                        self._window_all_accepted.append(new_peak)
                        continue

                if rr_to_new < self.min_rr:
                    continue
                if rr_to_new > self.max_rr:
                    self._current_segment = [new_peak]
                    self._window_all_accepted.append(new_peak)
                    continue

                self._current_segment.append(new_peak)
                self._window_all_accepted.append(new_peak)
                self._recent_rr_intervals.append(rr_to_new)

            # ── Steady-state phase ────────────────────────────────────
            else:
                expected_rr = np.median(self._recent_rr_intervals)
                threshold   = expected_rr * threshold_multiplier

                if rr_to_new > threshold:
                    # Start a new segment while retaining the recent RR history.
                    self._current_segment = [new_peak]
                    self._window_all_accepted.append(new_peak)

                elif rr_to_new < expected_rr * 0.5:
                    if len(self._current_segment) >= 2:
                        # Keep the peak whose RR interval is closer to the expected rhythm.
                        second_last  = self._current_segment[-2]
                        rr_with_last = last_peak - second_last
                        rr_with_new  = new_peak  - second_last

                        if abs(rr_with_new - expected_rr) < abs(rr_with_last - expected_rr):
                            self._current_segment[-1]     = new_peak
                            self._recent_rr_intervals[-1] = rr_with_new
                            self._window_all_accepted.append(new_peak)
                    else:
                        reference_rr = self._recent_rr_intervals[-1]
                        if rr_to_new >= reference_rr * 0.5:
                            self._current_segment.append(new_peak)
                            self._window_all_accepted.append(new_peak)
                            self._recent_rr_intervals.append(rr_to_new)
                            if len(self._recent_rr_intervals) > adaptive_window:
                                self._recent_rr_intervals.pop(0)

                else:
                    self._current_segment.append(new_peak)
                    self._window_all_accepted.append(new_peak)
                    self._recent_rr_intervals.append(rr_to_new)
                    if len(self._recent_rr_intervals) > adaptive_window:
                        self._recent_rr_intervals.pop(0)

    def _compute_bpm(self, has_new_peaks=True):
        """
        If new peaks were accepted this window, compute BPM from the last
        4 peaks (3 RR intervals). Otherwise, return the last valid BPM.

        Args:
            has_new_peaks: Whether new peaks were added to _current_segment
                           in this window.

        Returns:
            dict with "bpm", "timestamp", and "status".
        """
        result = {"bpm": None, "timestamp": None, "status": "insufficient_data"}

        seg = self._current_segment
        timestamp = float(self._global_sample_offset / self.sample_rate)

        if not has_new_peaks:
            # No new peaks — repeat last valid BPM
            if self._last_computed_bpm is not None:
                return {"bpm": self._last_computed_bpm, "timestamp": timestamp, "status": "ok"}
            return result

        if len(seg) < 4:
            return result

        # Use last 4 peaks → 3 RR intervals
        last4 = seg[-4:]
        rr_intervals = [last4[j + 1] - last4[j] for j in range(3)]
        rr_avg = np.mean(rr_intervals)
        bpm = 60.0 * self.sample_rate / rr_avg

        if config.HR_MIN_BPM < bpm < config.HR_MAX_BPM:
            bpm = round(bpm, 1)
            return {"bpm": bpm, "timestamp": timestamp, "status": "ok"}
        else:
            return {"bpm": round(bpm, 1), "timestamp": timestamp, "status": "invalid"}

    def _filter_sudden_change(self, result):
        """Limit sudden BPM changes relative to the previous computed value.

        Changes at or above config.HR_CHANGE_THRESHOLD are restricted to one
        threshold step and marked as "filtered".
        """
        if result["status"] != "ok":
            return result

        bpm = result["bpm"]

        # First valid reading — accept unconditionally
        if self._last_computed_bpm is None:
            self._last_computed_bpm = bpm
            return result

        delta = bpm - self._last_computed_bpm
        if abs(delta) >= config.HR_CHANGE_THRESHOLD:
            # Limit the change to one threshold step.
            if delta > 0:
                bpm = self._last_computed_bpm + config.HR_CHANGE_THRESHOLD
            else:
                bpm = self._last_computed_bpm - config.HR_CHANGE_THRESHOLD
            result["bpm"] = round(bpm, 1)
            result["status"] = "filtered"

        self._last_computed_bpm = result["bpm"]

        return result

    def _smooth_bpm(self, result, sigma=1):
        """
        Apply Gaussian smoothing over a rolling buffer of recent BPM values.
        """
        if result["bpm"] is None:
            return result

        self._bpm_history.append(result["bpm"])

        if len(self._bpm_history) >= 3:
            smoothed = gaussian_filter1d(self._bpm_history, sigma=sigma)
            result["bpm"] = round(float(smoothed[-1]), 1)

        return result
