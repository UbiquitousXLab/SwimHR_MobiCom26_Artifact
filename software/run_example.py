"""Run the SwimHR pipeline and plot four reference devices for the sample recording."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from processing.heart_rate import HeartRateCalculator
from processing.peak_detector import PeakDetector
from processing.preprocessor import ECGPreprocessor


SOFTWARE_DIR = Path(__file__).resolve().parent
DATASET_DIR = SOFTWARE_DIR / "sample-dataset"
OUTPUT_PATH = SOFTWARE_DIR / "output" / "sample_hr_comparison.png"


def load_swimhr_ecg():
    """Load the first 13 EXG channels and the original recording start time."""
    path = DATASET_DIR / "sample_swimhr_ecg.txt"
    frame = pd.read_csv(path, comment="%", skipinitialspace=True)
    columns = [f"EXG Channel {channel}" for channel in range(config.NUM_CHANNELS)]
    raw_ecg = frame[columns].to_numpy(dtype=np.float32)
    start_time = pd.to_datetime(frame["Timestamp (Formatted)"].iloc[0])
    return start_time, raw_ecg


def load_polar_hr(filename):
    """Load absolute timestamps and 1 Hz HR values from a Polar CSV."""
    path = DATASET_DIR / filename
    metadata = pd.read_csv(path, nrows=1).iloc[0]
    recording_start = pd.to_datetime(
        f"{metadata['Date']} {metadata['Start time']}",
        format="%d-%m-%Y %H:%M:%S",
    )
    samples = pd.read_csv(path, header=2)
    timestamps = recording_start + pd.to_timedelta(samples["Time"])
    return pd.DatetimeIndex(timestamps), samples["HR (bpm)"].to_numpy(dtype=float)


def load_apple_watch_hr():
    """Load Apple Watch HR and forward-fill its irregular samples to 1 Hz."""
    frame = pd.read_csv(DATASET_DIR / "sample_apple_watch.csv")
    frame["timestamp"] = pd.to_datetime(frame["startDate"]).dt.tz_localize(None)
    series = frame.sort_values("timestamp").set_index("timestamp")["value"].resample("1s").ffill()
    return series.index, series.to_numpy(dtype=float)


def run_swimhr(raw_ecg):
    """Replay the four-second pipeline at a one-second update interval."""
    preprocessor = ECGPreprocessor()
    peak_detector = PeakDetector()
    heart_rate = HeartRateCalculator()

    stride_samples = config.SAMPLE_RATE
    window_samples = 4 * config.SAMPLE_RATE
    context_samples = int(0.5 * config.SAMPLE_RATE)
    buffer_samples = window_samples + 2 * context_samples
    rolling_buffer = None
    timestamps = []
    bpm_values = []

    for start in range(0, len(raw_ecg) - stride_samples + 1, stride_samples):
        chunk = raw_ecg[start:start + stride_samples]
        if rolling_buffer is None:
            rolling_buffer = np.tile(chunk[0], (buffer_samples, 1)).astype(np.float32)
        rolling_buffer = np.roll(rolling_buffer, -stride_samples, axis=0)
        rolling_buffer[-stride_samples:] = chunk

        filtered_full, normalized_full = preprocessor.process(rolling_buffer)
        usable = slice(context_samples, context_samples + window_samples)
        peaks, _ = peak_detector.detect(filtered_full[usable], normalized_full[usable])
        result = heart_rate.compute(peaks)
        if result["bpm"] is not None:
            timestamps.append(result["timestamp"])
            bpm_values.append(result["bpm"])

    return np.asarray(timestamps, dtype=float), np.asarray(bpm_values, dtype=float)


def minimize_mae_alignment(hr_target, hr_reference_full, start, end, search_range=(-10, 10)):
    """Find the reference offset that minimizes MAE against the target HR."""
    best_offset = 0
    min_mae = float("inf")

    for offset in range(search_range[0], search_range[1] + 1):
        adj_start = start + offset
        adj_end = end + offset

        if adj_start < 0 or adj_end > len(hr_reference_full):
            continue

        hr_ref_slice = hr_reference_full[adj_start:adj_end]
        if len(hr_ref_slice) != len(hr_target):
            continue

        current_mae = np.mean(np.abs(hr_target - hr_ref_slice))
        if current_mae < min_mae:
            min_mae = current_mae
            best_offset = offset

    return best_offset, min_mae


def align_reference_sensors(our_hr_bpm, our_hr_times_seconds, open_bci_ts, ref_data):
    """Anchor SwimHR to H10, then align the other sensors to the H10 slice."""
    # Store each sensor's minimum MAE and aligned 1 Hz HR segment.
    mae_results = {}
    aligned_hrs = {}
    reference_hr = None

    # Use timestamps for the initial H10 position, then refine it within ±10 seconds.
    if "chest_belt" in ref_data:
        belt = ref_data["chest_belt"]
        delta = open_bci_ts - belt["timestamp"]
        start = int(delta.total_seconds()) + int(our_hr_times_seconds[0])
        end = start + len(our_hr_times_seconds)
        offset, mae = minimize_mae_alignment(
            our_hr_bpm, belt["hr_vector"], start, end
        )
        mae_results["swimhr_vs_h10"] = float(mae)
        aligned_hrs["chest_belt"] = np.asarray(
            belt["hr_vector"][start + offset:end + offset]
        )
        reference_hr = aligned_hrs["chest_belt"]

    # The remaining sensors require the aligned H10 segment as their reference.
    if reference_hr is None:
        return mae_results, aligned_hrs

    # Timestamp-align each wearable, then independently search ±10 seconds against H10.
    for sensor in ("apple_watch", "forearm", "temple"):
        info = ref_data.get(sensor)
        if info is None:
            continue
        delta = open_bci_ts - info["timestamp"]
        start = int(delta.total_seconds()) + int(our_hr_times_seconds[0])
        end = start + len(our_hr_times_seconds)
        offset, mae = minimize_mae_alignment(
            reference_hr, info["hr_vector"], start, end
        )
        mae_results[f"{sensor}_vs_h10"] = float(mae)
        aligned_hrs[sensor] = np.asarray(
            info["hr_vector"][start + offset:end + offset]
        )

    return mae_results, aligned_hrs


def plot_comparison(ecg_start, duration_s, swimhr_time, swimhr_bpm):
    """Plot four aligned references and return their MAEs against SwimHR."""
    apple_time, apple_hr = load_apple_watch_hr()
    belt_time, belt_hr = load_polar_hr("sample_polar_h10_ground_truth.csv")
    forearm_time, forearm_hr = load_polar_hr(
        "sample_polar_verity_sense_forearm.csv"
    )
    temple_time, temple_hr = load_polar_hr(
        "sample_polar_verity_sense_temple.csv"
    )
    ref_data = {
        "apple_watch": {"hr_vector": apple_hr, "timestamp": apple_time[0]},
        "chest_belt": {"hr_vector": belt_hr, "timestamp": belt_time[0]},
        "forearm": {"hr_vector": forearm_hr, "timestamp": forearm_time[0]},
        "temple": {"hr_vector": temple_hr, "timestamp": temple_time[0]},
    }
    alignment_mae, aligned_hrs = align_reference_sensors(
        swimhr_bpm, swimhr_time, ecg_start, ref_data
    )
    references = [
        ("apple_watch", "Apple Watch"),
        ("chest_belt", "ECG Chest Belt"),
        ("forearm", "PPG Forearm"),
        ("temple", "PPG Temple"),
    ]

    fig, axis = plt.subplots(figsize=(14, 7))
    for sensor, label in references:
        axis.plot(swimhr_time, aligned_hrs[sensor], label=label, alpha=0.7, linewidth=1.5)
    axis.plot(swimhr_time, swimhr_bpm, label="SwimHR", color="red", linewidth=2.5)

    axis.set_xlim(0, duration_s)
    axis.set_xlabel("Time [s]", fontsize=14)
    axis.set_ylabel("Heart Rate [bpm]", fontsize=14)
    axis.set_title(f"SwimHR Sample - {duration_s:.0f}s", fontsize=16)
    axis.legend(loc="best", fontsize=11)
    axis.grid(True, alpha=0.3)
    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200)
    plt.close(fig)
    return alignment_mae


def main():
    """Run the sample pipeline and save the comparison figure."""
    ecg_start, raw_ecg = load_swimhr_ecg()
    swimhr_time, swimhr_bpm = run_swimhr(raw_ecg)
    duration_s = len(raw_ecg) / config.SAMPLE_RATE
    mae = plot_comparison(ecg_start, duration_s, swimhr_time, swimhr_bpm)
    print(f"SwimHR vs ECG Chest Belt: {mae['swimhr_vs_h10']:.2f} bpm")
    print(f"Polar Verity Sense forearm vs ECG Chest Belt: {mae['forearm_vs_h10']:.2f} bpm")
    print(f"Apple Watch vs ECG Chest Belt: {mae['apple_watch_vs_h10']:.2f} bpm")
    print(f"Polar Verity Sense temple vs ECG Chest Belt: {mae['temple_vs_h10']:.2f} bpm")
    print(f"Saved {len(swimhr_bpm)} SwimHR estimates to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
