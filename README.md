# SwimHR Artifact

This repository contains the artifacts for the ACM MobiCom 2026 paper *SwimHR: Wearable-Free Heart Rate Monitoring for Swimmers via Water-Mediated ECG Sensing*. It includes the hardware design, signal-processing software, sample dataset, and simulation models. The artifact is organized into three main folders, with the sample data included under `software/` to support direct execution of the processing example.

## Repository Structure

```text
.
├── hardware/    Hardware design source and PCB views
├── software/    Processing software and sample dataset
└── simulation/  COMSOL models, exported results, and analysis notebook
```



## Hardware

The `hardware/` folder contains the design files for the SwimHR bias-elimination circuit.

- `ProPrj_Bias-elimination-final.epro` is the editable EasyEDA project containing the schematic and PCB layouts.
- `Bias-elimination.png` provides a visual preview of the PCB layouts.
- `PCB_Bias-elimination.pdf` provides a detailed PCB view for inspection.



## Software

The `software/` folder contains the offline ECG processing pipeline, including signal preprocessing, neural-network-based peak detection, heart-rate estimation, and the trained model weights.

`run_example.py` processes the sample SwimHR recording, compares its heart-rate estimates with four reference measurements, and saves the resulting figure as `software/output/sample_hr_comparison.png`.

The required Conda environment is defined in `environment.yml`. Create and activate the environment, then run the example:

```bash
cd SwimHR_MobiCom26_Artifact
conda env create -f environment.yml
conda activate swimhr-artifact
python software/run_example.py
```

##### Sample Dataset

The `software/sample-dataset/` folder contains one sample SwimHR ECG recording and time-stamped heart-rate measurements from four reference devices.

The authors collected these data from their own swimming session in a 25 m pool. The ECG was recorded at 125 Hz. Reference streams are aligned by their timestamps and refined within a ±10 s window by minimizing mean absolute error.

- `sample_swimhr_ecg.txt`: 16-channel OpenBCI recording sampled at 125 Hz; the example uses channels 0–12.
- `sample_polar_h10_ground_truth.csv`: Polar H10 chest-belt heart rate at 1 Hz.
- `sample_apple_watch.csv`: irregularly sampled Apple Watch heart rate.
- `sample_polar_verity_sense_forearm.csv`: Polar Verity Sense forearm heart rate at 1 Hz.
- `sample_polar_verity_sense_temple.csv`: Polar Verity Sense temple heart rate at 1 Hz.

Original timestamps are retained to support synchronization across devices.



## Simulation

The `simulation/` folder contains models developed in COMSOL Multiphysics 6.2. These models examine the effects of pool size, water conductivity, reinforced concrete boundaries, and a metal ladder on the simulated ECG signal. Compact model files are stored in `simulation/models/`, while exported numerical results are stored in `simulation/results/`.

The notebook `simulation/figures/simulation_results.ipynb` reads the exported results, converts voltage from volts to microvolts, calculates peak-to-peak amplitude, and reproduces the simulation figures. Run the notebook from `simulation/figures/` so that its relative paths resolve correctly.

The generated `pool_size.png` and `water_conductivity.png` correspond to Figure 15 in the paper appendix.
