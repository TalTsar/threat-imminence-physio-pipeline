# Psychophysiology-Signal-Analysis

## Overview
This repo outlines the architecture and methodology of an automated signal processing pipeline designed for psychological and physiological research. The pipeline extracts, cleans, and analyzes continuous physiological signals—specifically Electromyography (EMG) and Electrocardiogram (ECG/HR)—time-locked to experimental stimuli (e.g., thermal pain/heat, startle paradigms).

Python Libraries: NumPy, Pandas, Matplotlib, SciPy (Signal filtering, IIR/Butterworth design), NeuroKit2 (ECG cleaning/peak detection)


#### Note: The full source code and datasets for this specific experiment are currently confidential. This repository serves as a methodological description of the signal processing and data engineering techniques utilized.


## Core Methodologies

### 1. Event-Related Epoching & Trigger Logic
The pipeline dynamically segments the continuous data streams based on digital event triggers.

* Dynamic Windowing: Extracts specific trial phases, including pre-stimulus Baseline, Fixation, Stimulus (Heat/Startle), and Inter-Trial Intervals (ITI).

* Time-Binning: Slices the post-stimulus response into 2-second analytical bins to track the temporal dynamics of the physiological reaction.

* Baseline Correction: Automatically calculates the pre-stimulus baseline metric and applies normalization to the active trial segments (specifically for EMG amplitude).

### 2. ECG to Continuous Heart Rate (HR) Pipeline

* Filtering: Applied a 3rd-order bandpass filter (5-25 Hz) to isolate the QRS complex.

* Peak Detection: Utilized NeuroKit2 for algorithmic R-peak detection and QRS cleaning.

* HR & Artifact Rejection: Calculated beat-to-beat interval (BPM). Implemented a local median-replacement algorithm to identify and correct unphysiological spikes/drops (e.g., < 40 BPM or > 120 BPM) caused by movement artifacts.

### 3. EMG Amplitude & Envelope Extraction

* Noise Reduction: Applied iterative IIR Notch filtering to remove 50Hz electrical line noise and its harmonics.

* Frequency Isolation: Utilized a 4th-order Butterworth bandpass filter (28-400Hz) to isolate the biological muscle frequency band.

* Rectification & Smoothing: Converted the raw signal to absolute values (rectification) and applied a 3rd-order 2Hz lowpass filter to extract the linear signal envelope, providing a clean metric of mean muscle activation per trial phase.
