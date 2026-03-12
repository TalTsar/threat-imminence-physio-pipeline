# TIM Experiment Pipeline

## Overview
Pipeline for extracting, cleaning, and analyzing continuous physiological signals:
Electromyography (EMG), Electrocardiogram (ECG → Heart Rate), and Electro Dermal Activity (EDA → SCR).

Libraries: NumPy, Pandas, Matplotlib, SciPy (signal filtering), NeuroKit2 (ECG cleaning and R-peak detection).

## Core Methodologies

### 1. Event-Related Epoching & Trigger Logic
The pipeline dynamically segments the continuous data streams based on digital event triggers.

* Dynamic Windowing: Extracts specific trial phases, including pre-stimulus Baseline, Fixation, Stimulus (Heat/Startle), and Inter-Trial Intervals (ITI).

* Time-Binning: Slices the post-stimulus response into 2-second analytical bins to track the temporal dynamics of the physiological reaction.

* Baseline Correction: Automatically calculates the pre-stimulus baseline metric and applies normalization to the active trial segments (specifically for EMG amplitude).

### 2. Heart Rate Extraction

* Filtering: 0.5Hz highpass filter
  
* Peak Detection: Utilized NeuroKit2 for algorithmic R-peak detection and QRS cleaning.

* HR & Artifact Rejection: Calculated beat-to-beat interval (BPM). Implemented a local median-replacement algorithm to identify and correct unphysiological spikes/drops (e.g., < 40 BPM or > 140 BPM).

### 3. EMG Envelope Extraction

* Noise Reduction: Applied iterative IIR Notch filtering to remove 50Hz electrical line noise and its harmonics.

* Frequency Isolation: Utilized a 4th-order Butterworth bandpass filter (28-400Hz) to isolate the biological muscle frequency band.

* Rectification & Smoothing: Converted the raw signal to absolute values (rectification) and applied a 3rd-order 2Hz lowpass filter to extract the linear signal envelope, providing a clean metric of mean muscle activation per trial phase.

### 4. SCR Extraction

* Decomposition: Utilized NeuroKit 2 for decomposition of the signal to Tonic and Phasic components using convex optimization method.

* SCR calculation: Defined as the global EDA - tonic component
