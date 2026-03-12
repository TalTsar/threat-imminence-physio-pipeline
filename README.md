# Threat Imminence Experiment Pipeline

## Overview
Pipeline for extracting, cleaning, and analyzing continuous physiological signals:
Electromyography (EMG), Electrocardiogram (ECG → Heart Rate), and Electro Dermal Activity (EDA → SCR). 
Automates metric extraction according to experiment phases.

Libraries: NumPy, Pandas, Matplotlib, SciPy (signal filtering), NeuroKit2 (ECG cleaning and R-peak detection).

## Methods

### 1. Heart Rate Extraction

* Filtering: 0.5Hz highpass filter
  
* Peak Detection: Utilized NeuroKit2 for algorithmic R-peak detection and QRS cleaning.

* HR & Artifact Rejection: Calculated beat-to-beat interval (BPM). Implemented a local median-replacement algorithm to identify and correct unphysiological spikes/drops (e.g., < 40 BPM or > 140 BPM).

### 2. EMG Envelope Extraction

* Noise Reduction: Applied iterative IIR Notch filtering to remove 50Hz electrical line noise and its harmonics.

* Frequency Isolation: Utilized a 4th-order Butterworth bandpass filter (28-400Hz).

* Rectification & Smoothing: Rectified and applied a 3rd-order 2Hz lowpass filter to extract the signal envelope.

### 3. SCR Extraction

* Decomposition: Utilized NeuroKit 2 for decomposition of the signal to Tonic and Phasic components using convex optimization method.

* SCR calculation: Defined as the global EDA minus tonic component
