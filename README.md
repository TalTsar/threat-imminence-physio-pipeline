# Threat Imminence Experiment Pipeline

## Overview
Pipeline for extracting, cleaning, and analyzing continuous physiological signals:
Electromyography (EMG), Electrocardiogram (ECG → Heart Rate), and Electro Dermal Activity (EDA → SCR). 
Automates metric extraction according to experiment phases.

Libraries: NumPy, Pandas, Matplotlib, SciPy (signal filtering), NeuroKit2 (ECG & EDA).

## Methods

### 1. Heart Rate Extraction (NeuroKit2)

* Filtering: 0.5Hz highpass filter.
  
* HR & Artifact Rejection: Calculated from R-Peak detection and represented in beats per minute (BPM). Local median-replacement algorithm identifies and corrects unphysiological spikes/drops ( < 40 BPM or > 140 BPM).

### 2. EMG Envelope Extraction

* Noise Reduction:  IIR Notch filtering to remove 50Hz electrical line noise and its harmonics.

* Frequency Isolation: 4th-order Butterworth bandpass filter (28-400Hz).

* Rectification & Smoothing: Rectification and a 3rd-order 2Hz lowpass filter to extract the signal envelope.

### 3. SCR Extraction (NeuroKit2)

* Decomposition: Decomposition  to Tonic and Phasic components using convex optimization method.

* SCR calculation: Defined as the global EDA minus tonic component
