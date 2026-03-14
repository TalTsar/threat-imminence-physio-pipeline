
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, find_peaks, stft, welch, resample
from scipy.stats import entropy, kurtosis
from scipy.interpolate import PchipInterpolator
import neurokit2 as nk
import matplotlib.pyplot as plt


""" ================================================================
                            SIGNAL PROCESSING 
  ================================================================"""

def downsample_array(arr, factor):
    return arr[::factor]

def update_triggers(triggers, ds_factor, target_len):
    trig_idx = np.where(triggers != 0)[0]
    trig_vals = triggers[trig_idx]
    new_idx = np.round(trig_idx / ds_factor).astype(int)
    triggers_ds = np.zeros(target_len, dtype=triggers.dtype)
    valid = new_idx < target_len
    triggers_ds[new_idx[valid]] = trig_vals[valid]
    return triggers_ds

def downsample_signals(signals, triggers, fs, ds_factor=2):
    signals_ds = [downsample_array(sig, ds_factor) for sig in signals]
    fs_ds = fs / ds_factor
    triggers_ds = update_triggers(triggers, ds_factor, len(signals_ds[0]))
    return signals_ds, triggers_ds, fs_ds

def clean_repeated_triggers(signal):
    signal = signal.copy()
    n = len(signal)
    for i in range(n - 1):
        # if current value equals next value and is > 0, zero out current
        if signal[i] > 0 and signal[i] == signal[i + 1]:
            signal[i] = 0
    return signal

def butter_filter(data, fs, low=None, high=None, order=4, btype='band'):
    nyq = fs / 2
    if btype == 'band':
        Wn = [low / nyq, high / nyq]
    elif btype == 'low':
        Wn = low / nyq
    elif btype == 'high':
        Wn = high / nyq

    b, a = butter(order, Wn, btype=btype)
    return filtfilt(b, a, data)


def dynamic_notch_filter(data, fs, base_noise=50, max_freq=400):
    """Dynamically finds and removes only active powerline harmonics."""

    freqs, psd = welch(data, fs, nperseg=fs * 2)
    psd_db = 10 * np.log10(np.maximum(psd, 1e-12))

    # Find peaks with high prominence (sharp spikes sticking out of the fuzz)
    peaks, _ = find_peaks(psd_db, prominence=5.0)
    found_peak_freqs = freqs[peaks]

    # Cross-reference peaks with expected harmonics
    target_notches = []
    for peak_freq in found_peak_freqs:
        if peak_freq < 10:
            continue

        if peak_freq <= max_freq:
            # Check if the peak is within +/- 2 Hz of a 50Hz multiple
            remainder = peak_freq % base_noise
            if remainder <= 2 or remainder >= (base_noise - 2):
                target_notches.append(np.round(peak_freq))

    # Apply the notches only if we found verified noise spikes
    if target_notches:
        print(f"Active noise detected. Applying notches at: {target_notches} Hz")
        notch_width = 1.0
        for f0 in set(target_notches):
            Q_dynamic = f0 / notch_width
            b, a = iirnotch(w0=f0 / (fs / 2), Q=Q_dynamic)
            clean_data = filtfilt(b, a, data)
    else:
        clean_data = data
        print("Signal is clean of powerline harmonics. Skipping notches!")

    return clean_data



def analyze_emg_psd(emg_signal, fs,plotExp=False, entropy_thresh=4.0, hpr_thresh=2.0, kurtosis_thresh=20.0):
    """
    Calculates PSD and identifies potential noise contamination based on spectral shape.

    Parameters:
    emg_signal: 1D array of raw EMG data
    fs: Sampling frequency (Hz)
    entropy_thresh: Minimum allowed spectral entropy
    hpr_thresh: Maximum allowed Harmonic-to-Physiological Ratio
    kurtosis_thresh: Maximum allowed spikiness (catches massive harmonic spikes)
    """
    # Calculate PSD using Welch's method
    freqs, psd = welch(emg_signal, fs, nperseg=int(fs / 2))

    # Restrict analysis strictly to the relevant EMG band (0 to 500 Hz)
    valid_idx = np.where(freqs <= 500)[0]
    f_band = freqs[valid_idx]
    p_band = psd[valid_idx]

    # Calculate Spectral Shape Metrics
    psd_norm = p_band / np.sum(p_band)
    spec_entropy = entropy(psd_norm)
    spec_kurtosis = kurtosis(p_band)

    # Harmonic Power Ratio (HPR) - FIXED TO USE A WINDOW
    harmonics = [50, 100, 150, 200, 250, 300, 350, 400, 450]
    noise_mask = np.zeros_like(f_band, dtype=bool)

    # Catch power within +/- 2 Hz of every harmonic
    for h in harmonics:
        noise_mask |= (f_band >= h - 2) & (f_band <= h + 2)

    noise_power = np.sum(p_band[noise_mask])

    # Physiological power is the 28-400Hz band, explicitly excluding the noise windows
    physio_mask = (f_band >= 28) & (f_band <= 400) & (~noise_mask)
    physio_power = np.sum(p_band[physio_mask])

    hpr = noise_power / (physio_power + 1e-10)

    # Check for Lost Cause / Contamination - ADDED KURTOSIS
    is_noisy = False
    if (spec_kurtosis > kurtosis_thresh and spec_entropy < entropy_thresh) or (hpr > hpr_thresh):
        is_noisy = True

    metrics = {
        "Entropy": spec_entropy,
        "Kurtosis": spec_kurtosis,
        "HPR": hpr
    }

    # Visualization
    if plotExp:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # --- Top Subplot: PSD ---
        ax1.semilogy(f_band, p_band)

        title_text = f'PSD of EMG - FLAGGED AS NOISY' if is_noisy else f'PSD of EMG'
        ax1.set_title(title_text, color='red' if is_noisy else 'black', fontweight='bold')

        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Power/Freq (V^2/Hz)')
        ax1.grid(True, which='both')
        ax1.set_xlim(0, 500)

        textstr = '\n'.join((
            f'Entropy: {spec_entropy:.2f}',
            f'Kurtosis: {spec_kurtosis:.2f}',
            f'HPR: {hpr:.2f}'
        ))
        props = dict(boxstyle='round', facecolor='white', alpha=0.9)
        ax1.text(0.95, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='right', bbox=props)

        if is_noisy:
            for h in harmonics[:5]:  # Plot lines for the first few harmonics
                ax1.axvline(h, color='r', linestyle='--', alpha=0.3)

        # --- Bottom Subplot: Raw Time-Series ---
        # Create a time vector based on the sampling frequency
        time_vector = np.arange(len(emg_signal)) / fs

        ax2.plot(time_vector, emg_signal, color='steelblue', linewidth=0.8)
        ax2.set_title(f'Raw Time-Series of EMG')
        ax2.set_xlabel('Time (Seconds)')
        ax2.set_ylabel('Amplitude')
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    return freqs, psd, is_noisy, metrics

def preprocess_emg_signal(raw_emg, fs):
    """Clean EMG: Artifact Suppress -> Targeted Notches -> Bandpass -> NaN Artifacts"""

    clean_emg = np.clip(raw_emg, -1, 1)

    clean_emg = dynamic_notch_filter(clean_emg, fs)

    emg_bpf = butter_filter(clean_emg, fs, low=28, high=400, btype='band')

    emg_bpf[np.abs(raw_emg) > 1.0] = np.nan

    return emg_bpf




def preprocess_hr_signal(raw_ecg, fs):
    """
    HR Pipelin with NeuroKit2: Clean -> Peaks -> HR calculation -> Artifact Removal
    Returns: Continuous HR (BPM) matching the original Fs
    """

    signals, info = nk.ecg_process(raw_ecg, sampling_rate=fs)
    hr_continuous = signals['ECG_Rate'].to_numpy()


    return hr_continuous

def preprocess_scr_signal(raw_eda, fs):

    target_fs = 50
    num_samples_original = len(raw_eda)
    num_samples_target = int(num_samples_original * target_fs / fs)

    try:

        eda_small = resample(raw_eda, num_samples_target)

        eda_cleaned = nk.eda_clean(eda_small, sampling_rate=target_fs)
        eda_decomposed = nk.eda_phasic(eda_cleaned, sampling_rate=target_fs, method='cvxEDA')
        tonic_small = eda_decomposed['EDA_Tonic'].values

        eda_global_tonic = eda_cleaned - tonic_small

        phasic_restored = resample(eda_global_tonic, num_samples_original)
        return phasic_restored

    except Exception as e:
        print(f"Error in SCR processing: {e}")
        return np.zeros_like(raw_eda)


