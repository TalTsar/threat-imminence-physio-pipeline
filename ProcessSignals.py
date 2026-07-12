
import numpy as np
import pandas as pd
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
    # 1. Calculate PSD using Welch's method
    def interp_nans(x):
        nans = np.isnan(x)
        if np.any(nans):
            not_nans = ~nans
            x[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(not_nans), x[not_nans])
        return x

    emg_signal_clean = interp_nans(emg_signal)

    freqs, psd = welch(emg_signal_clean, fs, nperseg=int(fs / 2))

    # Restrict analysis strictly to the relevant EMG band (0 to 500 Hz)
    valid_idx = np.where(freqs <= 500)[0]
    f_band = freqs[valid_idx]
    p_band = psd[valid_idx]

    # 2. Calculate Spectral Shape Metrics
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

    # 3. Check for Lost Cause / Contamination - ADDED KURTOSIS
    is_noisy = False
    if (spec_kurtosis > kurtosis_thresh and spec_entropy < entropy_thresh) or (hpr > hpr_thresh):
        is_noisy = True

    metrics = {
        "Entropy": spec_entropy,
        "Kurtosis": spec_kurtosis,
        "HPR": hpr
    }

    # 4. Visualization
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

        # Prevent the labels from the top plot from overlapping the title of the bottom plot
        plt.tight_layout()
        plt.show()

    return freqs, psd, is_noisy, metrics




def analyze_emg_broadband(emg_signal, fs, plotExp=False, hf_power_thresh=1e-4, hf_ratio_thresh=0.25):
    """
    Calculates PSD and identifies broadband noise contamination.
    Safely handles NaN values within the input signal.

    Parameters:
    emg_signal: 1D array of raw EMG data (can contain NaNs)
    fs: Sampling frequency (Hz)
    hf_power_thresh: Maximum allowed absolute power in 300-500 Hz (adjusted for scientific notation)
    hf_ratio_thresh: Maximum allowed percentage of total power residing in 300-500 Hz
    """
    # --- 1. SAFELY HANDLE NAN VALUES ---
    emg_signal = np.array(emg_signal, dtype=np.float64).copy()
    nan_mask = np.isnan(emg_signal)

    if np.any(nan_mask):
        # If the entire array is NaNs, we can't process it
        if np.all(nan_mask):
            print("Warning: Signal is entirely NaNs. Skipping analysis.")
            return None, None, True, {}

        # Repair NaNs using linear interpolation to preserve timing structure
        x_indices = np.arange(len(emg_signal))
        emg_signal[nan_mask] = np.interp(x_indices[nan_mask], x_indices[~nan_mask], emg_signal[~nan_mask])

    # --- 2. COMPUTE PSD (Identical to before) ---
    freqs, psd = welch(emg_signal, fs, nperseg=int(fs / 2))

    valid_idx = np.where(freqs <= 500)[0]
    f_band = freqs[valid_idx]
    p_band = psd[valid_idx]

    total_power = np.sum(p_band)

    hf_mask = (f_band >= 300) & (f_band <= 500)
    hf_power = np.sum(p_band[hf_mask])

    hf_ratio = hf_power / (total_power + 1e-10)

    psd_norm = p_band / (total_power + 1e-10)
    spec_entropy = entropy(psd_norm)
    spec_kurtosis = kurtosis(p_band)

    is_noisy = False
    if (hf_power > hf_power_thresh) or (hf_ratio > hf_ratio_thresh):
        is_noisy = True

    metrics = {
        "HF_Absolute_Power": hf_power,
        "HF_Power_Ratio": hf_ratio,
        "Entropy": spec_entropy,
        "Kurtosis": spec_kurtosis
    }

    # --- 3. VISUALIZATION (With formatting fixes) ---
    if plotExp:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        ax1.semilogy(f_band, p_band, color='teal')

        title_text = 'PSD of EMG - FLAGGED AS BROADBAND NOISE' if is_noisy else 'PSD of EMG (Clean Baseline)'
        ax1.set_title(title_text, color='red' if is_noisy else 'black', fontweight='bold')

        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Power/Freq (V^2/Hz)')
        ax1.grid(True, which='both')
        ax1.set_xlim(0, 500)

        ax1.axvspan(300, 500, color='red' if is_noisy else 'gray', alpha=0.1, label='Noise Floor Zone (300-500Hz)')
        ax1.legend(loc='lower left')

        # CHANGED TO UNIFORM SCIENTIFIC NOTATION (:.2e) SO TINY VALUES DON'T SHOW AS 0.00
        textstr = '\n'.join((
            f'HF Abs Power: {hf_power:.2e} (Max: {hf_power_thresh:.2e})',
            f'HF Power Ratio: {hf_ratio:.2%} (Max: {hf_ratio_thresh:.0%})',
            f'Spectral Entropy: {spec_entropy:.2f}',
            f'Spectral Kurtosis: {spec_kurtosis:.2f}'
        ))
        props = dict(boxstyle='round', facecolor='white', alpha=0.9)
        ax1.text(0.95, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='right', bbox=props)

        time_vector = np.arange(len(emg_signal)) / fs
        ax2.plot(time_vector, emg_signal, color='steelblue', linewidth=0.8)
        ax2.set_title('Raw Time-Series of EMG (NaNs Repaired)')
        ax2.set_xlabel('Time (Seconds)')
        ax2.set_ylabel('Amplitude (mV)')
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
    clipped_ecg = np.clip(raw_ecg, -1.1, 1.1)

    # ----- Filtering -----
    clean_ecg = dynamic_notch_filter(clipped_ecg, fs)
    bpf_ecg = butter_filter(clean_ecg, fs, low=0.5, high = 40, order=3, btype='band')

    return bpf_ecg


def validate_peaks_by_template(raw_ecg, rr_peaks, fs, threshold=0.7):
    if len(rr_peaks) < 30:
        return rr_peaks  # Not enough peaks to reliably grab a clean template

    # 1. Define a QRS window size (e.g., 100ms total -> 50ms before, 50ms after)
    win = int(0.05 * fs)

    # 2. Extract a template from a known clean zone (e.g., the 15th peak)
    template_peak = rr_peaks[15]
    template = raw_ecg[template_peak - win: template_peak + win]

    # Double check template boundaries are safe
    if len(template) != (2 * win):
        return rr_peaks

    valid_peaks = []

    # 3. Cross-correlate every peak against the template
    for peak in rr_peaks:
        if peak - win < 0 or peak + win > len(raw_ecg):
            continue

        segment = raw_ecg[peak - win: peak + win]

        # Calculate Pearson correlation coefficient
        corr = np.corrcoef(template, segment)[0, 1]

        # If it looks like a real QRS complex, keep it
        if corr >= threshold:
            valid_peaks.append(peak)

    return np.array(valid_peaks)



def calculate_hr_signal(raw_ecg, fs,method='hr'):


    clean_seg_ecg = raw_ecg.copy()


    dynamic_height = np.mean(clean_seg_ecg) + (2.8 * np.std(clean_seg_ecg))

    rr_peaks, _ = find_peaks(
        clean_seg_ecg,
        height=dynamic_height,
        distance=int(0.3 * fs)
    )

    rr_peaks_neg, _ = find_peaks(
        -clean_seg_ecg,
        height=dynamic_height,
        distance=int(0.3 * fs)
    )

    if len(rr_peaks) < 2 and len(rr_peaks_neg) < 2:
        print("not enough peaks")
        return None

    if len(rr_peaks_neg) > len(rr_peaks) and len(rr_peaks_neg) > 10:
        rr_peaks = rr_peaks_neg

    rr_peaks_val = validate_peaks_by_template(clean_seg_ecg, rr_peaks, fs, threshold=0.8)

    # plt.figure()
    # # --- Plot 1: Filtered ECG ---
    # t_full = np.arange(len(raw_ecg)) / fs
    # plt.plot(t_full, clean_seg_ecg, label='Filtered ECG', color='teal', alpha=0.7)
    # plt.plot(t_full[rr_peaks], clean_seg_ecg.iloc[rr_peaks], "x", label='Peaks Positive', color='red')
    # plt.plot(t_full[rr_peaks_val], clean_seg_ecg.iloc[rr_peaks_val], "x", label='Peaks Positive', color='blue')
    # plt.title(f'Detected R-Peaks')
    # plt.ylabel('Amplitude')
    # # plt.legend(loc='upper right')
    # plt.grid(True, linestyle='--', alpha=0.5)
    # plt.tight_layout()
    # plt.show(block=True)
    # plt.close()

    rr_peaks = rr_peaks_val
    if len(rr_peaks) > len(rr_peaks_val):
        print("checkpoint")

    if method == 'ibi':

        ibi_sec = np.diff(rr_peaks) / fs
        ibi_ms = ibi_sec * 1000

        # ----- Artifact detection (RR-based) -----
        med_ibi = np.median(ibi_ms)

        # Define your physiological boundaries (in seconds)
        MIN_IBI_PHYS = 300  # E.g., ~170 BPM
        MAX_IBI_PHYS = 1500  # E.g., ~40 BPM

        artifact_mask = (
                (ibi_ms < 0.5 * med_ibi) |
                (ibi_ms > 1.5 * med_ibi) |
                (ibi_ms < MIN_IBI_PHYS) |
                (ibi_ms > MAX_IBI_PHYS)
        )

        ibi_ms_corr = ibi_ms.copy()
        ibi_ms_corr[artifact_mask] = np.median(ibi_ms_corr[~artifact_mask])

        signal = ibi_ms_corr


    elif method == 'hr':
        rr_diff = np.diff(rr_peaks)
        rr_sec = rr_diff / fs

        # Convert to a pandas Series to easily use rolling windows
        rr_series = pd.Series(rr_sec)

        # 1. Compute a rolling local median (window of 11 beats centers it nicely)
        # min_periods=1 ensures it still calculates values near the edges
        local_med_rr = rr_series.rolling(window=9, center=True, min_periods=1).median().to_numpy()

        # 2. Calculate relative percentage change between consecutive beats
        consecutive_diffs = np.abs(np.diff(rr_sec))
        pct_change = np.zeros_like(rr_sec)
        pct_change[1:] = consecutive_diffs / rr_sec[:-1]

        # 3. Hard physiological limits
        MIN_RR_PHYS = 0.3  # ~200 BPM
        MAX_RR_PHYS = 1.5  # ~40 BPM
        MAX_PCT_CHANGE = .5  # Allow up to a 25% instantaneous jump

        # 4. Build the mask using the LOCAL median instead of the global one
        artifact_mask = (
                (rr_sec < 0.50 * local_med_rr) |  # Back to a reasonable threshold, but local!
                (rr_sec > 1.50 * local_med_rr) |
                (rr_sec < MIN_RR_PHYS) |
                (rr_sec > MAX_RR_PHYS)
        )

        # Flag the trailing edge of a single sharp spike if necessary
        artifact_mask[:-1] = artifact_mask[:-1] | (pct_change[1:] > MAX_PCT_CHANGE)

        # Convert mask to numpy boolean array
        artifact_mask = np.array(artifact_mask, dtype=bool)

        rr_sec_corr = rr_sec.copy()
        rr_sec_corr[artifact_mask] = np.nan

        artifacts_pct = (np.sum(artifact_mask) * 100) / len(artifact_mask)
        print(f"Artifacts Percent: {artifacts_pct:.2f}%")

        if artifacts_pct >= 70: # should not even bother with interpolation
            return None


        hr_bpm = 60 / rr_sec_corr
        signal = hr_bpm

        # Calculate midpoints for peak times
    t_pks = (rr_peaks[1:] + rr_peaks[:-1]) / 2 / fs

    # 3. Create a boolean mask tracking only the true, clean data points
    valid_mask = ~np.isnan(signal)

    fs_rr = 1000
    t_uniform = np.arange(t_pks[0], t_pks[-1], 1 / fs_rr)

    # 4. Interpolate ONLY using the valid times and signals
    signal_uniform = np.interp(
        t_uniform,
        t_pks[valid_mask],  # Drops the bad time coordinates
        signal[valid_mask]  # Drops the bad HR coordinates
    )

    if len(signal_uniform) < 21:
        return None

    if method == 'ibi':
        signal_smooth = butter_filter(signal_uniform, fs=fs_rr,
                                  low=0.01, high=.2, order=3, btype="band")
    else:
        signal_smooth = signal_uniform


    rr_len = len(rr_peaks)
    if artifacts_pct > 20 or rr_len < 30:
        print(f'Num Peaks: {rr_len} ; Artifacts Percent: {artifacts_pct}')

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

        # --- Plot 1: ECG + peaks ---
        t_full = np.arange(len(raw_ecg)) / fs

        ax1.plot(t_full, clean_seg_ecg, label='Filtered ECG', color='teal', alpha=0.7)
        ax1.plot(t_full[rr_peaks], clean_seg_ecg.iloc[rr_peaks], "x", color='red', label='Peaks')
        ax1.set_title('Detected R-Peaks')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, linestyle='--', alpha=0.5)

        # --- Plot 2: Smooth signal ---
        ax2.plot(t_uniform, signal_smooth, color='purple', label='Smooth Signal')
        ax2.set_title('Smooth Signal')
        ax2.set_ylabel('HR / Signal')
        ax2.set_xlabel('Time (s)')
        ax2.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()
        plt.close()


        return None

    return signal_smooth, rr_len




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


