import numpy as np
import ProcessSignals

""" ================================================================
                            GET EXPERIMENT METRICS
    ================================================================"""



def get_metric(segment, signal_type, fs):
    """
    Central switch for calculating the value of a segment
    based on the signal type.
    """
    if signal_type == 'EMG':
        segment = segment * 1000
        envelope = ProcessSignals.butter_filter(np.abs(segment), fs, low=2, order=3, btype='low')
        return np.mean(envelope)

    elif signal_type == 'HR':
        return np.nanmean(segment)

    elif signal_type == 'SCR':
        val = np.nanmean(segment)
        if val<0:
            return 0
        return val

    return np.nan

def extract_experiment_metrics(signal, triggers, fs, levels_dict, signal_type):
    """
    Loops through events and cuts segments for Fixation, Heat, ITI, and Bins.
    Returns: A list of dictionaries (rows) for the dataframe.
    """
    results = []

    # Identify trials (Startles/Bin1)
    startles_idx = np.where((triggers % 10) == 1)[0]
    startles_vals = triggers[startles_idx]
    trials = {int(idx): float(val) for idx, val in zip(startles_idx, startles_vals)}

    # print(f"Num Blocks: {len(np.where(triggers == 100)[0])}")

    prev_level = "Start"
    for n, (idx, trig_code) in enumerate(trials.items(), start=1):

        if trig_code in levels_dict:
            level = levels_dict[trig_code]

        # Define window: 8s before to 30s after
        start_sample = idx - int(fs * 8)
        end_sample = idx + int(30 * fs)

        if start_sample < 0 or end_sample > len(signal):
            continue

        full_seg = signal[start_sample:end_sample]
        full_trig = triggers[start_sample:end_sample]

        # --- 1. Baseline Correction (ITI period before fixation) ---
        baseline_seg = full_seg[:int(fs * 4)]
        baseline_val = get_metric(baseline_seg, signal_type, fs)

        # Helper to get value and correct baseline
        def calc_val(seg):
            val = get_metric(seg, signal_type, fs)
            # Apply baseline correction for EMG and SCR (Phasic)
            if signal_type in ['SCR']:
                return val
            elif signal_type in ['EMG','HR']:
                return val - baseline_val
            return val

        # --- 2. Extract Phases ---

        # Fixation (4s to 8s)
        fix_val = calc_val(full_seg[int(fs * 4):int(fs * 8)])
        results.append({'subject': None, 'level': level, 'bin': 'Fixation', 'val': fix_val, 'preITI': baseline_val, 'prev_level': prev_level})

        # --- Heat, Rating, ITI ---
        t6_indices = np.where(full_trig % 10 == 6)[0]
        t7_indices = np.where(full_trig % 10 == 7)[0]
        t8_indices = np.where(full_trig % 10 == 8)[0]

        # Logic for Heat (6 -> 7)
        if t6_indices.size > 0 and t7_indices.size > 0:
            idx_6 = t6_indices[0]
            idx_7 = t7_indices[0]
            trig6_val = calc_val(full_seg[idx_6:idx_7])
            results.append({'subject': None, 'level': level, 'bin': 'Heat', 'val': trig6_val, 'preITI': baseline_val, 'prev_level': prev_level})

        # Logic for Rating & ITI (7 -> 8)
        if t7_indices.size > 0 and t8_indices.size > 0:
            idx_7 = t7_indices[0]
            idx_8 = t8_indices[0]

            heat_val = calc_val(full_seg[idx_7:idx_8])
            results.append({'subject': None, 'level': level, 'bin': 'Rating', 'val': heat_val, 'preITI': baseline_val, 'prev_level': prev_level})

            # ITI (8 to 8+5s)
            iti_end = idx_8 + int(5 * fs)
            if iti_end <= len(full_seg):
                iti_val = calc_val(full_seg[idx_8:iti_end])
                results.append({'subject': None, 'level': level, 'bin': 'ITI', 'val': iti_val, 'preITI': baseline_val, 'prev_level': prev_level})

        # --- 3. Bins Analysis (0-8s in 2s steps) ---
        for bin_i, bin_start_sec in enumerate(range(0, 8, 2), start=1):
            bin_seg = full_seg[int((8 + bin_start_sec) * fs): int((8 + bin_start_sec + 2) * fs)]
            bin_val = calc_val(bin_seg)

            results.append({
                'subject': None,
                'level': level,
                'bin': bin_i,
                'val': bin_val,
                'preITI': baseline_val,
                'prev_level': prev_level
            })
        prev_level = level

    return results
