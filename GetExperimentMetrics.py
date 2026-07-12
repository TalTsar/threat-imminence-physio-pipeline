import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        pad_width = int(3 * fs / 5)  # depends on filter length
        segment = np.pad(segment, pad_width, mode='reflect')
        envelope = ProcessSignals.butter_filter(np.abs(segment), fs, low=5, order=3, btype='low')

        return np.median(envelope)

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
    trials = {
        int(idx): float(val)
        for idx, val in zip(startles_idx, startles_vals)
        if int(val) in levels_dict
    }

    # print(f"Num Blocks: {len(np.where(triggers == 100)[0])}")

    prev_level = "Start"
    for n, (idx, trig_code) in enumerate(trials.items(), start=1):

        if trig_code  not in levels_dict:
            continue

        level = levels_dict[trig_code]
        block = ((n - 1) // 6) + 1

        # Define window: 8s before to 30s after
        start_sample = idx - int(fs * 8)
        tmp_sig = signal[start_sample:]
        tmp_trig = triggers[start_sample:]

        tmp_iti = np.where(tmp_trig % 10 == 8)[0][0]
        if tmp_iti < 31000 or tmp_iti > 34000:
            print('Issue with segment')
            continue

        end_sample = tmp_iti + int(5 * fs)

        if start_sample < 0 or end_sample > len(signal):
            continue

        full_seg = tmp_sig[:end_sample]
        full_trig = tmp_trig[:end_sample]

        if signal_type == 'HR':
            seg_ecg = pd.Series(full_seg)
            result = ProcessSignals.calculate_hr_signal(seg_ecg, fs,'hr')

            if result is None:
                continue

            full_seg, rr_len = result




        # import viewSpecificSubjects as view
        # view.plotSigAndTrig(full_seg, full_trig, fs)


        # --- 1. Baseline Correction (ITI period before fixation) ---
        # baseline_seg = full_seg[:int(fs * 4)]
        # pre_ITI = get_metric(baseline_seg, signal_type, fs)

        baseline_val = get_metric(full_seg[int(fs * 4):int(fs * 8)], signal_type, fs) # Fixation

        if baseline_val is None:
            print('Skipping segment')
            continue

        # Helper to get value and correct baseline
        def calc_val(seg):
            val = get_metric(seg, signal_type, fs)
            # Apply baseline correction for EMG and SCR (Phasic)
            if signal_type in ['SCR']:
                return val
            elif signal_type in ['EMG','HR']:
                return val
            return val

        # --- 2. Extract Phases ---

        # Fixation (4s to 8s)
        # fix_val = calc_val(full_seg[int(fs * 4):int(fs * 8)])
        results.append({'ID': None, 'Block': block,'Magnitude': level,
                        'Imminence': 0, 'Val': baseline_val,'prev_level': prev_level})


        # --- Heat, Rating, ITI ---
        t6_indices = np.where(full_trig % 10 == 6)[0]
        t7_indices = np.where(full_trig % 10 == 7)[0]
        t8_indices = np.where(full_trig % 10 == 8)[0]

        # Logic for Heat (6 -> 7)
        if t6_indices.size > 0 and t7_indices.size > 0:
            idx_6 = t6_indices[0]

            heat14_val = calc_val(full_seg[idx_6:idx_6 + int(fs * 4)])
            heat48_val = calc_val(full_seg[idx_6 + int(fs * 4):idx_6 + int(fs * 8)])

            results.append({'ID': None, 'Block': block, 'Magnitude': level, 'Imminence': 5, 'Val': heat14_val,'prev_level': prev_level
                            # 'preITI': baseline_val,'prev_level': prev_level
                            })
            results.append({'ID': None, 'Block': block, 'Magnitude': level, 'Imminence': 6, 'Val': heat48_val,'prev_level': prev_level
                            })

        # Logic for Rating & ITI (7 -> 8)
        if t7_indices.size > 0 and t8_indices.size > 0:
            idx_7 = t7_indices[0]
            idx_8 = t8_indices[0]

            trig7_val = calc_val(full_seg[idx_7:idx_8])
            results.append({'ID': None, 'Block': block, 'Magnitude': level,
                            'Imminence': 7, 'Val': trig7_val,'prev_level': prev_level})

            # ITI (8 to 8+5s)
            iti_val = calc_val(full_seg[idx_8:])
            results.append({'ID': None,'Block': block, 'Magnitude': level,
                            'Imminence': 8, 'Val': iti_val,'prev_level': prev_level})

        # --- 3. Bins Analysis (0-8s in 2s steps) ---
        for bin_i, bin_start_sec in enumerate(range(0, 8, 2), start=1):
            bin_seg = full_seg[int((8 + bin_start_sec) * fs): int((8 + bin_start_sec + 2) * fs)]
            bin_val = calc_val(bin_seg)

            results.append({'ID': None,'Block':block, 'Magnitude': level,
                            'Imminence': bin_i, 'Val': bin_val,'prev_level': prev_level })
        prev_level = level

    return results
