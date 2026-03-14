import pandas as pd
import numpy as np
import scipy.io as sio
import os
from glob import glob
from datetime import datetime

import config
import LoadAcq
import ProcessSignals
import GetExperimentMetrics


date = datetime.now().strftime('%d%m%Y')

# Define input paths
acqFiles = os.path.join(config.acqDataFolder, '*.acq')
dataDir = os.path.join(config.matDataFolder, '*.mat')

levels_dict = config.levels_dict

# ---- Initialize Result Tables -----
all_data_emg = []
all_data_hr = []
all_data_scr = []

""" ================================================================
                            MAIN
    ================================================================"""

def main():

    if config.loadNew:
        LoadAcq.LoadFiles(acqFiles, dataDir)

    file_paths = sorted(glob(dataDir))


    for i, file_path in enumerate(file_paths[config.start_idx:], start=config.start_idx):
        subject_id = os.path.basename(file_path).split('_')[1]

        # Load Data
        data_mat = sio.loadmat(file_path)
        raw_signals = data_mat['Sig']
        fs = int(data_mat['Fs'][0][0])

        # triggers logic
        triggers = raw_signals[:, config.trig_col_idx]
        triggers = ProcessSignals.clean_repeated_triggers(triggers)

        # --- Handle Experiment Scope (Start/End) ---
        exp_scope_100 = np.where(triggers == 100)[0]
        exp_scope_255 = np.where(triggers == 255)[0]
        start_idx_sig = exp_scope_100[0] if len(exp_scope_100) > 0 else exp_scope_255[0]

        exp_end = np.where(triggers == 92)[0]
        end_idx_sig = exp_end[-1] if len(exp_end) > 0 else len(triggers)

        # Crop signals to experiment start
        triggers = triggers[start_idx_sig:end_idx_sig]
        emg_sig = raw_signals[start_idx_sig:end_idx_sig, config.emg_col_idx]
        hr_sig = raw_signals[start_idx_sig:end_idx_sig, config.hr_col_idx]
        eda_sig = raw_signals[start_idx_sig:end_idx_sig, config.eda_col_idx]

        # ----- Downsample Signals -----
        signals = [emg_sig, hr_sig, eda_sig]
        signals, triggers, fs = ProcessSignals.downsample_signals(signals, triggers, fs)
        emg_sig, hr_sig, eda_sig = signals


        #                               --- PROCESS EMG ---
        if config.runEMG:
            if int(subject_id) not in config.skip_emg and int(subject_id) not in config.skip_glob :
                print(f"Processing EMG for Subject {subject_id} idx {i}...")

                clean_emg = ProcessSignals.preprocess_emg_signal(emg_sig, fs)

                freqs, psd, is_noisy, metrics = ProcessSignals.analyze_emg_psd(clean_emg, fs, False)

                if not is_noisy:
                    emg_rows = (GetExperimentMetrics.extract_experiment_metrics
                                (clean_emg, triggers, fs, levels_dict, signal_type='EMG'))

                    for row in emg_rows:
                        row['subject'] = subject_id
                        all_data_emg.append(row)
                else:
                    print(f"NOISY - Skipping EMG for Subject {subject_id}...")

            else:
                print(f"Skipping EMG for Subject {subject_id} idx {i}...")

        #                               --- PROCESS HR ---
        if config.runECG:
            if int(subject_id) not in config.skip_hr and int(subject_id) not in config.skip_glob:
                print(f"Processing ECG for Subject {subject_id} idx {i}...")

                clean_hr, clean_ibi = ProcessSignals.preprocess_hr_signal(hr_sig, fs)

                hr_rows = (GetExperimentMetrics.extract_experiment_metrics
                           (clean_hr, triggers, fs, levels_dict, signal_type='HR'))

                for row in hr_rows:
                    row['subject'] = subject_id
                    all_data_hr.append(row)
            else:
                print(f"Skipping ECG for Subject {subject_id}...")

        #                               --- PROCESS SCR ---
        if config.runSCR:
            if (int(subject_id) not in config.skip_scr and int(subject_id) not in config.skip_glob ) and eda_sig is not None:
                print(f"Processing SCR for Subject {subject_id}...")

                # Decompose and get Phasic component
                clean_scr_phasic = ProcessSignals.preprocess_scr_signal(eda_sig, fs)

                scr_rows = (GetExperimentMetrics.extract_experiment_metrics
                            (clean_scr_phasic, triggers, fs, levels_dict, signal_type='SCR'))

                for row in scr_rows:
                    row['subject'] = subject_id
                    all_data_scr.append(row)
            else:
                 print(f"Skipping SCR for Subject {subject_id}...")

    # ----- Export Results -----
    def save_table(all_data, type):
        if not all_data:
            print(f"No data to save for {type}.")
            return
        df = pd.DataFrame(all_data)
        timestamp = datetime.now().strftime('%d%m%Y')
        df.to_excel(f'TIM_{type}_{timestamp}.xlsx', index=False)
        print(f"Done saving {type}.")

    if config.runEMG:
        save_table(all_data_emg, 'EMG')
    if config.runECG:
        save_table(all_data_hr, 'HR')
    if config.runSCR:
        save_table(all_data_scr, 'SCR')

if __name__ == '__main__':
    main()