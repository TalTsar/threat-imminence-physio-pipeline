import pandas as pd
import numpy as np
import scipy.io as sio
import os
from glob import glob
from datetime import datetime

import paths
import LoadAcq
import ProcessSignals
import GetExperimentMetrics


""" ================================================================
                        DEFINE EXPERIMENT CONFIG
    ================================================================"""


start_idx = 0
loadNew = False

runEMG= True
runECG = True
runSCR = True

# Skip lists

skip_glob = [384,369,424,355]
skip_emg = [491, 397, 362, 401,402,410,411,413, 414,416]
skip_hr = [340,352,421,434,440,459]
skip_scr = [340,341,401,410,414]

unsure=[]

date = datetime.now().strftime('%d%m%Y')
levels_dict = {21: 'Low', 41: 'Moderate', 81: 'High'}


# Define input paths

acqFiles = os.path.join(paths.acqDataFolder, '*.acq')
dataDir = os.path.join(paths.matDataFolder, '*.mat')

# ---- Define Biopac Channels ----
eda_col_idx = 0
hr_col_idx = 1
emg_col_idx = 2
trig_col_idx = 13

# ---- Initialize Result Tables -----
all_data_emg = []
all_data_hr = []
all_data_scr = []

""" ================================================================
                            MAIN
    ================================================================"""

def main():

    if loadNew:
        LoadAcq.LoadFiles(acqFiles, dataDir)

    file_paths = sorted(glob(dataDir))


    for i, file_path in enumerate(file_paths[start_idx:], start=start_idx):
        subject_id = os.path.basename(file_path).split('_')[1]

        # Load Data
        data_mat = sio.loadmat(file_path)
        raw_signals = data_mat['Sig']
        fs = int(data_mat['Fs'][0][0])

        # triggers logic
        triggers = raw_signals[:, trig_col_idx]
        triggers = ProcessSignals.clean_repeated_triggers(triggers)

        # --- Handle Experiment Scope (Start/End) ---
        exp_scope_100 = np.where(triggers == 100)[0]
        exp_scope_255 = np.where(triggers == 255)[0]
        start_idx_sig = exp_scope_100[0] if len(exp_scope_100) > 0 else exp_scope_255[0]

        exp_end = np.where(triggers == 92)[0]
        end_idx_sig = exp_end[-1] if len(exp_end) > 0 else len(triggers)

        # Crop signals to experiment start
        triggers = triggers[start_idx_sig:end_idx_sig]
        emg_sig = raw_signals[start_idx_sig:end_idx_sig, emg_col_idx]
        hr_sig = raw_signals[start_idx_sig:end_idx_sig, hr_col_idx]
        eda_sig = raw_signals[start_idx_sig:end_idx_sig, eda_col_idx]

        # ----- Downsample Signals -----
        signals = [emg_sig, hr_sig, eda_sig]
        signals, triggers, fs = ProcessSignals.downsample_signals(signals, triggers, fs)
        emg_sig, hr_sig, eda_sig = signals


        # --- PROCESS EMG ---
        if runEMG:
            if int(subject_id) not in skip_emg and int(subject_id) not in skip_glob :
                print(f"Processing EMG for Subject {subject_id} idx {i}...")

                freqs, psd, is_noisy, metrics = ProcessSignals.analyze_emg_psd(emg_sig, fs, False)
                if not is_noisy:
                    clean_emg = ProcessSignals.preprocess_emg_signal(emg_sig, fs)
                    emg_rows = (GetExperimentMetrics.extract_experiment_metrics
                                (clean_emg, triggers, fs, levels_dict, signal_type='EMG'))
                    for row in emg_rows:
                        row['subject'] = subject_id
                        all_data_emg.append(row)
                else:
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

        # --- PROCESS HR ---
        if runECG:
            if int(subject_id) not in skip_hr and int(subject_id) not in skip_glob:
                print(f"Processing ECG for Subject {subject_id} idx {i}...")
                clean_hr, clean_ibi = ProcessSignals.preprocess_hr_signal(hr_sig, fs)

                hr_rows = GetExperimentMetrics.extract_experiment_metrics(clean_hr, triggers, fs, levels_dict, signal_type='HR')
                for row in hr_rows:
                    row['subject'] = subject_id
                    all_data_hr.append(row)
            else:
                print(f"Skipping ECG for Subject {subject_id}...")

        # --- PROCESS SCR (New) ---
        if runSCR:
            if (int(subject_id) not in skip_scr and int(subject_id) not in skip_glob ) and eda_sig is not None:
                print(f"Processing SCR for Subject {subject_id}...")
                # Decompose and get Phasic component
                clean_scr_phasic = ProcessSignals.preprocess_scr_signal(eda_sig, fs)

                # Extract metrics using the same logic (Mean of phasic)
                scr_rows = (GetExperimentMetrics.extract_experiment_metrics
                            (clean_scr_phasic, triggers, fs, levels_dict, signal_type='SCR'))
                for row in scr_rows:
                    row['subject'] = subject_id
                    all_data_scr.append(row)
            else:
                 print(f"Skipping SCR for Subject {subject_id}...")

    # Export
    def save_table(all_data, type):
        if not all_data:
            print(f"No data to save for {type}.")
            return
        df = pd.DataFrame(all_data)
        timestamp = datetime.now().strftime('%d%m%Y')
        df.to_excel(f'TIM_{type}_{timestamp}.xlsx', index=False)
        print(f"Done saving {type}.")

    if runEMG:
        save_table(all_data_emg, 'EMG')
    if runECG:
        save_table(all_data_hr, 'HR')
    if runSCR:
        save_table(all_data_scr, 'SCR')

if __name__ == '__main__':
    main()