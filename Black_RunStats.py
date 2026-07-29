import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Qt5Agg')

import os
from datetime import datetime

# import functions from other files
import config as config
from statAnalysis.lme import run_lme
from statAnalysis.util import get_latest_file,filter_subs
from statAnalysis.modifyTable import winsorize_group,merge_tables
from statAnalysis.subtractions.sub_describe import sub_describe,heat_diff
from statAnalysis.corrs import corr_randSlope_Anx, corr_sub_Anx
from statAnalysis.ttests import run_contrast_ttest

#
timestamp = datetime.now().strftime('%d%m%Y')

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

cats = ['Low', 'High']

markers = {'Low': 'o', 'High': 's'}
line_styles = {'Low': '--', 'High': '-'}


def run_analysis_pipeline(signal_type, file_path, unit_label):
    print(f"\n\n{'=' * 40}")
    print(f"        STARTING ANALYSIS: {signal_type}")
    print(f"{'=' * 40}")

    # 1. Load Data
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    df = pd.read_excel(file_path)

    df = filter_subs(df,signal_type)

    mergedTables = merge_tables(df,file_path,signal_type,DEMOG)

    mergedTables['Val_clean'] = ( mergedTables
        .groupby(['Imminence', 'Magnitude'])['Val']
        .transform(winsorize_group) )

    changed_win_only = (mergedTables['Val_clean'] != mergedTables['Val']).sum()
    changed_prec_win = round((changed_win_only / len(mergedTables)) * 100, 2)
    print(f"{signal_type}: {changed_win_only} rows changed with Z-Score based Winsorizing ({changed_prec_win}%)")


    # ----------- create separate tables -----------
    df_cont_anticipation = mergedTables[mergedTables['Imminence'].isin([1, 2, 3, 4])].copy()

    df_cont_anticipation['Magnitude_H_L_B'] = df_cont_anticipation['Magnitude'].apply(
        lambda x: 'Black' if 'Black' in x else ('Low' if 'Low' in x else 'High')
    )

    df_cont_anticipation['Magnitude_H_L_B'] = pd.Categorical(
        df_cont_anticipation['Magnitude_H_L_B'],
        categories=['Low', 'High', 'Black'],
        ordered=False
    )

    mapping = {'Start': 0, 'Low': 1, 'High': 2, 'Black': 3}
    df_cont_anticipation['Magnitude_H_L_B'] = df_cont_anticipation['Magnitude_H_L_B'].map(mapping)

    df_heat = mergedTables[mergedTables['Imminence'] == 5].copy()
    df_heat['Magnitude'] = df_heat['Magnitude'].astype('category')

    df_heat['Magnitude_H_L'] = df_heat['Magnitude'].apply(lambda x: 'Low' if 'Low' in x else 'High')
    df_heat['Magnitude_H_L'] = pd.Categorical(df_heat['Magnitude_H_L'], categories=['Low', 'High'], ordered=False)
    df_heat['Magnitude_H_L'] = df_heat['Magnitude_H_L'].map({'Low': 1, 'High': 2})

    df_heat['Type'] = df_heat['Magnitude'].apply(lambda x: 'Black' if 'Black' in x else 'Regular')
    df_heat['Type'] = pd.Categorical(df_heat['Type'], categories=['Regular', 'Black'], ordered=False)
    df_heat['Type'] = df_heat['Type'].map({'Regular': 1, 'Black': 2})

    heat_diff(df_heat)


    # =========================================================
    # TIME-BIN SHIFTS: ISOLATE AND PIVOT BINS
    # =========================================================

    df_time = mergedTables[mergedTables['Imminence'].isin([1, 4, 5, 6])].copy()
    df_time_subj = df_time.groupby(['ID', 'Magnitude', 'Imminence'], as_index=False, observed=True)['Val_clean'].mean()

    df_time_pivot = df_time_subj.pivot(
        index=['ID', 'Magnitude'],
        columns='Imminence',
        values='Val_clean'
    ).reset_index()

    df_time_pivot = df_time_pivot.rename(columns={1: 'Bin1', 4: 'Bin4', 5: 'Heat14', 6: 'Heat48'})

    df_time_pivot['Diff_4_minus_1'] = df_time_pivot['Bin4'] - df_time_pivot['Bin1']
    df_time_pivot['Diff_5_minus_4'] = df_time_pivot['Heat14'] - df_time_pivot['Bin4']
    df_time_pivot['Diff_6_minus_4'] = df_time_pivot['Heat48'] - df_time_pivot['Bin4']
    df_time_pivot['Diff_6_minus_5'] = df_time_pivot['Heat48'] - df_time_pivot['Heat14']


    cleaned_time_diffs = sub_describe(df_time_pivot)

    # =========================================================
    # STAT ANALYSIS: Anticipation
    # =========================================================
    target_variables = ['Val_clean']

    for target in target_variables:
        print(f"\n" + "=" * 60)
        print(f" STARTING ANALYSIS FOR TARGET: {target.upper()}")
        print("=" * 60)

        df_clean = df_cont_anticipation.dropna(subset=[target]).copy()
        df_clean['ID'] = df_clean['ID'].astype('string')
        df_clean['Gender'] = df_clean['Gender'].astype('category')
        df_clean['Imminence_c'] = df_clean['Imminence'] - df_clean['Imminence'].mean()

        formula = f'{target} ~ Magnitude_H_L_B * Imminence_c + Block + Age + Gender'

        res, df_with_slopes = run_lme(df_clean, fixed_formula=formula, random_formula="~Imminence_c", group_var="ID",
                                      method="lbfgs")


        corr_randSlope_Anx(df_with_slopes)

        # =========================================================
        # HEAT MODELS
        # =========================================================
        formula = f'{target} ~ Magnitude_H_L * Type + Block + Age + Gender'
        res = run_lme(df_heat, fixed_formula=formula, random_formula="~1", group_var="ID", method="lbfgs",
                      extractSlopes=False)

        df_plot_time = pd.concat(cleaned_time_diffs, ignore_index=True)

        if 'Age' in mergedTables.columns and 'Gender' in mergedTables.columns:
            df_demog_sub = mergedTables.drop_duplicates(subset='ID')
            df_lme_diff = (df_plot_time[
                               ['ID', 'Magnitude', 'Contrast', 'Difference', 'Comp1_Val', 'Comp2_Val', 'Comp1_Name',
                                'Comp2_Name']]
                           .merge(df_demog_sub[['ID', 'Gender', 'Age',
                                                'GAD7_total','STAI_total','PHQ9_total','PCL5_total','IUS_total']], on='ID', how='left'))
            diff_formula = 'Difference ~ Magnitude_H_L * Type + Age + Gender'
        else:
            df_lme_diff = df_plot_time.copy()
            diff_formula = 'Difference ~ Magnitude_H_L * Type'

        df_lme_diff['Magnitude_H_L'] = df_lme_diff['Magnitude'].apply(lambda x: 'Low' if 'Low' in x else 'High')
        df_lme_diff['Magnitude_H_L'] = pd.Categorical(df_lme_diff['Magnitude_H_L'], categories=['Low', 'High'],
                                                      ordered=False)
        df_lme_diff['Magnitude_H_L'] = df_lme_diff['Magnitude_H_L'].map({'Low': 1, 'High': 2})

        df_lme_diff['Type'] = df_lme_diff['Magnitude'].apply(lambda x: 'Black' if 'Black' in x else 'Regular')
        df_lme_diff['Type'] = pd.Categorical(df_lme_diff['Type'], categories=['Regular', 'Black'], ordered=False)
        df_lme_diff['Type'] = df_lme_diff['Type'].map({'Regular': 1, 'Black': 2})

        df_lme_diff['ID'] = df_lme_diff['ID'].astype('string')
        df_lme_diff['Magnitude_H_L'] = df_lme_diff['Magnitude_H_L'].astype('category')
        if 'Gender' in df_lme_diff.columns:
            df_lme_diff['Gender'] = df_lme_diff['Gender'].astype('category')

        magnitude_pairs = [
            ('Low', 'Low-Black'),
            ('High', 'High-Black')
        ]
        contrast_label_map = config.contrast_label_map

        run_contrast_ttest(df_lme_diff)

        corr_sub_Anx(df_lme_diff,contrast_label_map)


# =========================================================
# MAIN EXECUTION LOOP
# =========================================================
if __name__ == "__main__":
    DATA_FOLDER = config.output_folder
    BLACK = config.black
    PALETTE = config.palette_dict
    ORDER = config.order
    EXCLUSIONS = config.exclusions
    CUSTOM_X_LABELS = config.custom_x_labels
    DEMOG = config.demographic_df

    signals_to_process = [
        ('EMG', 'Amplitude (uV)'),  #Baseline Corrected
        ('HR', 'Heart Rate (BPM)'),
        # ('SCR', 'Phasic (√uS)')
    ]

    for sig_type, unit in signals_to_process:
        file_path = get_latest_file(sig_type, DATA_FOLDER)
        print(file_path)

        if file_path:
            run_analysis_pipeline(sig_type, file_path, unit)
        else:
            print(f"Skipping {sig_type} (No file found).")