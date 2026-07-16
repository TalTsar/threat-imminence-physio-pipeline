import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Qt5Agg')
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import zscore, skew
import pingouin as pg
from pathlib import Path
from scipy.stats.mstats import winsorize
from scipy.stats import pearsonr
import matplotlib.transforms as transforms
import glob
import os
from datetime import datetime

import config_TIMblack as config

timestamp = datetime.now().strftime('%d%m%Y')

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

cats = ['Low', 'High']

markers = {'Low': 'o', 'High': 's'}
line_styles = {'Low': '--', 'High': '-'}


def get_latest_file(signal_type, folder_path):
    """Finds the most recent Excel file for a given signal type."""
    search_pattern = os.path.join(folder_path, f"TIM_{signal_type}_*.xlsx")
    files = glob.glob(search_pattern)
    if not files:
        print(f"No files found for {signal_type} in {folder_path}")
        return None
    # Sort by modification time
    return max(files, key=os.path.getmtime)


def run_analysis_pipeline(signal_type, file_path, unit_label):
    print(f"\n\n{'=' * 40}")
    print(f"        STARTING ANALYSIS: {signal_type}")
    print(f"{'=' * 40}")

    # 1. Load Data
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    df = pd.read_excel(file_path)

    # Descriptive counters
    valCountPerBlock = df.groupby(['ID', 'Block'])[['Val']].count()
    valMeansPerCondition = df.groupby(['Magnitude', 'Imminence'])['Val'].mean()
    valMeansPerConditionPerSubj = df.groupby(['Magnitude', 'Imminence', 'ID'])['Val'].mean()

    # 2. Filtering
    bad_subs = EXCLUSIONS.get(signal_type, [])
    global_bad_subs = EXCLUSIONS.get('GLOB', [])

    bad_df = df[df['ID'].isin(bad_subs)]
    bad_valMeansPerConditionPerSubj = bad_df.groupby(['Magnitude', 'Imminence', 'ID'])['Val'].mean()
    bad_valCountPerBlock = bad_df.groupby(['ID', 'Block'])[['Val']].count()

    df = df[~df['ID'].isin(bad_subs)]
    valCountPerBlock_postExclusion = df.groupby(['ID', 'Block'])[['Val']].count()

    # remove NaNs
    df = df.dropna(subset=['Val'])
    print(f"Loaded {len(df)} rows. Subjects: {df['ID'].nunique()}")

    # ------- Z-score guided Winsorize (per group) -------
    def winsorize_group(x, thresh=3):
        x = x.astype(float)
        z = zscore(x)
        in_range = x[(z >= -thresh) & (z <= thresh)]

        if len(in_range) == 0:
            return x

        min_in_range = in_range.min()
        max_in_range = in_range.max()

        x_wins = x.copy()
        x_wins[z < -thresh] = min_in_range
        x_wins[z > thresh] = max_in_range

        return x_wins

    # ------- General LME function -------
    def run_lme(df, fixed_formula, random_formula="~1", group_var="ID", method="nm", vc_formula=None,
                extractSlopes=True):
        df = df.copy()

        mdl = smf.mixedlm(
            fixed_formula,
            data=df,
            groups=df[group_var],
            re_formula=random_formula,
            vc_formula=vc_formula
        )

        res_mdl = mdl.fit(method=method)

        if extractSlopes:
            sub_slopes = pd.DataFrame.from_dict(
                res_mdl.random_effects,
                orient='index'
            )
            sub_slopes.index.name = 'ID'
            sub_slopes = sub_slopes.rename(columns={
                'Group': 'Intercept',
                'Imminence_c': 'Imminence_slope'
            })
            sub_slopes = sub_slopes.reset_index()

            df_merged = df.merge(sub_slopes, on='ID', how='left')

            print(f"\n=== {fixed_formula} + ( {random_formula} | {group_var} ) ===")
            if vc_formula is not None:
                print(f"    + independent slopes via vc_formula: {list(vc_formula.keys())}")

            print(res_mdl.summary())

            # --- NEW: Print ANOVA-style Table ---
            print("\n=== ANOVA Table (Wald Tests for Fixed Effects) ===")
            print(res_mdl.wald_test_terms(scalar=True).summary_frame())

            return res_mdl, df_merged

        # Fallback path if extractSlopes=False
        print(f"\n=== {fixed_formula} + ( {random_formula} | {group_var} ) ===")
        if vc_formula is not None:
            print(f"    + independent slopes via vc_formula: {list(vc_formula.keys())}")

        print(res_mdl.summary())

        # --- NEW: Print ANOVA-style Table ---
        print("\n=== ANOVA Table (Wald Tests for Fixed Effects) ===")
        print(res_mdl.wald_test_terms().summary_frame())

        return res_mdl

    # ----------- Stats with Demog -----------
    if DEMOG:
        demog = pd.read_excel(DEMOG)
        mergedTables = df.merge(demog[['Sub_ID','Gender','Age','GAD7_total','STAI_total','PHQ9_total','PCL5_total','IUS_total']], left_on='ID', right_on='Sub_ID')
        mergedTables = mergedTables.drop(columns='Sub_ID')

        p = Path(file_path)
        new_file_path = p.with_name(f'{signal_type}_Black_Merged.xlsx')
        mergedTables.to_excel(new_file_path, index=False)
    else:
        mergedTables = df.copy()
        print('no Demog')

    mergedTables['Val_clean'] = (
        mergedTables
        .groupby(['Imminence', 'Magnitude'])['Val']
        .transform(winsorize_group)
    )

    changed_win_only = (mergedTables['Val_clean'] != mergedTables['Val']).sum()
    changed_prec_win = round((changed_win_only / len(mergedTables)) * 100, 2)
    print(f"{signal_type}: {changed_win_only} rows changed with Z-Score based Winsorizing ({changed_prec_win}%)")

    # ------------- Define tables for isolated analyses -------------
    def morey_corrected_se(df, var, levels, bins):
        df['norm_val'] = (
                df[var]
                - df.groupby('ID')[var].transform('mean')
                + df[var].mean()
        )
        group_stats = df.groupby([levels, bins]).agg(
            mean_val=('norm_val', 'mean'),
            SD=('norm_val', 'std')
        ).reset_index()

        n_subjects = df['ID'].nunique()
        n_levels = df[levels].nunique()
        n_bins = df[bins].nunique()
        n_conditions = n_bins * n_levels

        morey_factor = np.sqrt(n_conditions / (n_conditions - 1))
        group_stats['SE_morey'] = (group_stats['SD'] / np.sqrt(n_subjects)) * morey_factor

        return group_stats

    df_cont = mergedTables.copy()
    group_stats = morey_corrected_se(df_cont, 'Val_clean', 'Magnitude', 'Imminence')

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

    # =========================================================
    # HEAT PHASE: DIFFERENCE SCORES & OUTLIER-CLEANED CI
    # =========================================================
    print("\n=== HEAT DIFFERENCES & CONFIDENCE INTERVALS (3SD CLEANED) ===")

    df_heat_subj = df_heat.groupby(['ID', 'Magnitude'], as_index=False, observed=True)['Val_clean'].mean()
    df_heat_pivot = df_heat_subj.pivot(index='ID', columns='Magnitude', values='Val_clean').reset_index()

    df_heat_pivot['Diff_Low'] = df_heat_pivot['Low'] - df_heat_pivot['Low-Black']
    df_heat_pivot['Diff_High'] = df_heat_pivot['High'] - df_heat_pivot['High-Black']

    # Loop ONLY prints summary diagnostics
    for col_name in ['Diff_Low', 'Diff_High']:
        raw_series = df_heat_pivot[col_name].dropna()

        cleaned_series = winsorize_group(raw_series, 3)
        excluded_count = raw_series != cleaned_series

        n_clean = len(cleaned_series)
        mean_clean = cleaned_series.mean()
        sem_clean = cleaned_series.sem()

        if n_clean > 1:
            ci_lower, ci_upper = stats.t.interval(0.95, df=n_clean - 1, loc=mean_clean, scale=sem_clean)
        else:
            ci_lower, ci_upper = np.nan, np.nan

        print(f"\n--- {col_name} ---")
        print(f"  Total Subjects: {len(raw_series)} (Winsorized {sum(excluded_count)} outlier(s))")
        print(f"  Cleaned Mean:   {mean_clean:.4f}")
        print(f"  95% CI:         [{ci_lower:.4f}, {ci_upper:.4f}]")

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

    # =========================================================
    # STATISTICAL SUMMARIES & SINGLE-PASS OUTLIER FILTERING
    # =========================================================

    cleaned_time_diffs = []
    diff_columns = ['Diff_4_minus_1', 'Diff_5_minus_4', 'Diff_6_minus_4', 'Diff_6_minus_5']

    contrast_label_map = {
        'Diff_4_minus_1': 'Bin 4 - Bin 1',
        'Diff_5_minus_4': 'Heat14 - Bin 4',
        'Diff_6_minus_4': 'Heat48 - Bin 4',
        'Diff_6_minus_5': 'Heat48 - Heat14'
    }

    # Dynamic mapping to trace original component pillars
    component_map = {
        'Diff_4_minus_1': ('Bin4', 'Bin1'),
        'Diff_5_minus_4': ('Heat14', 'Bin4'),
        'Diff_6_minus_4': ('Heat48', 'Bin4'),
        'Diff_6_minus_5': ('Heat48', 'Heat14')
    }

    for t_col in diff_columns:
        print(f"\n--- METRIC: {t_col} ---")
        comp1, comp2 = component_map[t_col]

        for mag in ORDER:
            # 1. Get the raw DataFrame subset including original components
            subset_raw = df_time_pivot[df_time_pivot['Magnitude'] == mag][[t_col, 'ID', comp1, comp2]].dropna()

            if subset_raw.empty:
                continue

            # 2. Extract the actual series for calculation
            raw_series = subset_raw[t_col]

            # 3. Clean the series
            cleaned_values = winsorize_group(raw_series, 3)
            excluded_count = (raw_series != cleaned_values).sum()

            # 4. Calculate Stats
            n_clean = len(cleaned_values)
            mean_clean = cleaned_values.mean()
            sem_clean = cleaned_values.sem()

            if n_clean > 1:
                ci_lower, ci_upper = stats.t.interval(0.95, df=n_clean - 1, loc=mean_clean, scale=sem_clean)
            else:
                ci_lower, ci_upper = np.nan, np.nan

            print(f"  Total Subjects: {len(raw_series)} (Winsorized {excluded_count} outlier(s))")
            print(f"  Cleaned Mean:   {mean_clean:.4f}")
            print(f"  95% CI:         [{ci_lower:.4f}, {ci_upper:.4f}]")

            # 5. Prepare data for saving by bringing it back into a DataFrame
            df_clean = subset_raw.copy()
            df_clean[t_col] = cleaned_values  # Replace raw values with winsorized values
            df_clean['Magnitude'] = mag
            df_clean['Contrast'] = contrast_label_map[t_col]
            df_clean = df_clean.rename(columns={t_col: 'Difference', comp1: 'Comp1_Val', comp2: 'Comp2_Val'})
            df_clean['Comp1_Name'] = comp1
            df_clean['Comp2_Name'] = comp2

            # 6. Append
            cleaned_time_diffs.append(df_clean[['ID', 'Magnitude', 'Contrast', 'Difference', 'Comp1_Val', 'Comp2_Val',
                                                'Comp1_Name', 'Comp2_Name']])

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

        df_subj = df_with_slopes.drop_duplicates(subset='ID').copy()
        df_subj['slope_clean'] = winsorize_group(df_subj['Imminence_slope'])

        r, p = pearsonr(df_subj['slope_clean'], df_subj['GAD7_total'])
        print(f'Pearson with GAD7: r = {r}, p = {p}')

        r, p = pearsonr(df_subj['slope_clean'], df_subj['STAI_total'])
        print(f'Pearson with STAI: r = {r}, p = {p}')

        r, p = pearsonr(df_subj['slope_clean'], df_subj['IUS_total'])
        print(f'Pearson with IUS Total: r = {r}, p = {p}')

        r, p = pearsonr(df_subj['slope_clean'], df_subj['PCL5_total'])
        print(f'Pearson with PCL5: r = {r}, p = {p}')

        r, p = pearsonr(df_subj['slope_clean'], df_subj['PHQ9_total'])
        print(f'Pearson with PHQ9 Total: r = {r}, p = {p}')

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

        # Initialize an empty list to store validation results
        ttest_results = []

        for contrast in contrast_label_map.values():
            # Subset data for the current contrast
            df_contrast = df_lme_diff[df_lme_diff['Contrast'] == contrast]

            # DYNAMIC CONFIGURATION BASED ON CONTRAST TYPE
            if contrast == 'Bin 4 - Bin 1':
                mag_col = 'Magnitude_H_L'
                unique_vals = df_contrast[mag_col].dropna().unique()
                if 1 in unique_vals or 2 in unique_vals:
                    magnitude_pairs_to_run = [(1, 2)]
                else:
                    magnitude_pairs_to_run = [('Low', 'High')]
            else:
                mag_col = 'Magnitude'
                magnitude_pairs_to_run = [('Low', 'Low-Black'), ('High', 'High-Black')]

            for mag1, mag2 in magnitude_pairs_to_run:
                # Filter using the dynamically assigned target column (mag_col)
                df_pair = df_contrast[df_contrast[mag_col].isin([mag1, mag2])]

                # 1. Pivot Table for basic Differences
                df_pivot = df_pair.pivot_table(
                    index='ID',
                    columns=mag_col,
                    values='Difference',
                    aggfunc='mean',
                    observed=False
                ).dropna()

                # Clean labels for the final output table if numeric
                mag1_label = 'Low' if mag1 == 1 else mag1
                mag2_label = 'High' if mag2 == 2 else ('High-Black' if mag2 == 'High-Black' else mag2)
                comparison_label = f"{mag1_label} vs {mag2_label}"

                if df_pivot.empty or len(df_pivot) < 2:
                    ttest_results.append({
                        'Contrast': contrast, 'Comparison': comparison_label, 'N_Pairs': len(df_pivot),
                        'Mean_1_Diff': np.nan, 'Mean_2_Diff': np.nan, 'T_Stat_Diff': np.nan, 'P_Val_Diff': np.nan,
                        'Comp1_Name': '-', 'T_Stat_Comp1': np.nan, 'P_Val_Comp1': np.nan,
                        'Comp2_Name': '-', 'T_Stat_Comp2': np.nan, 'P_Val_Comp2': np.nan
                    })
                    continue

                # Perform paired t-test on Difference scores
                t_stat_diff, p_val_diff = stats.ttest_rel(df_pivot[mag1], df_pivot[mag2])
                mean_1_diff = df_pivot[mag1].mean()
                mean_2_diff = df_pivot[mag2].mean()

                # 2. Pivot Table and T-Test on Component 1
                comp1_name = df_pair['Comp1_Name'].iloc[0] if 'Comp1_Name' in df_pair.columns else 'Comp1'
                df_pivot_c1 = df_pair.pivot_table(index='ID', columns=mag_col, values='Comp1_Val',
                                                  aggfunc='mean',observed=False).dropna()
                common_ids_c1 = df_pivot.index.intersection(df_pivot_c1.index)
                if len(common_ids_c1) >= 2:
                    t_stat_c1, p_val_c1 = stats.ttest_rel(df_pivot_c1.loc[common_ids_c1, mag1],
                                                          df_pivot_c1.loc[common_ids_c1, mag2])
                else:
                    t_stat_c1, p_val_c1 = np.nan, np.nan

                # 3. Pivot Table and T-Test on Component 2
                comp2_name = df_pair['Comp2_Name'].iloc[0] if 'Comp2_Name' in df_pair.columns else 'Comp2'
                df_pivot_c2 = df_pair.pivot_table(index='ID', columns=mag_col, values='Comp2_Val',
                                                  aggfunc='mean',observed=False).dropna()
                common_ids_c2 = df_pivot.index.intersection(df_pivot_c2.index)
                if len(common_ids_c2) >= 2:
                    t_stat_c2, p_val_c2 = stats.ttest_rel(df_pivot_c2.loc[common_ids_c2, mag1],
                                                          df_pivot_c2.loc[common_ids_c2, mag2])
                else:
                    t_stat_c2, p_val_c2 = np.nan, np.nan

                # Append to summary metrics list
                ttest_results.append({
                    'Contrast': contrast,
                    'Comparison': comparison_label,
                    'N_Pairs': len(df_pivot),
                    'Mean_1_Diff': mean_1_diff,
                    'Mean_2_Diff': mean_2_diff,
                    'T_Stat_Diff': t_stat_diff,
                    'P_Val_Diff': p_val_diff,
                    'Comp1_Name': comp1_name,
                    'T_Stat_Comp1': t_stat_c1,
                    'P_Val_Comp1': p_val_c1,
                    'Comp2_Name': comp2_name,
                    'T_Stat_Comp2': t_stat_c2,
                    'P_Val_Comp2': p_val_c2
                })

        # Create the expanded summary DataFrame
        df_ttest_summary = pd.DataFrame(ttest_results)

        # Print the comprehensive formatted table
        print("\n" + "=" * 125)
        print("                                     PAIRED T-TESTS COMPREHENSIVE SUMMARY TABLE (WITH COMPONENTS)")
        print("=" * 125)
        print(df_ttest_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("=" * 125)

        # --------------- corrs with questionnaires --------------
        anx_cols = ['GAD7_total','STAI_total','PHQ9_total','PCL5_total','IUS_total']

        if 'Gender_numeric' not in df_lme_diff.columns:
            df_lme_diff['Gender_numeric'] = df_lme_diff['Gender'].astype('category').cat.codes

        covariates = ['Age', 'Gender_numeric']
        correlation_results = []

        # =========================================================
        # MAIN CORRELATION LOOP
        # =========================================================
        for contrast in contrast_label_map.values():
            df_contrast = df_lme_diff[df_lme_diff['Contrast'] == contrast]

            for anx in anx_cols:
                if contrast == 'Bin 4 - Bin 1':
                    formula = f'Difference ~ {anx} + Magnitude_H_L + Age + Gender'
                    mag_col = 'Magnitude_H_L'
                    current_mags = [1, 2]
                else:
                    formula = f'Difference ~ {anx} + Magnitude + Age + Gender'
                    mag_col = 'Magnitude'
                    current_mags = ['Low', 'Low-Black', 'High', 'High-Black']

                # Run the mixed model
                res, df_with_slopes = run_lme(df_contrast, fixed_formula=formula, group_var="ID", method='powell')

                # Extract post-hoc PARTIAL correlations
                for mag in current_mags:
                    df_mag = df_contrast[df_contrast[mag_col] == mag].copy()

                    # Track components safely inside check lists
                    vars_to_check = ['Difference', 'Comp1_Val', 'Comp2_Val', anx] + covariates
                    df_mag_clean = df_mag.dropna(subset=vars_to_check)

                    # Collapse duplicates per subject
                    df_mag_clean = df_mag_clean.groupby('ID')[vars_to_check].mean().reset_index()

                    if len(df_mag_clean) < 5:
                        correlation_results.append({
                            'Contrast': contrast,
                            'Anxiety': anx,
                            'Magnitude': 'Low' if mag == 1 else ('High' if mag == 2 else mag),
                            'N': len(df_mag_clean),
                            'Partial_r': np.nan,
                            'p_value': np.nan,
                            'Significant': 'No Data'
                        })
                        continue

                    # Calculate Partial Correlation using Pingouin
                    stats_res = pg.partial_corr(
                        data=df_mag_clean,
                        x='Difference',
                        y=anx,
                        covar=covariates
                    )

                    p_col = [col for col in stats_res.columns if col in ['p-val', 'p-value', 'p_val', 'p']][0]

                    r = stats_res['r'].values[0]
                    p = stats_res[p_col].values[0]
                    is_sig = 'Yes (*)' if p < 0.05 else 'No'
                    mag_label = 'Low' if mag == 1 else ('High' if mag == 2 else mag)

                    correlation_results.append({
                        'Contrast': contrast,
                        'Anxiety': anx,
                        'Magnitude': mag_label,
                        'N': len(df_mag_clean),
                        'Partial_r': r,
                        'p_value': p,
                        'Significant': is_sig
                    })

                    # -------------------------------------------------------------------------
                    # FOLLOW-UP OLS DECOMPOSITION ANALYSIS (Only executed if significant)
                    # -------------------------------------------------------------------------
                    if p < 0.15:
                        comp1_name = df_mag['Comp1_Name'].iloc[0] if 'Comp1_Name' in df_mag.columns else 'Comp1'
                        comp2_name = df_mag['Comp2_Name'].iloc[0] if 'Comp2_Name' in df_mag.columns else 'Comp2'

                        print(
                            f"\n>>> [FOLLOW-UP OLS] Significant Partial Corr found for {contrast} ({mag_label}) with {anx} (p = {p:.4f})")

                        # OLS Evaluation for Component 1
                        formula_c1 = f"Comp1_Val ~ {anx} + Age + Gender_numeric"
                        ols_res_c1 = smf.ols(formula_c1, data=df_mag_clean).fit()
                        print(
                            f"    -> Component 1 ({comp1_name}) vs {anx}: Beta = {ols_res_c1.params[anx]:.4f}, p = {ols_res_c1.pvalues[anx]:.4f}")

                        # OLS Evaluation for Component 2
                        formula_c2 = f"Comp2_Val ~ {anx} + Age + Gender_numeric"
                        ols_res_c2 = smf.ols(formula_c2, data=df_mag_clean).fit()
                        print(
                            f"    -> Component 2 ({comp2_name}) vs {anx}: Beta = {ols_res_c2.params[anx]:.4f}, p = {ols_res_c2.pvalues[anx]:.4f}")

        # Summary Table for the Correlations
        df_corr_summary = pd.DataFrame(correlation_results)

        print("\n" + "=" * 95)
        print("             PARTIAL CORRELATIONS (Controlling for Age & Gender) SUMMARY TABLE")
        print("=" * 95)
        print(df_corr_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("=" * 95)


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