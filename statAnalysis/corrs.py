import numpy as np
import pandas as pd
import pingouin as pg
import statsmodels.formula.api as smf

from statAnalysis.modifyTable import winsorize_group,merge_tables
from scipy.stats import pearsonr
import config
from statAnalysis.lme import run_lme

def corr_randSlope_Anx(df_with_slopes):
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


def corr_sub_Anx(df_lme_diff,contrast_label_map):

    # --------------- corrs with questionnaires --------------
    anx_cols = ['GAD7_total', 'STAI_total', 'PHQ9_total', 'PCL5_total', 'IUS_total']

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

    return df_corr_summary