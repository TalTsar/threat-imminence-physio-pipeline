from scipy import stats
import numpy as np
import pandas as pd
import config


def run_contrast_ttest(df_lme_diff):
    # Initialize an empty list to store validation results
    ttest_results = []
    component_map = config.component_map
    contrast_label_map = config.contrast_label_map

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
                                              aggfunc='mean' ,observed=False).dropna()
            common_ids_c1 = df_pivot.index.intersection(df_pivot_c1.index)
            if len(common_ids_c1) >= 2:
                t_stat_c1, p_val_c1 = stats.ttest_rel(df_pivot_c1.loc[common_ids_c1, mag1],
                                                      df_pivot_c1.loc[common_ids_c1, mag2])
            else:
                t_stat_c1, p_val_c1 = np.nan, np.nan

            # 3. Pivot Table and T-Test on Component 2
            comp2_name = df_pair['Comp2_Name'].iloc[0] if 'Comp2_Name' in df_pair.columns else 'Comp2'
            df_pivot_c2 = df_pair.pivot_table(index='ID', columns=mag_col, values='Comp2_Val',
                                              aggfunc='mean' ,observed=False).dropna()
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


    return df_ttest_summary