import numpy as np
from scipy import stats
import config as config
from config import component_map, contrast_label_map
from statAnalysis.modifyTable import winsorize_group



DATA_FOLDER = config.output_folder
BLACK = config.black
PALETTE = config.palette_dict
ORDER = config.order
EXCLUSIONS = config.exclusions
CUSTOM_X_LABELS = config.custom_x_labels
DEMOG = config.demographic_df



def heat_diff(df_heat):
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
# STATISTICAL SUMMARIES & SINGLE-PASS OUTLIER FILTERING
# =========================================================
def sub_describe(df_time_pivot):
    cleaned_time_diffs = []

    diff_columns = ['Diff_4_minus_1', 'Diff_5_minus_4', 'Diff_6_minus_4', 'Diff_6_minus_5']

    component_map = config.component_map
    contrast_label_map = config.contrast_label_map

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

    return cleaned_time_diffs