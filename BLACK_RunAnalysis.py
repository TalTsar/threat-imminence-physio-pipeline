import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import zscore, skew
import pingouin as pg
from pathlib import Path
from scipy.stats.mstats import winsorize
import glob
import os
from datetime import datetime

import config_TIMblack as config

imestamp = datetime.now().strftime('%d%m%Y')

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
    print(f"       STARTING ANALYSIS: {signal_type}")
    print(f"{'=' * 40}")

    # 1. Load Data
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    df = pd.read_excel(file_path)


    # how many subjects? how many blocks per subject? how many segments? how many trials overall
    valCountPerBlock = df.groupby(['ID','Block'])[['Val']].count()
    valMeansPerCondition = df.groupby(['Magnitude', 'Imminence'])['Val'].mean()
    valMeansPerConditionPerSubj = df.groupby(['Magnitude','Imminence','ID'])['Val'].mean()


    # 2. Filtering
    bad_subs = EXCLUSIONS.get(signal_type, [])
    global_bad_subs = EXCLUSIONS.get('GLOB', [])

    bad_df = df[df['ID'].isin(bad_subs)]
    bad_valMeansPerConditionPerSubj = bad_df.groupby(['Magnitude','Imminence','ID'])['Val'].mean()
    bad_valCountPerBlock = bad_df.groupby(['ID','Block'])[['Val']].count()

    df = df[~df['ID'].isin(bad_subs)]

    valCountPerBlock_postExclusion = df.groupby(['ID','Block'])[['Val']].count()



    # ----------- Define Exclusion Criteria -----------
    # remove NaNs
    df = df.dropna(subset=['Val'])
    print(f"Loaded {len(df)} rows. Subjects: {df['ID'].nunique()}")




    # ------- Z-score guided Winsorize (per group) -------
    def winsorize_group(x,thresh=3):
        x = x.astype(float)
        z = zscore(x)
        in_range = x[(z >= -thresh) & (z <= thresh)]

        if len(in_range) == 0:
            return x  # nothing to winsorize (fixation)

        min_in_range = in_range.min()
        max_in_range = in_range.max()

        x_wins = x.copy()
        x_wins[z < -thresh] = min_in_range
        x_wins[z > thresh] = max_in_range


        return x_wins

    # ------- General LME function -------
    def run_lme(df, fixed_formula, random_formula="~1", group_var="ID", method="nm", vc_formula=None, extractSlopes=True):
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

            sub_slopes = {
                sub: effects['Imminence_c']
                for sub, effects in res_mdl.random_effects.items()
            }

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

            return res_mdl, df_merged



        print(f"\n=== {fixed_formula} + ( {random_formula} | {group_var} ) ===")
        if vc_formula is not None:
            print(f"    + independent slopes via vc_formula: {list(vc_formula.keys())}")

        print(res_mdl.summary())

        return res_mdl


    # ----------- Stats with Demog -----------
    if DEMOG:
        demog = pd.read_excel(DEMOG)
        mergedTables = df.merge(demog, left_on='ID', right_on='Sub_ID')
        mergedTables = mergedTables.drop(columns='Sub_ID')

        p=Path(file_path)
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
    def morey_corrected_se(df,var,levels,bins):
        # Morey Corrected SE
        df['norm_val'] = (
                df[var]
                - df.groupby('ID')[var].transform('mean')
                + df[var].mean()
        )
        group_stats = df_cont.groupby([levels, bins]).agg(
            mean_val=('norm_val', 'mean'),
            SD=('norm_val', 'std')
        ).reset_index()

        n_subjects = df_cont['ID'].nunique()
        n_levels = df_cont[levels].nunique()
        n_bins = df_cont[bins].nunique()
        n_conditions = n_bins * n_levels

        morey_factor = np.sqrt(n_conditions / (n_conditions - 1))
        group_stats['SE_morey'] = (group_stats['SD'] / np.sqrt(n_subjects)) * morey_factor

        return group_stats

    df_cont = mergedTables.copy()
    group_stats = morey_corrected_se(df_cont, 'Val_clean', 'Magnitude', 'Imminence')

    df_cont_anticipation = mergedTables[mergedTables['Imminence'].isin([1, 2, 3, 4])].copy()

    df_cont_anticipation['Magnitude_H_L_B'] = df_cont_anticipation['Magnitude'].apply(
        lambda x: 'Black' if 'Black' in x
        else ('Low' if 'Low' in x else 'High')
    )

    df_cont_anticipation['Magnitude_H_L_B'] = pd.Categorical(
        df_cont_anticipation['Magnitude_H_L_B'],
        categories=['Low','High', 'Black'],
        ordered=False
    )

    mapping = {
        'Start':0,
        'Low': 1,
        'High': 2,
        'Black': 3
    }

    df_cont_anticipation['Magnitude_H_L_B'] = df_cont_anticipation['Magnitude_H_L_B'].map(mapping)
    # df_cont_anticipation['prev_level'] = df_cont_anticipation['prev_level'].map(mapping)

    df_heat = mergedTables[mergedTables['Imminence'] == 5].copy()
    df_heat['Magnitude'] = df_heat['Magnitude'].astype('category') # Ensure 'Magnitude' categorical


    # =========================================================
    # HEAT PHASE: DIFFERENCE SCORES & OUTLIER-CLEANED CI
    # =========================================================
    print("\n=== HEAT DIFFERENCES & CONFIDENCE INTERVALS (3SD CLEANED) ===")


    # 1. Get the mean Val_clean for each subject and condition
    df_heat_subj = df_heat.groupby(['ID', 'Magnitude'], as_index=False, observed=True)['Val_clean'].mean()

    # 2. Pivot the table so Magnitudes become columns
    df_heat_pivot = df_heat_subj.pivot(index='ID', columns='Magnitude', values='Val_clean').reset_index()

    # 3. Calculate raw differences (Normal - Black)
    df_heat_pivot['Diff_Low'] = df_heat_pivot['Low'] - df_heat_pivot['Low-Black']
    df_heat_pivot['Diff_High'] = df_heat_pivot['High'] - df_heat_pivot['High-Black']

    import scipy.stats as stats

    for col_name in ['Diff_Low', 'Diff_High']:
        # Grab raw difference values (dropping NaNs if a subject is missing data)
        raw_series = df_heat_pivot[col_name].dropna()

        # Calculate initial mean and standard deviation for outlier boundaries
        m_initial = raw_series.mean()
        sd_initial = raw_series.std()

        # Define 3SD threshold boundaries
        # Note: This is standard two-tailed filtering (+/- 3SD).
        # If you strictly want to exclude ONLY values above the upper limit,
        # change the mask below to: raw_series <= m_initial + (3 * sd_initial)
        lower_bound = m_initial - (3 * sd_initial)
        upper_bound = m_initial + (3 * sd_initial)

        # Filter out the outliers
        cleaned_series = raw_series[(raw_series >= lower_bound) & (raw_series <= upper_bound)]
        excluded_count = len(raw_series) - len(cleaned_series)

        # Calculate final descriptives and Standard Error of the Mean (SEM)
        n_clean = len(cleaned_series)
        mean_clean = cleaned_series.mean()
        sem_clean = cleaned_series.sem()

        # Compute 95% Confidence Interval using the t-distribution
        if n_clean > 1:
            ci_lower, ci_upper = stats.t.interval(0.95, df=n_clean - 1, loc=mean_clean, scale=sem_clean)
        else:
            ci_lower, ci_upper = np.nan, np.nan

        print(f"\n--- {col_name} ---")
        print(f"  Total Subjects: {len(raw_series)} (Excluded {excluded_count} outlier(s) outside 3SD)")
        print(f"  Cleaned Mean:   {mean_clean:.4f}")
        print(f"  95% CI:         [{ci_lower:.4f}, {ci_upper:.4f}]")


    # =========================================================
    # PLOT: DIFFERENCE SCORES (STRIP + POINTPLOT)
    # =========================================================
    import scipy.stats as stats

    # 1. Isolate and clean the difference data using the 3SD rule
    cleaned_diffs = []
    for col_name in ['Diff_Low', 'Diff_High']:
        raw_series = df_heat_pivot[[col_name, 'ID']].dropna()

        m_init = raw_series[col_name].mean()
        sd_init = raw_series[col_name].std()

        # Apply 3SD Filter
        cleaned_subset = raw_series[
            (raw_series[col_name] >= m_init - 3 * sd_init) &
            (raw_series[col_name] <= m_init + 3 * sd_init)
            ]

        # Format into a long-form structure for Seaborn
        for _, row in cleaned_subset.iterrows():
            cleaned_diffs.append({
                'ID': row['ID'],
                'Contrast': 'Low vs Low-Black' if col_name == 'Diff_Low' else 'High vs High-Black',
                'Difference': row[col_name]
            })

    df_plot_diff = pd.DataFrame(cleaned_diffs)

    # 2. Generate the plot using your exact formatting layout
    plt.figure(figsize=(6, 4))

    # Reference Line at 0 to easily visualize statistical significance
    plt.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7, zorder=0)

    # Custom palette dictionary matching your long-form labels
    DIFF_ORDER = ['Low vs Low-Black', 'High vs High-Black']
    DIFF_PALETTE = {
        'Low vs Low-Black': PALETTE.get('Low', '#1f77b4'),
        'High vs High-Black': PALETTE.get('High', '#ff7f0e')
    }

    # Point plot (Mean + 95% Bootstrapped CI)
    sns.pointplot(
        data=df_plot_diff,
        x='Contrast',
        y='Difference',
        hue='Contrast',
        legend=False,
        order=DIFF_ORDER,
        palette='dark:black',
        errorbar=('ci', 95),
        linestyle='none',
        capsize=0.1,
        zorder=4
    )

    # Strip plot (Individual Cleaned Subject Differences)
    sns.stripplot(
        data=df_plot_diff,
        x='Contrast',
        y='Difference',
        hue='Contrast',
        legend=False,
        order=DIFF_ORDER,
        palette=DIFF_PALETTE,
        jitter=True,
        alpha=0.6,
        size=6,
        zorder=3
    )

    # Formatting and layout titles
    plt.title(f'{signal_type}: Heat Difference Scores (± 95% CI)')
    plt.ylabel(f'Δ Mean {unit_label} (Regular - Black)')
    plt.xlabel('Condition Contrast')

    # Adjust axes constraints for clean visuals
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle=':', alpha=0.5, zorder=0)

    plt.tight_layout()
    plt.show(block=True)

    # =========================================================
    # TIME-BIN SHIFTS: ISOLATE AND PIVOT BINS (INCLUDING BIN 6)
    # =========================================================
    print("\n=== ALL TIME BIN DIFFERENCES & CONFIDENCE INTERVALS (3SD CLEANED) ===")

    # 1. Filter dataset down to the relevant time bins: 1, 4, 5, and 6
    df_time = mergedTables[mergedTables['Imminence'].isin([1, 4, 5, 6])].copy()

    # 2. Get the clean target mean for each subject, magnitude condition, and time bin
    df_time_subj = df_time.groupby(['ID', 'Magnitude', 'Imminence'], as_index=False, observed=True)['Val_clean'].mean()

    # 3. Pivot the table so Imminence Bins become distinct columns
    df_time_pivot = df_time_subj.pivot(
        index=['ID', 'Magnitude'],
        columns='Imminence',
        values='Val_clean'
    ).reset_index()

    # Rename columns for clear math operations
    df_time_pivot = df_time_pivot.rename(columns={1: 'Bin1', 4: 'Bin4', 5: 'Heat14', 6: 'Heat48'})

    # 4. Calculate all raw time-wise difference metrics per subject per condition
    df_time_pivot['Diff_4_minus_1'] = df_time_pivot['Bin4'] - df_time_pivot['Bin1']
    df_time_pivot['Diff_4_minus_5'] = df_time_pivot['Bin4'] - df_time_pivot['Heat14']
    df_time_pivot['Diff_4_minus_6'] = df_time_pivot['Bin4'] - df_time_pivot['Heat48']
    df_time_pivot['Diff_5_minus_6'] = df_time_pivot['Heat14'] - df_time_pivot['Heat48']

    # =========================================================
    # STATISTICAL SUMMARIES & OUTLIER FILTERING
    # =========================================================
    import scipy.stats as stats

    cleaned_time_diffs = []
    diff_columns = ['Diff_4_minus_1', 'Diff_4_minus_5', 'Diff_4_minus_6', 'Diff_5_minus_6']

    # Map your calculation columns to the EXACT names you want displayed on your graph panels
    contrast_label_map = {
        'Diff_4_minus_1': 'Bin 4 - Bin 1',
        'Diff_4_minus_5': 'Bin 4 - Heat14',
        'Diff_4_minus_6': 'Bin 4 - Heat48',
        'Diff_5_minus_6': 'Heat14 - Heat48'
    }

    for col_name in diff_columns:
        print(f"\n--- METRIC: {col_name} ---")

        # Process each condition independently for accurate standard deviation thresholds
        for mag in ORDER:
            subset_raw = df_time_pivot[df_time_pivot['Magnitude'] == mag][[col_name, 'ID']].dropna()

            if subset_raw.empty:
                continue

            m_initial = subset_raw[col_name].mean()
            sd_initial = subset_raw[col_name].std()

            # 3SD threshold boundaries
            lower_bound = m_initial - (3 * sd_initial)
            upper_bound = m_initial + (3 * sd_initial)

            # Filter out condition outliers
            cleaned_series = subset_raw[(subset_raw[col_name] >= lower_bound) & (subset_raw[col_name] <= upper_bound)]
            excluded_count = len(subset_raw) - len(cleaned_series)

            # Final descriptives and 95% Confidence Interval
            n_clean = len(cleaned_series)
            mean_clean = cleaned_series[col_name].mean()
            sem_clean = cleaned_series[col_name].sem()

            if n_clean > 1:
                ci_lower, ci_upper = stats.t.interval(0.95, df=n_clean - 1, loc=mean_clean, scale=sem_clean)
            else:
                ci_lower, ci_upper = np.nan, np.nan

            print(f"  [{mag}] Subjects: {len(subset_raw)} (Excluded {excluded_count} outside 3SD)")
            print(f"    Cleaned Mean: {mean_clean:.4f}")
            print(f"    95% CI:       [{ci_lower:.4f}, {ci_upper:.4f}]")

            # Format into long-form dataframe structure for seaborn plotting
            for _, row in cleaned_series.iterrows():
                cleaned_time_diffs.append({
                    'ID': row['ID'],
                    'Magnitude': mag,
                    'Contrast': contrast_label_map[col_name],  # Pulls the exact respectable string layout
                    'Difference': row[col_name]
                })

    df_plot_time = pd.DataFrame(cleaned_time_diffs)

    # =========================================================
    # PLOT: 1x4 TIME PANEL GRID COMPARED BY CONDITION
    # =========================================================
    PANEL_ORDER = ['Bin 4 - Bin 1', 'Bin 4 - Heat14', 'Bin 4 - Heat48', 'Heat14 - Heat48']

    g = sns.FacetGrid(
        df_plot_time,
        col='Contrast',
        col_order=PANEL_ORDER,
        height=4.5,
        aspect=1.0,
        sharey=False  # Free axes allow recovery scales to scale to their localized variance cleanly
    )

    def plot_strip_and_point(data, **kws):
        ax = plt.gca()

        # Horizontal reference line at zero
        ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7, zorder=0)

        # Strip plot for individual clean observations
        sns.stripplot(
            data=data,
            x='Magnitude',
            y='Difference',
            hue='Magnitude',
            order=ORDER,
            palette=PALETTE,
            jitter=True,
            alpha=0.5,
            size=4.5,
            legend=False,
            ax=ax,
            zorder=2
        )

        # Overlaid summary Point plot (Mean + Bootstrapped 95% Confidence Intervals)
        sns.pointplot(
            data=data,
            x='Magnitude',
            y='Difference',
            hue='Magnitude',
            order=ORDER,
            palette='dark:black',
            errorbar=('ci', 95),
            linestyle='none',
            capsize=0.15,
            legend=False,
            ax=ax,
            zorder=3
        )

        # Active panel aesthetics formatting
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.yaxis.grid(True, linestyle=':', alpha=0.5, zorder=0)

    # Map the drawing logic onto the grid layout panels
    g.map_dataframe(plot_strip_and_point)

    # Final formatting updates
    g.set_titles("{col_name}")
    g.set_axis_labels("Condition", f"Δ Mean {unit_label}")
    g.fig.suptitle(f'{signal_type}: Comprehensive Time-Bin Transitions (± 95% CI)', y=1.06, fontsize=13)

    plt.tight_layout()
    plt.show(block=True)
    # =========================================================
    # STAT ANALYSIS: Anticipation
    # =========================================================
    target_variables = ['Val_clean']

    for target in target_variables:
        print(f"\n" + "=" * 60)
        print(f" STARTING ANALYSIS FOR TARGET: {target.upper()}")
        print("=" * 60)

        # 1. Explicitly drop NaNs for this specific target
        df_clean = df_cont_anticipation.dropna(subset=[target]).copy()

        # 2. Format your grouping and categorical variables
        df_clean['ID'] = df_clean['ID'].astype('string')
        df_clean['Gender'] = df_clean['Gender'].astype('category')

        # 3. Mean-center your continuous predictors
        # df_clean['Magnitude_c'] = df_clean['Magnitude'] - df_clean['Magnitude'].mean()
        df_clean['Imminence_c'] = df_clean['Imminence'] - df_clean['Imminence'].mean()
        # df_clean['Imminence_c2'] = df_clean['Imminence_c'] ** 2

        # 4. Run the model
        formula = f'{target} ~ Magnitude_H_L_B * Imminence_c + Block + Age + Gender'

        try:
            res,df_with_slopes = run_lme(
                df_clean,
                fixed_formula=formula,
                random_formula="~Imminence_c",  # intercept only here
                group_var="ID",
                method="lbfgs"
            )
        except Exception as e:
            print(f"Model failed to converge or errored out for {target}: {e}")



        # -------- Heat Model --------

        formula = f'{target} ~ Magnitude + Block + Age + Gender'

        res = run_lme(
            df_heat,
            fixed_formula=formula,
            random_formula="~1",  # intercept only here
            group_var="ID",
            method="lbfgs",
            extractSlopes = False
        )

        # -------- some anxiety models --------
        try:
            from scipy.stats import pearsonr
            df_subj = df_with_slopes.drop_duplicates(subset='ID').copy()

            df_subj['slope_clean'] = winsorize_group(df_subj['Imminence_slope'])

            # ----------- GAD7 -----------
            r, p = pearsonr(
                df_subj['slope_clean'],
                df_subj['GAD7_Total']
            )


            print(f'Pearson with GAD7: r = {r}, p = {p}')


            # ----------- STAI -----------
            r, p = pearsonr(
                df_subj['slope_clean'],
                df_subj['STAIT_Total']
            )

            print(f'Pearson with STAI: r = {r}, p = {p}')


            # ----------- IUS -----------
            r, p = pearsonr(
                df_subj['slope_clean'],
                df_subj['IUS_Total']
            )

            print(f'Pearson with IUS Total: r = {r}, p = {p}')

        except Exception as e:
            print(f"Model failed to converge or errored out for {target}: {e}")


    # =========================================================
    # PLOT : Imminence Random Slopes & Anxiety Scores
    # =========================================================
        # =========================================================
        # PLOT : Imminence Random Slopes & Anxiety Scores (Publication Quality)
        # =========================================================
        from scipy.stats import pearsonr
        import matplotlib.transforms as transforms

        # Define the pairs we want to plot along with their display labels
        anxiety_vars = [
            ('GAD7_Total', 'GAD-7 Total Score'),
            ('STAIT_Total', 'STAI-T Total Score'),
            ('IUS_Total', 'IUS Total Score')
        ]

        for y_var, y_label in anxiety_vars:
            # Drop any missing rows between the two specific variables being plotted
            plot_df = df_subj[['slope_clean', y_var]].dropna()

            N = len(plot_df)
            if N < 3:
                print(f"Skipping plot for {y_var}: insufficient data points (N={N}).")
                continue

            # Compute the exact Pearson correlation stats
            r_val, p_val = pearsonr(plot_df['slope_clean'], plot_df[y_var])

            # Initialize the figure matching your theme proportions
            plt.figure(figsize=(5.5, 4.5))

            # Plot the individual points and the linear regression line with its 95% CI band
            sns.regplot(
                data=plot_df,
                x='slope_clean',
                y=y_var,
                scatter_kws={'alpha': 0.6, 's': 40, 'color': '#1f77b4'},  # Styled translucent points
                line_kws={'color': 'black', 'linewidth': 1.5},  # Clean, crisp regression line
                ci=95  # 95% Confidence Interval band
            )

            # Format statistical output text string
            # Formats p-values using standard scientific thresholding (e.g., p < .001)
            p_string = f"p < .001" if p_val < 0.001 else f"p = {p_val:.3f}"
            stats_text = f"N = {N}\nr = {r_val:.2f}\n{p_string}"

            # Insert a clean text box in the upper-right corner of the plot coordinates
            props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gainsboro', alpha=0.9)
            plt.gca().text(
                0.95, 0.95, stats_text,
                transform=plt.gca().transAxes,
                fontsize=10,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=props,
                fontfamily='monospace'  # Keeps statistics clean and vertically aligned
            )

            # Final layout aesthetics formatting
            # plt.title(f'{signal_type}: Random Slope vs. {y_label}', fontsize=11, pad=12, fontweight='bold')
            plt.xlabel(f'Random Slope: Imminence', fontsize=10, labelpad=8)
            plt.ylabel(y_label, fontsize=10, labelpad=8)

            # Clean axis spines
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, linestyle=':', alpha=0.4)

            plt.tight_layout()
            plt.show(block=True)


    # =========================================================
    # PLOT : Time Course
    # =========================================================
    if BLACK:

        # =========================================================
        # CREATE CONDITION LABELS
        # =========================================================

        df_cont['Condition_Type'] = df_cont['Magnitude'].apply(
            lambda x: 'Black' if 'Black' in x else 'Normal'
        )

        group_stats['Condition_Type'] = group_stats['Magnitude'].apply(
            lambda x: 'Black' if 'Black' in x else 'Normal'
        )

        # =========================================================
        # CREATE GAPLESS X MAPPING
        # =========================================================
        xticks_vals = sorted(group_stats['Imminence'].unique())
        x_pos_map = {val: i for i, val in enumerate(xticks_vals)}

        # =========================================================
        # PLOTTING FUNCTION (USES GROUPED DATA ONLY)
        # =========================================================
        def plot_lines(data, **kws):
            ax = plt.gca()

            for level in ORDER:
                subset = data[data['Magnitude'] == level]
                if subset.empty:
                    continue

                x = subset['Imminence'].map(x_pos_map)

                ax.errorbar(
                    x=x,
                    y=subset['mean_val'],
                    yerr=subset['SE_morey'],
                    label=level,
                    marker='o',
                    linestyle='-',
                    markersize=7,
                    markeredgecolor='white',
                    markeredgewidth=0.8,
                    linewidth=1.5,
                    color=PALETTE[level],
                    zorder=3
                )

            # =========================================================
            # BACKGROUND: ANTICIPATION WINDOW
            # =========================================================
            ax.axvspan(
                x_pos_map.get(0.9, 0),
                x_pos_map.get(4.1, 4),
                color='#F4F4F4',
                zorder=0
            )

            # =========================================================
            # SECTION LABELS
            # =========================================================
            trans = ax.get_xaxis_transform()
            ax.text(-0.3, 0.98, "Baseline", transform=trans,
                    ha='left', va='top', fontsize=10, color='#666666')
            ax.text(1.3, 0.98, "Anticipation", transform=trans,
                    ha='left', va='top', fontsize=10, color='#666666')
            ax.text(4.8, 0.98, "Heat & Recovery", transform=trans,
                    ha='left', va='top', fontsize=10, color='#666666')

            # =========================================================
            # AXES FORMATTING
            # =========================================================
            ax.set_xticks(list(x_pos_map.values()))
            ax.set_xticklabels(
                [CUSTOM_X_LABELS.get(k, k) for k in xticks_vals],
                rotation=45
            )

            ax.yaxis.grid(True, linestyle='--', alpha=0.2)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # =========================================================
        # 1x2 FACET GRID: NORMAL vs BLACK
        # =========================================================
        g = sns.FacetGrid(
            group_stats,
            col='Condition_Type',
            col_order=['Normal', 'Black'],
            height=4,
            aspect=1.3,
            sharey=True,
            sharex=True
        )

        g.map_dataframe(plot_lines)

        # =========================================================
        # LABELS / TITLES
        # =========================================================
        g.set_titles("{col_name}")
        g.set_axis_labels("Imminence", f"Mean {unit_label}")

        g.fig.suptitle(f'{signal_type}: Time Course (Normal vs Black)', y=1.05)

        plt.tight_layout()
        plt.show(block=False)


    # =========================================================
    # PLOT: Heat Phase Only (Stripplot)
    # =========================================================


    print("=== DESCRIPTIVE STATISTICS ===")

    df_heat_agg = df_heat.groupby(
        ['Magnitude'],
        as_index=False,
        observed=True
    ).agg(
        mean=('Val_clean', 'mean'),
        sd=('Val_clean', 'std'),
        n=('Val_clean', 'count')
    )

    print(df_heat_agg)
    print("\n")

    # ---------------------------------------------------------
    # Step 2: One-Way Repeated-Measures ANOVA
    # ---------------------------------------------------------
    print("=== ANOVA RESULTS (heat differs across Magitudes) ===")
    anova_results = pg.rm_anova(
        data=df_heat,
        dv='Val_clean',  # Dependent Variable
        within='Magnitude',  # Independent Variable (Within-subjects factor)
        subject='ID',  # Updated Participant identifier
        detailed=True,
        effsize='ng2'  # Calculates generalized eta squared (η²g)
    )
    print(anova_results)
    print("\n")

    # Extract row for your within-subject factor
    row = anova_results.loc[anova_results['Source'] == 'Magnitude'].iloc[0]

    # Uncorrected dfs
    df1_unc = row['DF']  # numerator df (e.g., 2)
    df2_unc = anova_results.loc[anova_results['Source'] == 'Error', 'DF'].iloc[0]  # denominator df (e.g., 78)

    # Greenhouse–Geisser epsilon
    eps = row['eps']

    # Corrected dfs
    df1_gg = df1_unc * eps
    df2_gg = df2_unc * eps

    print(f"GG-corrected dfs: ({df1_gg:.2f}, {df2_gg:.2f})")

    # ---------------------------------------------------------
    # Step 3: Bonferroni-Corrected Post-Hoc Tests
    # ---------------------------------------------------------
    print("=== POST-HOC COMPARISONS (BONFERRONI) ===")
    post_hocs = pg.pairwise_tests(
        data=df_heat,
        dv='Val_clean',
        within='Magnitude',
        subject='ID',  # Updated Participant identifier
        padjust='bonf'  # Applies Bonferroni correction
    )

    # Dynamically checks which columns exist in your specific version of Pingouin
    desired_columns = ['A', 'B', 'T','dof', 'p_unc', 'p_corr', 'p_adjust']
    columns_to_print = [col for col in desired_columns if col in post_hocs.columns]

    print(post_hocs[columns_to_print])


    # Aggregate by subject first (one point per subject per condition)
    df_heat_agg = df_heat.groupby(['Magnitude', 'ID'], as_index=False,observed=False)[['Val_clean']].mean()

    plt.figure(figsize=(6, 4))

    # Point plot (Mean + CI)
    sns.pointplot(
        data=df_heat_agg, x='Magnitude', y='Val_clean',hue='Magnitude',legend=False,
        order=ORDER,
        palette='dark:black', errorbar=('ci', 95), linestyle='none', capsize=0.1
    )

    # Strip plot
    sns.stripplot(
        data=df_heat_agg, x='Magnitude', y='Val_clean',hue='Magnitude',legend=False,
        order=ORDER,
        palette=PALETTE, jitter=True, alpha=0.6, size=6
    )



    plt.title(f'{signal_type}: Avg Heat Response (± 95% CI)')
    plt.ylabel(f'Mean {unit_label}')
    plt.xlabel('Condition')
    plt.tight_layout()
    plt.show(block=True)

    # # =========================================================
    # # PLOT: TIME COURSE WITH ANXIETY (GAD-7)
    # # =========================================================
    #
    # # print(df_cont_anticipation['GAD7_total'].describe())
    #
    # gad_med = df_cont_anticipation['GAD7_total'].median()
    #
    # df_cont_anticipation['GAD7_cat'] = np.where(
    #     df_cont_anticipation['GAD7_total'] <= gad_med,
    #     'Low',
    #     'High'
    # )
    #
    # # print(df_cont_anticipation.groupby(['ID','GAD7_cat'])[['GAD7_total']].mean())
    #
    # levels = ORDER
    # fig, axes = plt.subplots(1, len(levels), figsize=(4 * len(levels), 4), sharey=True)
    #
    # for ax, level in zip(axes, levels):
    #
    #     df_level = df_cont_anticipation[df_cont_anticipation['Magnitude'] == level]
    #     group_level = group_stats[group_stats['Magnitude'] == level]
    #
    #     for cat in cats:
    #         subset = group_level[group_level['GAD7_cat'] == cat]
    #
    #         # Use the new gad_dodge dictionary here
    #         shifted_x = subset['Imminence'] #+ gad_dodge[cat]
    #
    #         ax.errorbar(
    #             x=shifted_x,
    #             y=subset['mean_val'],
    #             yerr=subset['SE_morey'],
    #             label=cat,
    #             marker=markers[cat],
    #             linestyle=line_styles[cat],
    #             markersize=8,
    #             markeredgecolor='white',
    #             markeredgewidth=.8,
    #             linewidth=1,
    #             elinewidth=1.5,
    #             capsize=0,
    #             color=PALETTE[level],  # color fixed by heat level
    #             zorder=3
    #         )
    #
    #
    #     # axes
    #     xticks_vals = sorted(df_level['Imminence'].unique())
    #     xticks_labels = [CUSTOM_X_LABELS.get(k, k) for k in xticks_vals]
    #
    #     ax.set_xticks(xticks_vals)
    #     ax.set_xticklabels(xticks_labels, rotation=45)
    #     ax.set_xlabel("Imminence", fontsize=11, labelpad=10)
    #     ax.yaxis.grid(True, linestyle='--', color='gray', alpha=0.2, zorder=0)
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #     ax.set_title(f"{level}")
    #
    # axes[0].set_ylabel(f"Mean {unit_label}", fontsize=11, labelpad=10)
    # axes[-1].legend(frameon=False, loc='best', fontsize=10,title='GAD-7 Category')
    #
    # plt.tight_layout()
    # plt.show(block=True)




# =========================================================
# MAIN EXECUTION LOOP
# =========================================================
if __name__ == "__main__":

    # Get Data from Config
    DATA_FOLDER = config.output_folder

    BLACK = config.black
    PALETTE = config.palette_dict
    ORDER = config.order
    EXCLUSIONS = config.exclusions
    CUSTOM_X_LABELS = config.custom_x_labels
    DEMOG = config.demographic_df

    # Configuration for each signal
    # (Signal Name, Unit Label)
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