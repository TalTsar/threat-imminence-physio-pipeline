import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

import config


def run_lme(df, fixed_formula, random_formula="~1", group_var="ID", method="nm", vc_formula=None,
            extractSlopes=True):
    df = df.copy()

    mdl = smf.mixedlm(  fixed_formula, data=df, groups=df[group_var],
        re_formula=random_formula, vc_formula=vc_formula)

    res_mdl = mdl.fit(method=method)

    if extractSlopes:
        sub_slopes = pd.DataFrame.from_dict( res_mdl.random_effects, orient='index')
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