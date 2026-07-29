
import pandas as pd
from scipy.stats import zscore, skew
from pathlib import Path



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


def merge_tables(df,file_path,signal_type,DEMOG):
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


    return mergedTables