# util.py
import glob
import os
from datetime import datetime
import config as config
EXCLUSIONS = config.exclusions


def get_latest_file(signal_type, folder_path):
    """Finds the most recent Excel file for a given signal type."""
    search_pattern = os.path.join(folder_path, f"TIM_{signal_type}_*.xlsx")
    files = glob.glob(search_pattern)
    if not files:
        print(f"No files found for {signal_type} in {folder_path}")
        return None
    # Sort by modification time
    return max(files, key=os.path.getmtime)


def filter_subs(df,signal_type):
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
    return df


def create_col(df,colNames,colTypes):


    return df