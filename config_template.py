
# config_template.py
# Copy this to config.py and fill in your own paths


""" ================================================================
                        DEFINE EXPERIMENT CONFIG
    ================================================================"""


start_idx = 0           # Keep 0 to run all subjects
loadNew = True          # Run once with True to save acq files as mat files

# ----- Signals to Analyze ------
runEMG= True
runECG = True
runSCR = True

# ----- Define data folders ------
acqDataFolder = "path/to/your/.acq/files"
matDataFolder = "path/to/your/.mat/files"

# ---- Define Biopac Channels ----
eda_col_idx     = 0
hr_col_idx      = 1
emg_col_idx     = 2
trig_col_idx    = 13

# ----- Define experiment levels ------
levels_dict = {21: 'Low',
               41: 'Low-Black',
               61: 'High-Black',
               81: 'High'}

levels_num = {21: 2,
               41: 4,
               61: 6,
               81: 8}


# ----- Define figure design params ------
palette_dict = {
        'Low': '#4C9F70',  # Green
        'Low-Black': '#4C9F70',
        'High': '#C44E52',  # Red
        'High-Black': '#C44E52'
    }

line_styles = {
        'Low': (1, 0),
        'High': (1, 0),
        'Low-Black': (2, 2),
        'High-Black': (2, 2)
    }

custom_x_labels = {
    0: 'Baseline',
    1: 'Bin 1', 2: 'Bin 2', 3: 'Bin 3', 4: 'Bin 4',
    5: 'Heat14',
    6: 'Heat48',
    7: 'Rating',
    8: 'ITI'
}

order = ['Low', 'Low-Black', 'High', 'High-Black']

# ----- Optional: subjects to exclude from run ------
exclusions = {
    'GLOB': [],
    'EMG': [],
    'HR': [],
    'SCR': []
}


