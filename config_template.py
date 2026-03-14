
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
levels_dict = {21: 'Low', 41: 'Moderate', 81: 'High'}


# ----- Optional: subjects to exclude from run ------
skip_glob   = []        # Subjects with issues not specific to one signal type
skip_emg    = []
skip_hr     = []
skip_scr    = []
