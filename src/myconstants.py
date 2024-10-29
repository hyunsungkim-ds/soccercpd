import pandas as pd

# File paths and variable names
DIR_DATA = "./data"
DIR_UGP_DATA = f"{DIR_DATA}/ugp"
DIR_TEMP_DATA = f"{DIR_DATA}/rtemp"

# DataFrame headers
HEADER_ROLE_SEQ = [
    "player_id",
    "session",
    "time",
    "subsession",
    "player_period",
    "form_period",
    "role_period",
    "x",
    "y",
    "x_norm",
    "y_norm",
    "role",
    "base_role",
    "switch_rate",
]
HEADER_FORM_PERIODS = [
    "form_period",
    "session",
    "start_dt",
    "end_dt",
    "duration",
    "adj_mat",
]
HEADER_ROLE_PERIODS = [
    "form_period",
    "role_period",
    "session",
    "start_dt",
    "end_dt",
    "duration",
    "base_perm",
]
HEADER_ROLE_ALIGNS = [
    "activity_id",
    "form_period",
    "base_role",
    "aligned_role",
    "x",
    "y",
]
HEADER_ROLE_SUMMARY = [
    "activity_id",
    "player_period",
    "form_period",
    "role_period",
    "session",
    "start_dt",
    "end_dt",
    "duration",
    "player_id",
    "base_role",
    "x",
    "y",
]

# Numeric constants
SCALAR_CENTI = 100
SCALAR_MILLI = 1000
SCALAR_MICRO = 1000000
SCALAR_TIME = 60

# Hyperparameters for SoccerCPD
MAX_SWITCH_RATE = 0.8
MAX_PVAL = 0.01
MIN_PERIOD_DUR = 300
MIN_FORM_DIST = 6

ROLE_TEMPLATE = pd.DataFrame(
    [
        ["343", "LWB", "LCB", "CB", "RCB", "RWB", "RCM", "LCM", "LM", "CF", "RM"],
        ["352", "LWB", "LCB", "CB", "RCB", "RWB", "CDM", "LCM", "RCM", "LCF", "RCF"],
        ["442", "LB", "LCB", "RCB", "RB", "LCM", "RCM", "LM", "LCF", "RCF", "RM"],
        ["4231", "LB", "LCB", "RCB", "RB", "LDM", "RDM", "CAM", "LM", "CF", "RM"],
        ["433", "LB", "LCB", "RCB", "RB", "CDM", "LCM", "RCM", "LM", "CF", "RM"],
        ["4132", "LB", "LCB", "RCB", "RB", "CDM", "CAM", "LM", "LCF", "RCF", "RM"],
        ["others"] + [f"R{i}" for i in list(range(1, 11))],
    ],
    columns=["formation"] + list(range(1, 11)),
).set_index("formation")
