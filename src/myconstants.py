# File paths and variable names
DIR_DATA = "./data"
DIR_UGP_DATA = f"{DIR_DATA}/ugp"
DIR_TEMP_DATA = f"{DIR_DATA}/rtemp"

VARNAME_ACTIVITY_RECORDS = "activity_records"
VARNAME_PLAYER_RECORDS = "player_records"
VARNAME_PLAYER_PERIODS = "player_periods"

# Column names and headers
LABEL_ID = "id"
LABEL_NAME = "name"
LABEL_VARNAME = "varname"
LABEL_RECORDS = "records"
LABEL_HEADER = "header"
LABEL_DTYPES = "dtypes"
LABEL_PATH = "path"
LABEL_FILE = "file"
LABEL_EXTENSION = "extension"

LABEL_TEAM_ID = "team_id"
LABEL_TYPE = "type"
LABEL_DATE = "date"
LABEL_TEAM_NAME = "team_name"
LABEL_HOME_AWAY = "home_away"
LABEL_ROTATED_SESSION = "rotated_session"
LABEL_DATA_SAVED = "data_saved"
LABEL_STATS_SAVED = "stats_saved"
HEADER_ACTIVITY_RECORDS = [
    "activity_id",
    LABEL_TEAM_ID,
    LABEL_TYPE,
    LABEL_DATE,
    LABEL_TEAM_NAME,
    LABEL_HOME_AWAY,
    LABEL_ROTATED_SESSION,
    LABEL_DATA_SAVED,
    LABEL_STATS_SAVED,
]

HEADER_ROSTER = ["player_id", "squad_num", "player_name"]
HEADER_PLAYER_RECORDS = ["activity_id", LABEL_DATE, LABEL_TEAM_NAME] + HEADER_ROSTER

HEADER_PLAYER_PERIODS = [
    "activity_id",
    "phase",
    LABEL_TYPE,
    "session",
    "time",
    "start_dt",
    "end_dt",
    "duration",
    "players",
]

LABEL_UNIXTIME = "unixtime"
LABEL_SPEED = "speed"
HEADER_UGP = [
    "player_code",
    "session",
    "time",
    LABEL_UNIXTIME,
    "phase",
    "duration",
    "x",
    "y",
    LABEL_SPEED,
]

HEADER_ROLE_DETAILS = [
    "player_id",
    "session",
    "time",
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
    "activity_id",
    "form_period",
    "session",
    "start_dt",
    "end_dt",
    "duration",
    "coords",
    "edge_mat",
]
HEADER_ROLE_PERIODS = [
    "activity_id",
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
    "squad_num",
    "player_name",
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
MIN_FORM_DIST = 7
