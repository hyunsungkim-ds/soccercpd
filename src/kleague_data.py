import fnmatch
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd

from src import utils


class KLeagueData:
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.play_records = None
        self.subperiods = None

    def preprocess_times(self):
        start_dt = datetime(2024, 1, 1, 1)
        self.data["datetime"] = start_dt + self.data["timestamp"]
        self.data["timestamp"] = self.data["timestamp"].apply(lambda x: x.total_seconds())
        self.data["timestamp"] = ((self.data["timestamp"] * 25).astype(int) / 25).round(2)

        time_cols = ["period_id", "timestamp", "frame_id"]
        data_cols = [c for c in self.data.columns if c.split("_")[-1] in ["x", "y", "z", "d", "s", "speed"]]
        data_list = []

        for i in self.data["period_id"].unique():
            session_data = self.data[self.data["period_id"] == i].copy()
            session_data["timestamp"] = session_data["timestamp"] - session_data["timestamp"].iloc[0]
            session_data = session_data.resample("0.04S", on="datetime").first()
            session_data[time_cols + data_cols] = session_data[time_cols + data_cols].interpolate(limit_area="inside")
            data_list.append(session_data)

        self.data = pd.concat(data_list).astype({"period_id": int, "frame_id": int}).reset_index()

    @staticmethod
    def compute_play_records(data: pd.DataFrame) -> pd.DataFrame:
        players = utils.list_players(data)
        play_records = dict()

        for p in players:
            player_x = data[f"{p}_x"].dropna()
            play_records[p] = {"in_frame": player_x.index[0], "out_frame": player_x.index[-1]}

        return pd.DataFrame(play_records).T

    @staticmethod
    def label_subperiods(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Snap substitutions into constant-roster subperiods and interpolate dropouts. This is shared
        # with the Sportec path via utils.derive_subperiods; here we additionally pick the
        # goalkeeper per subperiod as the player with the most extreme mean x (min for home, max for away).
        fps = 25
        if "timestamp" in data.columns:
            steps = data.groupby("period_id")["timestamp"].diff()
            step = steps[steps > 0].median()
            if step and step > 0:
                fps = int(round(1.0 / step))

        data = utils.derive_subperiods(data, fps=fps)

        subperiods = []
        for team in ["home", "away"]:
            team_x_cols = fnmatch.filter(data.columns, f"{team}_*_x")
            for subperiod in sorted(p for p in data[f"{team}_sub_id"].unique() if p > 0):
                subperiod_data = data.loc[data[f"{team}_sub_id"] == subperiod]
                mean_x = subperiod_data[team_x_cols].mean()
                if mean_x.dropna().empty:
                    continue

                gk_col = mean_x.idxmin() if team == "home" else mean_x.idxmax()
                subperiods.append(
                    {
                        "home_away": team,
                        "start_frame": int(subperiod_data["frame_id"].iloc[0]),
                        "end_frame": int(subperiod_data["frame_id"].iloc[-1]),
                        "goalkeeper": int(gk_col.split("_")[1]),
                    }
                )

        return data, pd.DataFrame(subperiods)

    @staticmethod
    def rotate_pitch(data: pd.DataFrame, sessions_to_rotate=None):
        data = data.copy()

        home_x_cols = fnmatch.filter(data.columns, "home_*_x")
        home_y_cols = fnmatch.filter(data.columns, "home_*_y")
        away_x_cols = fnmatch.filter(data.columns, "away_*_x")
        away_y_cols = fnmatch.filter(data.columns, "away_*_y")
        xy_cols = home_x_cols + home_y_cols + away_x_cols + away_y_cols

        if sessions_to_rotate is not None:
            for i in sessions_to_rotate:
                session_data: pd.DataFrame = data[data["period_id"] == i]
                data.loc[session_data.index, xy_cols] = -session_data[xy_cols]

        elif home_x_cols and away_x_cols:
            for i in data["period_id"].unique():
                session_data: pd.DataFrame = data[data["period_id"] == i]
                home_mean_x = session_data[home_x_cols].mean().mean()
                away_mean_x = session_data[away_x_cols].mean().mean()
                if home_mean_x > away_mean_x:
                    data.loc[session_data.index, xy_cols] = -session_data[xy_cols]

        else:
            x_cols = home_x_cols if home_x_cols else away_x_cols
            mean_x_list = []

            for i in data["period_id"].unique():
                session_data: pd.DataFrame = data[data["period_id"] == i]
                session_mean_x = session_data[x_cols].mean().mean()
                mean_x_list.append(session_mean_x)

            if np.mean(mean_x_list[0::2]) < np.mean(mean_x_list[1::2]):
                even_session_data = data[data["period_id"] % 2 == 0]
                data.loc[even_session_data.index, xy_cols] = -even_session_data[xy_cols]
            else:
                odd_session_data = data[data["period_id"] % 2 == 1]
                data.loc[odd_session_data.index, xy_cols] = -odd_session_data[xy_cols]

        return data

    def convert_to_soccercpd_input(self, exclude_gks=True):
        if "home_sub_id" not in self.data.columns and "away_sub_id" not in self.data.columns:
            self.data, self.subperiods = KLeagueData.label_subperiods(self.data)

        players = utils.list_players(self.data)
        if exclude_gks and self.subperiods is None:
            _, self.subperiods = KLeagueData.label_subperiods(self.data)
        gks = self.subperiods["goalkeeper"].unique() if exclude_gks else []
        time_cols = ["datetime", "period_id", "timestamp", "frame_id"]
        data_list = []

        for p in players:
            if exclude_gks and int(p.split("_")[-1]) in gks:
                continue

            col_dict = {c: c.split("_")[-1] for c in self.data.columns if c.startswith(p)}
            player_data = self.data[col_dict.keys()].copy().rename(columns=col_dict)

            player_data["home_away"] = p.split("_")[0]
            if "home_sub_id" in self.data.columns and p.split("_")[0] == "away":
                player_data["x"] = -player_data["x"]
                player_data["y"] = -player_data["y"]

            player_data["player_id"] = int(p.split("_")[1])
            player_data["subperiod_id"] = self.data[f"{p.split('_')[0]}_sub_id"]

            data_list.append(pd.concat([self.data[time_cols], player_data], axis=1))

        rename_dict = {"speed": "s"}
        cols = ["datetime", "period_id", "timestamp", "subperiod_id", "home_away", "player_id", "x", "y", "s"]
        return pd.concat(data_list).rename(columns=rename_dict)[cols]
