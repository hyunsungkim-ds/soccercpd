from datetime import datetime

import numpy as np
import pandas as pd


class Kloppy:
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.players = np.sort([c[:-2] for c in data.columns if c[:4] in ["home", "away"] and c.endswith("_x")])
        self.play_records = None
        self.player_periods = None

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

    def compute_play_records(self) -> pd.DataFrame:
        play_records = dict()

        for p in self.players:
            player_x = self.data[f"{p}_x"].dropna()
            play_records[p] = {"in_frame": player_x.index[0], "out_frame": player_x.index[-1]}

        return pd.DataFrame(play_records).T

    def round_change_times(self):
        self.play_records = self.compute_play_records()

        chg_frames = np.concatenate([self.play_records["in_frame"].unique(), self.play_records["out_frame"].unique()])
        chg_frames = pd.DataFrame(np.sort(chg_frames), columns=["recorded"])
        chg_frames["rounded"] = chg_frames["recorded"]

        chg_ids = (chg_frames["recorded"].diff() > 750).astype(int).cumsum()
        chg_frames_rounded = ((chg_frames["recorded"].groupby(chg_ids).mean() + 1) / 250).astype(int) * 250

        for i in chg_frames_rounded.index[1:-1]:
            chg_frames.loc[chg_ids == i, "rounded"] = chg_frames_rounded[i]

        round_dict = chg_frames.set_index("recorded")["rounded"].to_dict()
        self.play_records["in_frame"] = self.play_records["in_frame"].map(round_dict)
        out_frames = self.play_records["out_frame"].map(round_dict)
        self.play_records["out_frame"] = np.where(out_frames % 250 == 0, out_frames - 1, out_frames)

        for p in self.play_records.index:
            t_in = self.play_records.at[p, "in_frame"]
            t_out = self.play_records.at[p, "out_frame"]
            player_cols = [c for c in self.data.columns if c.startswith(p)]
            inplay_data = self.data.loc[t_in:t_out, player_cols]

            self.data.loc[t_in:t_out, player_cols] = inplay_data.interpolate(limit_direction="both")
            self.data.loc[: t_in - 1, player_cols] = np.nan
            self.data.loc[t_out + 1 :, player_cols] = np.nan
            self.data.loc[t_in:t_out, player_cols]

    def label_player_periods(self):
        player_periods = []

        for team in ["home", "away"]:
            self.data[f"{team}_pp"] = 0

            team_play_records = self.play_records[self.play_records.index.str.startswith(team)]
            session_starts = self.data.groupby("period_id")["frame_id"].first().values
            chg_frames = np.sort(np.append(team_play_records["out_frame"].unique() + 1, session_starts))

            for i, t_start in enumerate(chg_frames[:-1]):
                t_end = chg_frames[i + 1] - 1
                self.data.loc[t_start:t_end, f"{team}_pp"] = i + 1

                team_x_cols = [c for c in self.data.columns if c[:4] == team and c[-2:] == "_x"]
                if team == "home":
                    gk = int(self.data.loc[t_start:t_end, team_x_cols].mean().idxmin().split("_")[1])
                else:
                    gk = int(self.data.loc[t_start:t_end, team_x_cols].mean().idxmax().split("_")[1])

                player_periods.append({"team": team, "start_frame": t_start, "end_frame": t_end, "goalkeeper": gk})

            self.player_periods = pd.DataFrame(player_periods)

    def convert_to_soccercpd_input(self, exclude_gks=True):
        if "home_pp" not in self.data.columns:
            self.label_player_periods()

        time_cols = ["datetime", "period_id", "timestamp", "frame_id"]
        gks = self.player_periods["goalkeeper"].unique()
        data_list = []

        for p in self.play_records.index:
            if exclude_gks and int(p.split("_")[-1]) in gks:
                continue

            col_dict = {c: c.split("_")[-1] for c in self.data.columns if c.startswith(p)}
            player_data = self.data[col_dict.keys()].copy().rename(columns=col_dict)

            player_data["team"] = p.split("_")[0]
            if p.split("_")[0] == "away":
                player_data["x"] = -player_data["x"]
                player_data["y"] = -player_data["y"]

            player_data["player_id"] = int(p.split("_")[1])
            player_data["player_period"] = self.data[f"{p.split('_')[0]}_pp"]

            data_list.append(pd.concat([self.data[time_cols], player_data], axis=1))

        col_dict = {"frame_id": "frame", "period_id": "session", "timestamp": "time", "s": "speed"}
        cols = ["datetime", "team", "player_id", "player_period", "session", "time", "x", "y", "speed"]
        return pd.concat(data_list).rename(columns=col_dict)[cols]
