from datetime import datetime

import numpy as np
import pandas as pd


class Kloppy:
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.players = np.sort([c[:-2] for c in data.columns if c[:4] in ["home", "away"] and c.endswith("_x")])
        self.play_records = None

    def preprocess_times(self):
        start_dt = datetime(2024, 1, 1)
        self.data["datetime"] = start_dt + self.data["timestamp"]
        self.data["timestamp"] = self.data["timestamp"].apply(lambda x: x.total_seconds())
        self.data["timestamp"] = ((self.data["timestamp"] * 25).astype(int) / 25).round(2)

        time_cols = ["period_id", "timestamp", "frame_id"]
        data_cols = [c for c in self.data.columns if c[:4] in ["home", "away", "ball"]]
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
            play_records[p] = {"start_frame": player_x.index[0], "end_frame": player_x.index[-1]}

        return pd.DataFrame(play_records).T

    def round_change_times(self):
        play_records = self.compute_play_records()

        chg_frames = np.concatenate([play_records["start_frame"].unique(), play_records["end_frame"].unique()])
        chg_frames = pd.DataFrame(np.sort(chg_frames), columns=["recorded"])
        chg_frames["rounded"] = chg_frames["recorded"]

        chg_ids = (chg_frames["recorded"].diff() > 750).astype(int).cumsum()
        chg_frames_rounded = (chg_frames["recorded"].groupby(chg_ids).mean() / 250).astype(int) * 250

        for i in chg_frames_rounded.index[1:-1]:
            chg_frames.loc[chg_ids == i, "rounded"] = chg_frames_rounded[i]

        round_dict = chg_frames.set_index("recorded")["rounded"].to_dict()
        play_records["start_frame"] = play_records["start_frame"].map(round_dict)
        end_frames = play_records["end_frame"].map(round_dict)
        play_records["end_frame"] = np.where(end_frames % 250 == 0, end_frames - 1, end_frames)

        for p in play_records.index:
            ts = play_records.at[p, "start_frame"]
            te = play_records.at[p, "end_frame"]
            player_cols = [c for c in self.data.columns if c.startswith(p)]
            self.data.loc[ts:te, player_cols] = self.data.loc[ts:te, player_cols].interpolate(limit_direction="both")
            self.data.loc[: ts - 1, player_cols] = np.nan
            self.data.loc[te + 1 :, player_cols] = np.nan
            self.data.loc[ts:te, player_cols]

        return play_records
