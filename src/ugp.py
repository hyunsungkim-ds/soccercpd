from datetime import timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm


class UGP:
    def __init__(self, activity_id: int) -> None:
        self.activity_id = activity_id

        player_periods = pd.read_pickle("data/player_periods.pkl")
        player_periods = player_periods[player_periods["activity_id"] == self.activity_id].copy()
        self.player_periods = player_periods.set_index("player_period").loc[1:]

        self.data = pd.read_pickle(f"data/ugp/{activity_id}.ugp")
        self.inplay_data = None

    def construct_inplay_data(self):
        start_dts = self.player_periods.groupby("session")["start_dt"].first()
        end_dts = self.player_periods.groupby("session")["end_dt"].last()
        sessions = pd.concat([start_dts, end_dts], axis=1)
        data_list = []

        for s in sessions.index:
            start_dt = sessions.at[s, "start_dt"]
            end_dt = sessions.at[s, "end_dt"]

            session_pp = self.player_periods[self.player_periods["session"] == s]
            session_dr = pd.date_range(start_dt + timedelta(seconds=0.1), end_dt, freq="0.1S")

            times = pd.DataFrame(s, index=session_dr, columns=["session"])
            times["time"] = (times.index.to_series() - start_dt).apply(lambda x: x.total_seconds())

            for player_id in self.data["player_id"].unique():
                player_data = self.data.loc[self.data["player_id"] == player_id, ["x", "y", "speed"]]
                player_data = pd.merge(times, player_data, left_index=True, right_index=True, how="left")
                player_data[["x", "y"]] /= 100
                player_data["player_id"] = int(player_id)
                player_data["player_period"] = 0

                for pp in session_pp.index:
                    start_dt = self.player_periods.at[pp, "start_dt"] + timedelta(seconds=0.1)
                    end_dt = self.player_periods.at[pp, "end_dt"]
                    player_data.loc[start_dt:end_dt, "player_period"] = pp
                    if player_id not in session_pp.at[pp, "player_ids"]:
                        player_data.loc[start_dt:end_dt, ["x", "y", "speed"]] = np.nan

                data_list.append(player_data.reset_index().rename(columns={"index": "datetime"}))

        self.inplay_data = pd.concat(data_list).sort_values(["player_id", "session", "time"], ignore_index=True)

    def rotate_pitch(self, pitch_size=(104, 68)):
        session_mean_x = self.inplay_data.groupby("session")["x"].mean()
        session_to_rotate = 1 if session_mean_x.loc[1] > session_mean_x.loc[2] else 2
        session_data = self.inplay_data[self.inplay_data["session"] == session_to_rotate]
        self.inplay_data.loc[session_data.index, "x"] = pitch_size[0] - session_data["x"]
        self.inplay_data.loc[session_data.index, "y"] = pitch_size[1] - session_data["y"]
