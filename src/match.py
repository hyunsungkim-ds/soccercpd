import os
from datetime import datetime, timedelta
from typing import Tuple

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rpy2.robjects.packages as rpackages
from matplotlib import animation
from tqdm import tqdm

from src.soccercpd import SoccerCPD


class Match:
    def __init__(self, home_id: int, away_id: int = 0) -> None:
        self.activity_ids = {"H": home_id, "A": away_id}
        self.roster = pd.read_csv(f"private_data/roster/{home_id}-{away_id}.csv", header=0).set_index("player_code")
        self.traces = pd.read_csv(f"private_data/traces/{home_id}-{away_id}.csv", header=0, parse_dates=["datetime"])

        self.team_traces = dict()
        self.phase_records = dict()
        self.team_cpd = dict()

    # optionally rotate the pitch so that each team always attack from left to right
    def rotate_pitch(self, pitch_size: Tuple = (104, 68)):
        for s in self.traces["session"].unique():
            s_traces = self.traces[self.traces["session"] == s]

            home_xy_cols = [f"{p}_{x}" for p in self.roster.index for x in ["x", "y"] if p[0] == "H"]
            away_xy_cols = [f"{p}_{x}" for p in self.roster.index for x in ["x", "y"] if p[0] == "A"]
            home_mean_x = s_traces[home_xy_cols[0::2]].mean().mean()
            away_mean_x = s_traces[away_xy_cols[0::2]].mean().mean()

            if home_mean_x < away_mean_x:
                self.traces.loc[s_traces.index, away_xy_cols[0::2]] = pitch_size[0] - s_traces[away_xy_cols[0::2]]
                self.traces.loc[s_traces.index, away_xy_cols[1::2]] = pitch_size[1] - s_traces[away_xy_cols[1::2]]
            else:
                self.traces.loc[s_traces.index, home_xy_cols[0::2]] = pitch_size[0] - s_traces[home_xy_cols[0::2]]
                self.traces.loc[s_traces.index, home_xy_cols[1::2]] = pitch_size[1] - s_traces[home_xy_cols[1::2]]

    def construct_team_traces(self):
        for team in ["H", "A"]:
            team_players = [c[:3] for c in self.traces.columns if c[0] == team and c.endswith("_x")]
            phase_col = "home_phase" if team == "H" else "away_phase"

            for phase in self.traces[phase_col].unique():
                team_xy_cols = [f"{p}_{x}" for p in team_players for x in ["x", "y"]]
                phase_x = self.traces.loc[self.traces[phase_col] == phase, team_xy_cols[::2]].dropna(axis=1)

                if len(phase_x.columns) > 10:  # if the goalkeeper was measured
                    gk = phase_x.mean().idxmin()[:3]
                    team_players.remove(gk)

                time_cols = ["datetime", "session", "time", phase_col]
                self.team_traces[team] = self.traces[time_cols + team_xy_cols].rename(columns={phase_col: "phase"})

                grouped = self.team_traces[team].groupby("phase")
                sessions = grouped["session"].first()
                start_dts = (grouped["datetime"].first() - timedelta(seconds=0.1)).rename("start_dt")
                end_dts = grouped["datetime"].last().rename("end_dt")
                self.phase_records[team] = pd.concat([sessions, start_dts, end_dts], axis=1)

    def run_soccercpd(self, team="H"):
        # install and import the R package 'gSeg' to be used in SoccerCPD
        utils = rpackages.importr("utils")
        utils.chooseCRANmirror(ind=1)
        if not rpackages.isinstalled("gSeg"):
            utils.install_packages("gSeg")
        rpackages.importr("gSeg")

        # run SoccerCPD
        team_roster = self.roster[self.roster.index.str.startswith(team)]
        cpd = SoccerCPD(self.activity_ids[team], team_roster, self.team_traces[team])
        cpd.run()

        self.team_cpd[team] = cpd
        return cpd
