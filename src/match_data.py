import fnmatch
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from src.config import PITCH_X, PITCH_Y
from src.utils import label_player_segs, list_players, timestamp_to_seconds


class MatchData(ABC):
    def __init__(self):
        self.lineup: pd.DataFrame
        self.events: pd.DataFrame
        self.tracking: pd.DataFrame
        self.fps: float

    def to_soccercpd_input(self, exclude_gks: bool = True, carry_possession: bool = True) -> pd.DataFrame:
        """Convert the wide (kloppy) tracking frame into the long format SoccerCPD consumes.

        Output columns: ``datetime, home_away, player_id, player_seg, period_id, timestamp, x, y, s``
        (plus ``ball_state, ball_owning_team_id`` when ``carry_possession`` and available).

        Coordinates are centered on the pitch and the away team is mirrored so that both teams
        attack the +x direction. kloppy's ``STATIC_HOME_AWAY`` orientation already keeps each team's
        attacking direction consistent across halves, so no per-period rotation is needed. The
        per-frame ``datetime`` is derived from ``frame_id`` (globally monotonic) so it stays ordered
        across the half-time break; ``player_seg`` is derived from substitutions.
        """
        tracking = self.tracking.copy()

        # Ensure a globally monotonic frame_id (kloppy usually provides it).
        if "frame_id" not in tracking.columns:
            tracking["frame_id"] = (tracking["timestamp"] * self.fps).round().astype(int)
            max_frame_p1 = tracking.loc[tracking["period_id"] == 1, "frame_id"].max()
            tracking.loc[tracking["period_id"] == 2, "frame_id"] += max_frame_p1 + 1

        # datetime from frame_id -> monotonic (with a gap at half-time, which is fine).
        base = datetime(2024, 1, 1, 1)
        tracking["datetime"] = base + pd.to_timedelta(tracking["frame_id"] / self.fps, unit="s")

        # Center coordinates (corner origin -> pitch center).
        players = list_players(tracking)
        x_cols = [f"{p}_x" for p in players]
        y_cols = [f"{p}_y" for p in players]
        tracking[x_cols] = tracking[x_cols] - PITCH_X / 2
        tracking[y_cols] = tracking[y_cols] - PITCH_Y / 2

        # Player segments (adds home_player_seg/away_player_seg columns).
        tracking = label_player_segs(tracking, fps=int(round(self.fps)))

        # Goalkeepers to exclude, identified by exact object_id from the lineup when available.
        gk_object_ids = set()
        if exclude_gks:
            lineup = getattr(self, "lineup", None)
            if lineup is not None:
                # GK is flagged by `playing_position == "GK"` (Sportec) or `is_keeper` (Tracab).
                if "playing_position" in lineup.columns:
                    gk_object_ids |= set(lineup.loc[lineup["playing_position"] == "GK", "object_id"])
                if "is_keeper" in lineup.columns:
                    gk_object_ids |= set(lineup.loc[lineup["is_keeper"].fillna(False), "object_id"])

        time_cols = ["datetime", "period_id", "timestamp", "frame_id"]
        if carry_possession:
            time_cols += [c for c in ["ball_state", "ball_owning_team_id"] if c in tracking.columns]

        data_list = []
        for p in players:
            if exclude_gks and p in gk_object_ids:
                continue

            team = p.split("_")[0]
            col_dict = {c: c.split("_")[-1] for c in tracking.columns if c.rsplit("_", 1)[0] == p}
            player_data = tracking[list(col_dict.keys())].copy().rename(columns=col_dict)
            player_data = player_data.rename(columns={"speed": "s"})  # processed uses _speed; kloppy raw uses _s

            if team == "away":  # mirror so the away team also attacks +x
                player_data["x"] = -player_data["x"]
                player_data["y"] = -player_data["y"]

            player_data["home_away"] = team
            player_data["player_id"] = int(p.split("_")[1])
            player_data["player_seg"] = tracking[f"{team}_player_seg"].values

            block = pd.concat(
                [tracking[time_cols].reset_index(drop=True), player_data.reset_index(drop=True)], axis=1
            )
            data_list.append(block)

        out_cols = ["datetime", "period_id", "timestamp", "player_seg", "home_away", "player_id", "x", "y", "s"]
        if carry_possession:
            out_cols += [c for c in ["ball_state", "ball_owning_team_id"] if c in tracking.columns]

        result = pd.concat(data_list, ignore_index=True)[out_cols]
        return result

    @staticmethod
    def calculate_event_seconds(events: pd.DataFrame) -> pd.DataFrame:
        assert "utc_timestamp" in events.columns  # in datetime
        events = events.copy()
        events["timestamp"] = 0.0

        for i in events["period_id"].unique():
            period_events: pd.DataFrame = events[events["period_id"] == i].copy()
            start_dt = period_events["utc_timestamp"].iloc[0]
            period_tds = period_events["utc_timestamp"] - start_dt
            events.loc[period_events.index, "timestamp"] = period_tds.dt.total_seconds()
            # events.loc[period_events.index, "utc_timestamp"] -= timedelta(microseconds=start_dt.microsecond)

        return events

    @staticmethod
    def calculate_tracking_datetimes(events: pd.DataFrame, tracking: pd.DataFrame, fps=25) -> pd.DataFrame:
        assert "timestamp" in tracking.columns  # in seconds
        tracking = tracking.copy()

        if "frame_id" not in tracking.columns:
            tracking["frame_id"] = (tracking["timestamp"] * fps).round().astype(int)
            max_frame_p1 = tracking.loc[tracking["period_id"] == 1, "frame_id"].max()
            tracking.loc[tracking["period_id"] == 2, "frame_id"] += max_frame_p1 + 1

        def utc_timestamp(t: float, offset: np.datetime64) -> np.datetime64:
            return offset + timedelta(seconds=t)

        if events is not None:
            tracking["utc_timestamp"] = pd.NaT
            for i in events["period_id"].unique():
                offset = events[events["period_id"] == i]["utc_timestamp"].iloc[0]
                period_tracking = tracking[tracking["period_id"] == i]
                period_ts = period_tracking["timestamp"].apply(utc_timestamp, args=(offset,))
                tracking.loc[period_ts.index, "utc_timestamp"] = period_ts

        return tracking

    def format_tracking_for_syncer(self) -> pd.DataFrame:
        tracking = self.tracking.copy()

        if "frame_id" not in tracking.columns or "utc_timestamp" not in tracking.columns:
            tracking = MatchData.calculate_tracking_datetimes(self.events, tracking, self.fps)

        home_players = [c[:-2] for c in tracking.columns if fnmatch.fnmatch(c, "home_*_x")]
        away_players = [c[:-2] for c in tracking.columns if fnmatch.fnmatch(c, "away_*_x")]
        objects = home_players + away_players + ["ball"]
        tracking_list = []

        for p in objects:
            object_tracking = tracking[["frame_id", "period_id", "timestamp", "utc_timestamp", "ball_state"]].copy()

            if p == "ball":
                object_tracking["player_id"] = None
                object_tracking["ball"] = True
            else:
                object_tracking["player_id"] = p
                object_tracking["ball"] = False

            object_tracking["x"] = tracking[f"{p}_x"].values.round(2)
            object_tracking["y"] = tracking[f"{p}_y"].values.round(2)
            object_tracking["z"] = tracking["ball_z"].values.round(2) if p == "ball" else np.nan

            for i in object_tracking["period_id"].unique():
                period_tracking = object_tracking[object_tracking["period_id"] == i].dropna(subset=["x"]).copy()
                if not period_tracking.empty:
                    vx = savgol_filter(np.diff(period_tracking["x"].values) * self.fps, window_length=15, polyorder=2)
                    vy = savgol_filter(np.diff(period_tracking["y"].values) * self.fps, window_length=15, polyorder=2)
                    speed = np.sqrt(vx**2 + vy**2)
                    period_tracking.loc[period_tracking.index[1:], "speed"] = speed
                    period_tracking["speed"] = period_tracking["speed"].bfill()

                    accel = savgol_filter(np.diff(speed) * self.fps, window_length=9, polyorder=2)
                    period_tracking.loc[period_tracking.index[1:-1], "accel_s"] = accel
                    period_tracking["accel_s"] = period_tracking["accel_s"].bfill().ffill()

                    ax = savgol_filter(np.diff(vx) * self.fps, window_length=9, polyorder=2)
                    ay = savgol_filter(np.diff(vy) * self.fps, window_length=9, polyorder=2)
                    period_tracking.loc[period_tracking.index[1:-1], "accel_v"] = np.sqrt(ax**2 + ay**2)
                    period_tracking["accel_v"] = period_tracking["accel_v"].bfill().ffill()
                    tracking_list.append(period_tracking)

        out = pd.concat(tracking_list, ignore_index=True)
        out = out[out["ball_state"] == "alive"].drop("ball_state", axis=1).reset_index(drop=True)
        return out.astype({"period_id": int, "z": float})

    @staticmethod
    def merge_events_and_tracking(events: pd.DataFrame, tracking: pd.DataFrame, fps=25, ffill=False) -> pd.DataFrame:
        events = events.copy()

        if "start_x" in events.columns:
            event_cols = ["period_id", "timestamp", "object_id", "spadl_type", "start_x", "start_y"]
        else:
            event_cols = ["period_id", "timestamp", "object_id", "spadl_type", "coordinates_x", "coordinates_y"]

        renamed_cols = ["period_id", "timestamp", "player_id", "event_type", "event_x", "event_y"]
        column_dict = dict(zip(event_cols, renamed_cols))

        events["timestamp"] = ((events["timestamp"] * fps).round().astype(int) / fps).round(2)
        merged = pd.merge(tracking, events[event_cols], how="left").rename(columns=column_dict)

        if ffill:
            merged[renamed_cols[2:]] = merged[renamed_cols[2:]].ffill()

        return merged

    def merge_synced_events_and_tracking(
        events: pd.DataFrame, tracking: pd.DataFrame, fps=25, ffill=False
    ) -> pd.DataFrame:
        assert "timestamp" in events.columns

        column_mapping = {"spadl_type": "event_type", "start_x": "annot_x", "start_y": "annot_y"}
        events = events.copy().rename(columns=column_mapping)

        synced_cols = ["period_id", "timestamp", "player_id", "event_type"]
        synced_events = events.loc[events["timestamp"].notna(), synced_cols].copy().reset_index(drop=True)
        synced_events["timestamp"] = synced_events["timestamp"].apply(timestamp_to_seconds).round(3)

        annot_cols = ["period_id", "utc_timestamp", "annot_x", "annot_y"]
        annot_events = events[annot_cols].copy()
        annot_events = MatchData.calculate_event_seconds(annot_events)
        annot_events["timestamp"] = ((annot_events["timestamp"] * fps).round().astype(int) / fps).round(2)
        annot_events.drop("utc_timestamp", axis=1, inplace=True)

        merged = pd.merge(tracking, synced_events, how="left")
        merged = pd.merge(merged, annot_events, how="left")

        event_mask = merged["player_id"].notna()
        merged.loc[event_mask, "event_x"] = merged.loc[event_mask, "ball_x"]
        merged.loc[event_mask, "event_y"] = merged.loc[event_mask, "ball_y"]

        if ffill:
            ffill_cols = ["player_id", "event_type", "event_x", "event_y", "annot_x", "annot_y"]
            merged[ffill_cols] = merged[ffill_cols].ffill()

        return merged
