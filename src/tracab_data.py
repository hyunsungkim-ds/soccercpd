import fnmatch
import json
import os
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from src import config, utils
from src.match_data import MatchData
from src.tracab_metadata import load_tracab_metadata_tables


class TracabData(MatchData):
    def __init__(
        self,
        match_id: str,
        load_tracking: bool = True,
        tracking_limit: Optional[int] = None,
        fps: float = 25.0,
        tracking_source: str = "processed",
    ):
        super().__init__()
        self.match_id = match_id
        self.data_dir = Path(__file__).resolve().parent.parent / "data" / "tracab"
        self.match_dir = self.data_dir / "raw" / match_id
        self.raw_tracking_dir = self.match_dir / f"TRACKING_DATA_{match_id}"
        self.tracking_path = self.match_dir / "tracking.parquet"
        self.processed_tracking_path = self.data_dir / "tracking_processed" / f"{match_id}.parquet"
        self.metadata_path, self.event_path, self.raw_tracking_path = self.resolve_paths(self.raw_tracking_dir)

        self.periods, players = load_tracab_metadata_tables(self.metadata_path)
        self.lineup = TracabData.build_lineup(players, self.periods)
        self.events, keeper_ids = TracabData.load_event_data(self.event_path, self.lineup)
        self.lineup["is_keeper"] = self.lineup["player_id"].isin(keeper_ids)

        self.tracking = None
        if load_tracking:
            self.fps = TracabData.load_frame_rate(self.metadata_path) or fps

            # Prefer the fully processed tracking (running features + ball_state/owner already labelled),
            # which is the SoccerCPD-ready wide frame; fall back to the raw kloppy conversion.
            if tracking_source == "processed" and self.processed_tracking_path.exists():
                self.tracking = pd.read_parquet(self.processed_tracking_path)
            elif self.tracking_path.exists():
                self.tracking = pd.read_parquet(self.tracking_path)
            else:
                self.tracking = self.load_tracking_data(
                    self.metadata_path,
                    self.raw_tracking_path,
                    self.lineup,
                    limit=tracking_limit,
                )
                self.tracking.to_parquet(self.tracking_path)

            if "frame_id" in self.tracking.columns:
                self.tracking["frame_id"] = np.arange(0, len(self.tracking))
            else:
                self.tracking.index.name = "frame_id"
                self.tracking.reset_index(inplace=True)

    EVENT_COLUMN_MAP = {
        "event_id": "event_id",
        "player_seq_id": "event_group_id",
        "half_time": "period_id",
        "match_run_time_in_ms": "timestamp",
        "event_type": "category",
        "event": "event_type",
        "from_player_id": "player_id",
        "from_player_name": "player_name",
        "x_location_start": "start_x",
        "y_location_start": "start_y",
        "to_player_id": "end_player_id",
        "to_player_name": "end_player_name",
        "x_location_end": "end_x",
        "y_location_end": "end_y",
        "outcome": "result",
        "body_type": "body_part_type",
    }

    OFF_BALL_TYPES = [
        "possession_outcome",
        "foul_for",
        "offside",
        "referee_event",
        "goal_conceded",
        "offer",
        "no_offer",
        "pressing",
        "pushing_on",
        "take_on_against",
        "goal_prevention",
        "active_engagement",
        "defensive_line_support",
        "throw",
        "kick_from_hands",
        "no_event",
        "game_period_start",
        "game_period_end",
        "kickoff",
        "substitution_on",
    ]

    @staticmethod
    def resolve_paths(raw_tracking_dir: Path) -> Tuple[str, str, str]:
        metadata_paths = sorted(raw_tracking_dir.glob("*_metadata.json"))
        if not metadata_paths:
            raise FileNotFoundError(f"No metadata found in {raw_tracking_dir}.")

        event_paths = sorted(raw_tracking_dir.parent.glob("*_Events.csv"))
        if not event_paths:
            raise FileNotFoundError(f"No event data file found in {raw_tracking_dir.parent}.")

        tracking_paths = sorted(raw_tracking_dir.glob("*.dat"))
        if not tracking_paths:
            raise FileNotFoundError(f"No tracking data file found in {raw_tracking_dir}.")

        return str(metadata_paths[0]), str(event_paths[0]), str(tracking_paths[0])

    @staticmethod
    def load_frame_rate(metadata_path: str) -> float:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return float(metadata["FrameRate"])

    @staticmethod
    def build_lineup(players: pd.DataFrame, periods: pd.DataFrame) -> pd.DataFrame:
        lineup = players.copy().rename(
            columns={"match_id": "match_id", "team_side": "home_away", "jersey_no": "uniform_number"}
        )
        lineup["object_id"] = lineup["home_away"] + "_" + lineup["uniform_number"].astype(str)

        first_period_start = periods["start_frame"].min() if not periods.empty else None
        lineup["is_starting"] = lineup["is_active"] & (lineup["start_frame"] == first_period_start)

        column_order = [
            "match_id",
            "team_id",
            "team_name",
            "home_away",
            "player_id",
            "player_name",
            "uniform_number",
            "object_id",
            "start_frame",
            "end_frame",
            "n_frames",
            "is_active",
            "is_starting",
        ]
        team_order = {"home": 0, "away": 1}
        lineup = lineup[column_order].copy()
        lineup["_team_order"] = lineup["home_away"].map(team_order)
        lineup = lineup.sort_values(["_team_order", "uniform_number"], ignore_index=True).drop(columns="_team_order")
        return lineup

    @staticmethod
    def load_event_data(event_path: str, lineup: pd.DataFrame):
        events: pd.DataFrame = pd.read_csv(event_path, header=0)
        events = events[TracabData.EVENT_COLUMN_MAP.keys()].rename(columns=TracabData.EVENT_COLUMN_MAP)
        events["timestamp"] = events["timestamp"] / 1000
        events[["start_x", "end_x"]] *= config.PITCH_X
        events[["start_y", "end_y"]] *= config.PITCH_Y

        keeper_ids = set(events.loc[events["category"] == "goal_keeping", "player_id"].dropna().unique())

        for period_id in events["period_id"].unique():
            mask = events["period_id"] == period_id
            pass_events = events.loc[mask & (events["event_type"] == "pass"), "timestamp"]
            if not pass_events.empty:
                kickoff_ts = pass_events.iloc[0]
            else:
                kickoff_ts = events.loc[mask & (events["event_type"] == "game_period_start"), "timestamp"].iloc[0]
            events.loc[mask, "timestamp"] -= kickoff_ts

        player_mapping = lineup.set_index("player_id")["object_id"]
        events["player_id"] = events["player_id"].map(player_mapping)
        events["end_player_id"] = events["end_player_id"].map(player_mapping)

        return events, keeper_ids

    @staticmethod
    def rename_tracking_columns(tracking: pd.DataFrame, lineup: pd.DataFrame) -> pd.DataFrame:
        tracking = tracking.copy()
        player_mapping = (
            lineup.assign(player_id=lineup["player_id"].astype(str)).set_index("player_id")["object_id"].to_dict()
        )
        column_mapping = {}

        for col in tracking.columns:
            for player_id, object_id in player_mapping.items():
                prefix = f"{player_id}_"
                if col.startswith(prefix):
                    column_mapping[col] = f"{object_id}_{col[len(prefix):]}"
                    break

        tracking = tracking.rename(columns=column_mapping)

        if "ball_owning_team_id" in tracking.columns:
            team_mapping = {}
            unique_teams = lineup[["team_id", "home_away"]].drop_duplicates()
            for _, row in unique_teams.iterrows():
                team_mapping[row["team_id"]] = row["home_away"]
                team_mapping[str(row["team_id"])] = row["home_away"]

            tracking["ball_owning_team_id"] = tracking["ball_owning_team_id"].map(
                lambda value: team_mapping.get(value, team_mapping.get(str(value), value))
            )

        player_cols = [col for col in tracking.columns if col[:4] in ["home", "away"]]
        ordered_player_cols = [
            f"{object_id}_{suffix}"
            for object_id in lineup["object_id"]
            for suffix in ["x", "y", "d", "s"]
            if f"{object_id}_{suffix}" in tracking.columns
        ]
        if player_cols:
            player_col_set = set(player_cols)
            reordered_cols = []
            inserted = False
            for col in tracking.columns:
                if col in player_col_set:
                    if not inserted:
                        reordered_cols.extend(ordered_player_cols)
                        inserted = True
                    continue
                reordered_cols.append(col)
            tracking = tracking[reordered_cols]

        home_x_cols = [c for c in tracking.columns if fnmatch.fnmatch(c, "home_*_x")]
        away_x_cols = [c for c in tracking.columns if fnmatch.fnmatch(c, "away_*_x")]
        if home_x_cols or away_x_cols:
            tracking = tracking.dropna(subset=home_x_cols + away_x_cols, how="all").copy()

        return tracking

    @staticmethod
    def load_tracking_data(
        metadata_path: str,
        raw_tracking_path: str,
        lineup: pd.DataFrame,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        from kloppy import tracab
        from kloppy.domain import Dimension, MetricPitchDimensions, Orientation

        tracking_ds = tracab.load(
            meta_data=metadata_path,
            raw_data=raw_tracking_path,
            limit=limit,
            coordinates="tracab",
            only_alive=False,
        )

        pitch_dims = MetricPitchDimensions(
            standardized=True,
            x_dim=Dimension(0, config.PITCH_X),
            y_dim=Dimension(0, config.PITCH_Y),
        )
        tracking_ds = tracking_ds.transform(
            to_orientation=Orientation.STATIC_HOME_AWAY,
            to_pitch_dimensions=pitch_dims,
        )
        tracking_df = tracking_ds.to_df().copy()
        if "timestamp" in tracking_df.columns and pd.api.types.is_timedelta64_dtype(tracking_df["timestamp"]):
            tracking_df["timestamp"] = tracking_df["timestamp"].dt.total_seconds()

        tracking_df = TracabData.rename_tracking_columns(tracking_df, lineup)
        return tracking_df

    @staticmethod
    def align_events_to_tracking(events: pd.DataFrame, tracking: pd.DataFrame) -> pd.DataFrame:
        events = events.copy()
        events["frame_id"] = -1

        if "frame_id" in tracking.columns:
            tracking = tracking.set_index("frame_id")

        track_ts = tracking["timestamp"].to_numpy(dtype=float)
        track_periods = tracking["period_id"].to_numpy()
        track_frame_ids = tracking.index.to_numpy()

        for period in events["period_id"].unique():
            ev_mask = events["period_id"] == period
            tr_idx = np.where(track_periods == period)[0]
            if len(tr_idx) == 0:
                continue

            tr_ts = track_ts[tr_idx]
            ev_ts = events.loc[ev_mask, "timestamp"].to_numpy(dtype=float)

            nearest = np.searchsorted(tr_ts, ev_ts, side="left").clip(0, len(tr_ts) - 1)
            left = np.clip(nearest - 1, 0, len(tr_ts) - 1)
            nearest = np.where(np.abs(tr_ts[left] - ev_ts) <= np.abs(tr_ts[nearest] - ev_ts), left, nearest)
            events.loc[ev_mask, "frame_id"] = track_frame_ids[tr_idx[nearest]]

        # Flip event coordinates per period if they are mirrored relative to tracking
        for period in events["period_id"].unique():
            pass_mask = (events["period_id"] == period) & (events["event_type"] == "pass")
            if not pass_mask.any():
                continue

            pass_events = events.loc[pass_mask]
            pass_x = pass_events["start_x"].to_numpy(dtype=float)
            pass_y = pass_events["start_y"].to_numpy(dtype=float)

            frames = pass_events["frame_id"].to_numpy()
            players = pass_events["player_id"].to_numpy()
            track_x = np.array([tracking.at[t, f"{p}_x"] for t, p in zip(frames, players)])
            track_y = np.array([tracking.at[t, f"{p}_y"] for t, p in zip(frames, players)])

            valid = np.isfinite(track_x) & np.isfinite(track_y)
            if valid.any():
                dist = np.sqrt((pass_x[valid] - track_x[valid]) ** 2 + (pass_y[valid] - track_y[valid]) ** 2)
                if dist.mean() > 20:
                    period_mask = events["period_id"] == period
                    for col_x, col_y in [("start_x", "start_y"), ("end_x", "end_y")]:
                        if col_x in events.columns:
                            events.loc[period_mask, col_x] = config.PITCH_X - events.loc[period_mask, col_x]
                        if col_y in events.columns:
                            events.loc[period_mask, col_y] = config.PITCH_Y - events.loc[period_mask, col_y]

        # Convert out_of_play possession_outcomes based on next set piece (after coordinate flipping)
        outcome_mask = events["event_type"] == "possession_outcome"
        oop_mask = events["result"].str.contains("out_of_play", na=False)
        for i in events.index[outcome_mask & oop_mask]:
            for j in range(i + 1, min(i + 10, len(events))):
                next_type = events.at[j, "event_type"]
                if next_type in ("goalkick", "corner"):
                    sp_x = events.at[j, "start_x"]
                    events.at[i, "event_type"] = "out"
                    events.at[i, "player_id"] = "out_left" if sp_x < config.PITCH_X / 2 else "out_right"
                    break
                elif next_type == "throwin":
                    sp_y = events.at[j, "start_y"]
                    events.at[i, "event_type"] = "out"
                    events.at[i, "player_id"] = "out_bottom" if sp_y < config.PITCH_Y / 2 else "out_top"
                    break

        # Fill out_* events with ball position
        frame_valid = events["frame_id"] >= 0
        out_mask = events["player_id"].str.startswith("out_", na=False) & frame_valid
        if out_mask.any():
            frames = events.loc[out_mask, "frame_id"].to_numpy()
            events.loc[out_mask, "start_x"] = np.array([tracking.at[t, "ball_x"] for t in frames])
            events.loc[out_mask, "start_y"] = np.array([tracking.at[t, "ball_y"] for t in frames])

        # Fill remaining missing (x, y) from player tracking position
        xy_missing = events["start_x"].isna() & events["player_id"].notna() & frame_valid
        if xy_missing.any():
            frames = events.loc[xy_missing, "frame_id"].to_numpy()
            players = events.loc[xy_missing, "player_id"].to_numpy()
            events.loc[xy_missing, "start_x"] = np.array([tracking.at[t, f"{p}_x"] for t, p in zip(frames, players)])
            events.loc[xy_missing, "start_y"] = np.array([tracking.at[t, f"{p}_y"] for t, p in zip(frames, players)])

        # ---- Clean up around OOP events ----
        # Process in frame order. Two passes:
        #   (1) Deduplicate consecutive out events: among a run of adjacent outs, keep only
        #       the first whose (x, y) is actually off-pitch. Drop the rest.
        #   (2) Between each surviving out and the immediately following OOP set piece
        #       (throwin/corner/goalkick within 10 events), drop any other events that fell
        #       in between so out and set piece are adjacent in the final sequence.
        sorted_idx = events.sort_values("frame_id", kind="stable").index.tolist()
        to_drop: set = set()

        def _is_off_pitch(idx):
            x = events.at[idx, "start_x"]
            y = events.at[idx, "start_y"]
            if pd.isna(x) or pd.isna(y):
                return False
            return x < 0 or x > config.PITCH_X or y < 0 or y > config.PITCH_Y

        # Pass (1): consecutive out dedup
        def _flush_group(group):
            if len(group) <= 1:
                return
            kept = next((gi for gi in group if _is_off_pitch(gi)), group[0])
            for gi in group:
                if gi != kept:
                    to_drop.add(gi)

        current_group: list = []
        for idx in sorted_idx:
            if events.at[idx, "event_type"] == "out":
                current_group.append(idx)
            else:
                _flush_group(current_group)
                current_group = []
        _flush_group(current_group)

        # Pass (2): drop events between out and next OOP set piece
        position = {idx: pos for pos, idx in enumerate(sorted_idx)}
        oop_set_piece = ("throwin", "corner", "goalkick")
        for out_idx in [idx for idx in sorted_idx if events.at[idx, "event_type"] == "out" and idx not in to_drop]:
            pos = position[out_idx]
            for k in range(pos + 1, min(pos + 10, len(sorted_idx))):
                nxt_idx = sorted_idx[k]
                if nxt_idx in to_drop:
                    continue
                if events.at[nxt_idx, "event_type"] in oop_set_piece:
                    for mid_pos in range(pos + 1, k):
                        mid_idx = sorted_idx[mid_pos]
                        if mid_idx not in to_drop:
                            to_drop.add(mid_idx)
                    break

        if to_drop:
            events = events.drop(index=list(to_drop)).reset_index(drop=True)

        return events

    @staticmethod
    def find_spadl_event_types(events: pd.DataFrame) -> pd.DataFrame:
        if "spadl_type" in events.columns and "success" in events.columns:
            return events

        events = events.copy()
        events["spadl_type"] = pd.Series(np.nan, index=events.index, dtype="object")
        events["success"] = pd.Series(np.nan, index=events.index, dtype="object")
        result = events["result"].fillna("")

        # Group-level lookups
        group_types = events.groupby("event_group_id")["event_type"].apply(set)

        # --- pass ---
        mask = events["event_type"] == "pass"
        events.loc[mask, "spadl_type"] = "pass"
        events.loc[mask, "success"] = result[mask] == "possession_complete"
        for idx in events.index[mask]:
            group = group_types.get(events.at[idx, "event_group_id"])
            if "goalkick" in group:
                events.at[idx, "spadl_type"] = "goalkick"
            elif "corner" in group:
                events.at[idx, "spadl_type"] = "corner_short"
            elif "freekick" in group:
                events.at[idx, "spadl_type"] = "freekick_short"

        # --- cross ---
        mask = events["event_type"] == "cross"
        events.loc[mask, "spadl_type"] = "cross"
        events.loc[mask, "success"] = result[mask] == "possession_complete"
        for idx in events.index[mask]:
            group = group_types.get(events.at[idx, "event_group_id"])
            if "corner" in group:
                events.at[idx, "spadl_type"] = "corner_crossed"
            elif "freekick" in group:
                events.at[idx, "spadl_type"] = "freekick_crossed"

        # --- throwin -> throw_in ---
        mask = events["event_type"] == "throwin"
        events.loc[mask, "spadl_type"] = "throw_in"
        events.loc[mask, "success"] = result[mask] == "teammate_reception"

        # --- reception -> control ---
        mask = events["event_type"] == "reception"
        events.loc[mask, "spadl_type"] = "control"
        events.loc[mask, "success"] = True

        # --- tackle ---
        mask = events["event_type"] == "tackle"
        keep_mask = result[mask].str.contains("interrupted|won", na=False)
        drop_mask = result[mask].str.contains("lost|opponent_retained", na=False)
        events.loc[mask, "spadl_type"] = "tackle"
        events.loc[mask, "success"] = result[mask].str.contains("won", na=False)
        # Remove tackles with lost/opponent_retained
        events.loc[mask & ~keep_mask & drop_mask, "spadl_type"] = np.nan

        # --- duel, aerial_duel, physical_duel -> tackle ---
        duel_types = {"duel", "aerial_duel", "physical_duel"}
        duel_mask = events["event_type"].isin(duel_types)
        events.loc[duel_mask, "spadl_type"] = "bad_touch"
        events.loc[duel_mask, "success"] = result[duel_mask].str.contains("won|possession_retained", na=False)

        # Pair duels: group by proximity (within 2 seconds)
        duel_indices = events.index[duel_mask].tolist()
        duel_ts = events.loc[duel_mask, "timestamp"].to_numpy(dtype=float)
        drop_duel = set()
        visited = set()
        for i, idx in enumerate(duel_indices):
            if idx in visited:
                continue
            # Find pair partner: different event_group_id, within 2s
            partners = []
            gid_i = events.at[idx, "event_group_id"]
            for j in range(i + 1, len(duel_indices)):
                jdx = duel_indices[j]
                if duel_ts[j] - duel_ts[i] > 1.0:
                    break
                if events.at[jdx, "event_group_id"] != gid_i:
                    partners.append(jdx)
                    break

            if not partners:
                # Singleton: keep as is
                continue

            pair = [idx, partners[0]]
            visited.update(pair)
            pair_results = [result[p] for p in pair]
            has_won = [("won" in r or "possession_retained" in r) for r in pair_results]

            if any(has_won):
                # Keep only those with won/possession_retained
                for k, p in enumerate(pair):
                    if not has_won[k]:
                        drop_duel.add(p)
            else:
                # No winner: remove lost/opposition_retained, keep rest
                has_lost = [("lost" in r or "opposition_retained" in r) for r in pair_results]
                if any(has_lost):
                    for k, p in enumerate(pair):
                        if has_lost[k]:
                            drop_duel.add(p)
                else:
                    # No lost either: keep interrupted
                    for k, p in enumerate(pair):
                        if "interrupted" not in pair_results[k]:
                            drop_duel.add(p)

        if drop_duel:
            events.loc[list(drop_duel), "spadl_type"] = np.nan

        # --- attempt_at_goal -> shot / shot_freekick / shot_penalty ---
        mask = events["event_type"] == "attempt_at_goal"
        events.loc[mask, "spadl_type"] = "shot"
        for idx in events.index[mask]:
            group = group_types.get(events.at[idx, "event_group_id"])
            if "freekick" in group:
                events.at[idx, "spadl_type"] = "shot_freekick"
            elif "penalty" in group:
                events.at[idx, "spadl_type"] = "shot_penalty"
            events.at[idx, "success"] = events.at[idx, "result"] == "on_target" and "goal" in group

        # --- ball_progression -> take_on ---
        mask = events["event_type"] == "ball_progression"
        events.loc[mask, "spadl_type"] = "take_on"
        events.loc[mask, "success"] = result[mask] == "opposition_beaten"

        # --- goalkeeper_intervention -> keeper_save / keeper_claim ---
        mask = events["event_type"] == "goalkeeper_intervention"
        events.loc[mask, "spadl_type"] = "keeper_claim"
        for idx in events.index[mask]:
            # Check if attempt_at_goal exists in preceding events (up to 5 rows back)
            for j in range(max(0, idx - 5), idx):
                if events.at[j, "event_type"] == "attempt_at_goal":
                    events.at[idx, "spadl_type"] = "keeper_save"
                    break

        # --- foul_against -> foul ---
        mask = events["event_type"] == "foul_against"
        events.loc[mask, "spadl_type"] = "foul"

        # --- clearance ---
        mask = events["event_type"] == "clearance"
        events.loc[mask, "spadl_type"] = "clearance"

        # --- block -> shot_block ---
        mask = events["event_type"] == "block"
        events.loc[mask, "spadl_type"] = "shot_block"

        # --- interception ---
        mask = events["event_type"] == "interception"
        events.loc[mask, "spadl_type"] = "interception"

        # --- goal ---
        mask = events["event_type"] == "goal"
        events.loc[mask, "spadl_type"] = "goal"
        for idx in events.index[mask]:
            x = events.at[idx, "start_x"]
            events.at[idx, "player_id"] = "goal_left" if x < config.PITCH_X / 2 else "goal_right"

        # --- out ---
        mask = events["event_type"] == "out"
        events.loc[mask, "spadl_type"] = "out"

        # Drop events without spadl_type
        events = events[events["spadl_type"].notna()].reset_index(drop=True)
        # events = events[~events["event_type"].isin(TracabData.OFF_BALL_TYPES)].reset_index(drop=True)

        return events

    @staticmethod
    def label_episodes(
        events: pd.DataFrame, tracking: pd.DataFrame, fps: float = 25.0
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        events = events.copy()
        events["episode_id"] = 0

        ep = 1
        prev_frame = -1
        for i in events.index:
            fid = events.at[i, "frame_id"]
            if fid < 0:
                continue

            spadl = events.at[i, "spadl_type"]
            is_set_piece = spadl in config.SET_PIECE
            gap = (fid - prev_frame) / fps if prev_frame >= 0 else 0

            if ep == 1 and prev_frame < 0:
                pass
            elif is_set_piece or gap >= 20:
                ep += 1

            events.at[i, "episode_id"] = ep
            prev_frame = fid

        # Label tracking episodes based on events
        tracking = tracking.copy()
        tracking["episode_id"] = 0
        if "ball_state" not in tracking.columns:
            tracking["ball_state"] = "dead"

        ep_ids = sorted(events.loc[events["episode_id"] > 0, "episode_id"].unique())
        for i, ep_id in enumerate(ep_ids):
            ep_events = events[events["episode_id"] == ep_id]
            ep_start_frame = ep_events["frame_id"].min()
            ep_end_frame = int(ep_events["frame_id"].max() + fps)  # 1 second after last event

            # Clamp to next episode start if overlapping
            if i + 1 < len(ep_ids):
                next_start = events.loc[events["episode_id"] == ep_ids[i + 1], "frame_id"].min()
                ep_end_frame = min(ep_end_frame, next_start - 1)

            alive_mask = tracking["frame_id"].between(ep_start_frame, ep_end_frame)
            tracking.loc[alive_mask, "episode_id"] = ep_id

        tracking.loc[tracking["episode_id"] == 0, "ball_state"] = "dead"
        tracking.loc[tracking["episode_id"] > 0, "ball_state"] = "alive"

        return events, tracking

    @staticmethod
    def label_possessions(events: pd.DataFrame, tracking: pd.DataFrame) -> pd.DataFrame:
        tracking = tracking.copy()
        tracking["player_id"] = pd.Series(None, index=tracking.index, dtype=object)
        tracking["ball_owning_team_id"] = pd.Series(None, index=tracking.index, dtype=object)

        event_poss = events[["frame_id", "player_id"]].dropna(subset="player_id")
        event_poss = event_poss.drop_duplicates(subset="frame_id", keep="first").set_index("frame_id")["player_id"]
        is_event_frame = tracking["frame_id"].isin(event_poss.index)
        tracking.loc[is_event_frame, "player_id"] = tracking.loc[is_event_frame, "frame_id"].map(event_poss).values

        for ep_id in tracking.loc[tracking["episode_id"] > 0, "episode_id"].unique():
            ep_mask = tracking["episode_id"] == ep_id
            poss_prev = tracking.loc[ep_mask, "player_id"].ffill()
            poss_next = tracking.loc[ep_mask, "player_id"].bfill()
            filled = poss_prev.where(poss_prev == poss_next, None)

            first_valid = filled.first_valid_index()
            if first_valid is not None:
                filled.loc[:first_valid] = filled.loc[:first_valid].bfill()

            last_valid = filled.last_valid_index()
            if last_valid is not None:
                filled.loc[last_valid:] = filled.loc[last_valid:].ffill()
            tracking.loc[ep_mask, "player_id"] = filled

        for ep_id in tracking.loc[tracking["episode_id"] > 0, "episode_id"].unique():
            ep_mask = tracking["episode_id"] == ep_id
            ep_team_poss = tracking.loc[ep_mask, "player_id"].apply(utils.player_to_team)
            tracking.loc[ep_mask, "ball_owning_team_id"] = ep_team_poss.bfill().ffill()

        return tracking

    @staticmethod
    def simplify_events(events: pd.DataFrame, tracking: pd.DataFrame, fps: float = 25.0) -> pd.DataFrame:
        """Simplify spadl_type events to kick/control/out, matching sportec/kleague event_processed format.

        Keeps only events whose frame_id exists in tracking (drops penalty-shootout events that have
        frame_id == -1), and rewrites timestamps so they are exact multiples of 1/fps tied to frame_id.
        """
        from src.config import INCOMING, OUTGOING

        events = events.copy()
        events = events.drop(columns=["event_type"], errors="ignore")
        events = events.rename(columns={"spadl_type": "event_type"})
        events = events[events["event_type"] != "foul"].copy()

        # Keep only events whose frame_id appears in tracking (drops frame_id=-1 / penalty shootout)
        events = events[events["frame_id"].isin(tracking["frame_id"])].copy()

        # Recompute timestamp to be (frame_id - period_start_frame) / fps so it aligns with frame ticks
        period_start = tracking.groupby("period_id")["frame_id"].min()
        events["timestamp"] = (events["frame_id"] - events["period_id"].map(period_start)) / fps

        events = events.sort_values("frame_id", ignore_index=True, kind="stable")

        next_player = events.groupby("episode_id")["player_id"].shift(-1)

        simple = pd.Series(index=events.index, dtype=object)
        simple[events["event_type"] == "out"] = "out"

        non_out = events["event_type"] != "out"
        same_player = events["player_id"] == next_player
        simple[non_out & same_player] = "control"
        simple[non_out & ~same_player & next_player.notna()] = "kick"

        last_mask = non_out & next_player.isna()
        simple[last_mask & events["event_type"].isin(OUTGOING)] = "kick"
        simple[last_mask & events["event_type"].isin(INCOMING)] = "control"
        simple[last_mask & simple.isna()] = "kick"

        events["event_type"] = simple
        events = events[events["event_type"].isin(["kick", "control", "out"])].copy()

        events["timestamp"] = events["timestamp"].apply(utils.seconds_to_timestamp)
        out_cols = ["frame_id", "period_id", "episode_id", "timestamp", "player_id", "event_type", "start_x", "start_y"]
        return events[out_cols].reset_index(drop=True)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / "data/tracab/raw"
    event_dir = base_dir / "data/tracab/event"
    event_output_dir = base_dir / "data/tracab/event_processed"
    tracking_output_dir = base_dir / "data/tracab/tracking_processed"
    os.makedirs(event_dir, exist_ok=True)
    os.makedirs(event_output_dir, exist_ok=True)
    os.makedirs(tracking_output_dir, exist_ok=True)

    match_dirs = sorted(d for d in raw_dir.iterdir() if d.is_dir() and any(d.glob("TRACKING_DATA_*")))

    for i, match_path in enumerate(match_dirs):
        match_id = match_path.name
        print(f"\n[{i + 1}/{len(match_dirs)}] {match_id}")

        match = TracabData(match_id, load_tracking=True)

        events = TracabData.align_events_to_tracking(match.events, match.tracking)
        events = TracabData.find_spadl_event_types(events)
        events, tracking = TracabData.label_episodes(events, match.tracking, fps=match.fps)
        events.to_parquet(event_dir / f"{match_id}.parquet", index=False)

        tracking = TracabData.label_possessions(events, tracking)
        drop_cols = [c for c in tracking.columns if c.endswith(("_d", "_s"))] + ["ball_speed"]
        tracking = tracking.drop(columns=drop_cols, errors="ignore")

        tracking_proc = utils.calculate_running_features(tracking, fps=match.fps)
        tracking_proc.to_parquet(tracking_output_dir / f"{match_id}.parquet", index=False)

        events_proc = TracabData.simplify_events(events, tracking, fps=match.fps)
        events_proc.to_parquet(event_output_dir / f"{match_id}.parquet", index=False)
