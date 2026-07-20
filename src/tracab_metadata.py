import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd

PathLike = Union[str, Path]
PERIOD_COLUMNS = [
    "competition_id",
    "match_id",
    "period_id",
    "start_frame",
    "end_frame",
    "n_frames",
    "duration_seconds",
]
PLAYER_COLUMNS = [
    "match_id",
    "team_side",
    "team_id",
    "team_name",
    "player_id",
    "player_name",
    "jersey_no",
    "start_frame",
    "end_frame",
    "n_frames",
    "is_active",
]

__all__ = ["load_tracab_metadata_tables"]


def _load_metadata(metadata_path: PathLike) -> Dict[str, Any]:
    path = Path(metadata_path)
    with path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    if not isinstance(metadata, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(metadata).__name__}.")

    return metadata


def _count_frames(start_frame: int, end_frame: int) -> int:
    if start_frame <= 0 or end_frame <= 0:
        return 0
    if end_frame < start_frame:
        raise ValueError(f"end_frame must be >= start_frame, got {start_frame} -> {end_frame}.")
    return end_frame - start_frame + 1


def _build_periods_df(metadata: Dict[str, Any], keep_empty_periods: bool) -> pd.DataFrame:
    frame_rate = int(metadata.get("FrameRate", 0) or 0)
    period_records = []

    for period_id in range(1, 6):
        start_frame = int(metadata.get(f"Phase{period_id}StartFrame", 0) or 0)
        end_frame = int(metadata.get(f"Phase{period_id}EndFrame", 0) or 0)
        if (start_frame == 0) != (end_frame == 0):
            raise ValueError(f"Period {period_id} has inconsistent bounds: {start_frame} -> {end_frame}.")
        n_frames = _count_frames(start_frame, end_frame)

        if not keep_empty_periods and n_frames == 0:
            continue

        duration_seconds = n_frames / frame_rate if frame_rate > 0 and n_frames > 0 else 0.0
        period_records.append(
            {
                "competition_id": metadata.get("CompetitionID"),
                "match_id": metadata.get("GameID"),
                "period_id": period_id,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "n_frames": n_frames,
                "duration_seconds": duration_seconds,
            }
        )

    return pd.DataFrame(period_records, columns=PERIOD_COLUMNS)


def _build_players_df(metadata: Dict[str, Any], include_referees: bool) -> pd.DataFrame:
    player_records = []

    for team_key, team_side in [("HomeTeam", "home"), ("AwayTeam", "away")]:
        team_info = metadata.get(team_key, {})
        team_name = team_info.get("LongName") or team_info.get("ShortName")

        for player in team_info.get("Players", []):
            start_frame = int(player.get("StartFrameCount", 0) or 0)
            end_frame = int(player.get("EndFrameCount", 0) or 0)
            first_name = player.get("FirstName") or ""
            last_name = player.get("LastName") or ""
            player_name = " ".join(part for part in [first_name, last_name] if part).strip() or pd.NA

            player_records.append(
                {
                    "match_id": metadata.get("GameID"),
                    "team_side": team_side,
                    "team_id": team_info.get("TeamID"),
                    "team_name": team_name,
                    "player_id": player.get("PlayerID"),
                    "player_name": player_name,
                    "jersey_no": player.get("JerseyNo"),
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "n_frames": _count_frames(start_frame, end_frame),
                    "is_active": start_frame > 0 and end_frame > 0,
                }
            )

    if include_referees:
        for referee in metadata.get("Referees", []):
            start_frame = int(referee.get("StartFrameCount", 0) or 0)
            end_frame = int(referee.get("EndFrameCount", 0) or 0)
            first_name = referee.get("FirstName") or ""
            last_name = referee.get("LastName") or ""
            player_name = " ".join(part for part in [first_name, last_name] if part).strip() or pd.NA

            player_records.append(
                {
                    "match_id": metadata.get("GameID"),
                    "team_side": "referee",
                    "team_id": pd.NA,
                    "team_name": pd.NA,
                    "player_id": referee.get("PlayerID"),
                    "player_name": player_name,
                    "jersey_no": referee.get("JerseyNo"),
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "n_frames": _count_frames(start_frame, end_frame),
                    "is_active": start_frame > 0 and end_frame > 0,
                }
            )

    return pd.DataFrame(player_records, columns=PLAYER_COLUMNS)


def load_tracab_metadata_tables(
    metadata_path: PathLike,
    keep_empty_periods: bool = False,
    include_referees: bool = False,
    keep_empty_phases: Optional[bool] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a FIFA tracking metadata JSON and return period/player summary tables.

    Parameters
    ----------
    metadata_path
        Path to a `*_metadata.json` file.
    keep_empty_periods
        If True, keep zero-filled period slots such as `Phase3StartFrame == 0`.
    include_referees
        If True, append referee rows to the players DataFrame with `team_side='referee'`.
    keep_empty_phases
        Backward-compatible alias for `keep_empty_periods`.

    Returns
    -------
    periods, players
        Two pandas DataFrames containing period-level and player-level metadata.
    """
    if keep_empty_phases is not None:
        keep_empty_periods = keep_empty_phases

    metadata = _load_metadata(metadata_path)
    periods = _build_periods_df(metadata, keep_empty_periods=keep_empty_periods)
    players = _build_players_df(metadata, include_referees=include_referees)
    return periods, players
