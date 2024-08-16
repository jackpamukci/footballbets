import os
import sys
from pathlib import Path
import json
import logging

sys.path.append("..")
BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = Path(BASE_DIR, "config")

# Team name replacements
TEAMNAME_REPLACEMENTS = {}

_f_custom_teamnname_replacements = CONFIG_DIR / "teamname_replacements.json"
if _f_custom_teamnname_replacements.is_file():
    with _f_custom_teamnname_replacements.open(encoding="utf8") as json_file:
        for team, to_replace_list in json.load(json_file).items():
            for to_replace in to_replace_list:
                TEAMNAME_REPLACEMENTS[to_replace] = team
    # logger.info(
    #     "Custom team name replacements loaded from %s.",
    #     _f_custom_teamnname_replacements,
    # )
else:
    # logger.info(
    #     "No custom team name replacements found. You can configure these in %s.",
    #     _f_custom_teamnname_replacements,
    # )
    print("didnt work")

STADIUM_LOCATIONS = {}

_f_stadium_locations = CONFIG_DIR / "stadium_locations.json"
if _f_stadium_locations.is_file():
    with _f_stadium_locations.open(encoding="utf8") as json_file:
        for team, address in json.load(json_file).items():
            STADIUM_LOCATIONS[team] = address


SCHEDULE_COLUMNS = set(
    [
        "Unnamed: 0",
        "index",
        "league",
        "season",
        "game",
        "stage_id",
        "ws_game_id",
        "status",
        "start_time",
        "home_team_id",
        "home_team",
        "home_yellow_cards",
        "home_red_cards",
        "away_team_id",
        "away_team",
        "away_yellow_cards",
        "away_red_cards",
        "has_incidents_summary",
        "has_preview",
        "score_changed_at",
        "elapsed",
        "last_scorer",
        "is_top_match",
        "home_team_country_code",
        "away_team_country_code",
        "comment_count",
        "is_lineup_confirmed",
        "is_stream_available",
        "match_is_opta",
        "home_team_country_name",
        "away_team_country_name",
        "date",
        "home_score",
        "away_score",
        "incidents",
        "bets",
        "aggregate_winner_field",
        "winner_field",
        "period",
        "extra_result_field",
        "home_extratime_score",
        "away_extratime_score",
        "home_penalty_score",
        "away_penalty_score",
        "started_at_utc",
        "first_half_ended_at_utc",
        "second_half_started_at_utc",
        "stage",
        "fbref_game_id",
        "und_game_id",
    ]
)
