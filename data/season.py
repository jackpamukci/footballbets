import os
import pandas as pd
import boto3
from dotenv import load_dotenv
from io import StringIO
import socceraction.spadl as spadl
from data import utils
from data.xg import xG
import numpy as np


class Season:
    def __init__(
        self,
        league_id: str,
        season_id: int,
        get_dist: bool = False,
        env_path: str = None,
        bucket: str = "footballbets",
    ):

        self.league_id = league_id
        self.season_id = season_id
        self.bucket = bucket
        self.env_path = env_path
        self.get_dist = get_dist
        self._get_s3_agent()

        _data = self._load_data()
        self.events = _data["events"]
        self.missing_players = _data["missing_players"]
        self.odds = _data["odds"]
        self.player_stats = _data["player_stats"]
        self.schedule = _data["schedule"]
        self.team_stats = _data["team_stats"]

        print("schdule")
        self._process_schedule()

        print("process data")
        self._process_event_data()

        # self._process_player_names()

    def _process_event_data(self):
        self.events = spadl.add_names(self.events)

        self.events = self.events.merge(
            self.schedule[["game", "home_team_id", "ws_game_id"]].rename(
                columns={"game": "fixture"}
            ),
            how="left",
            left_on="game_id",
            right_on="ws_game_id",
        )

        self.events["h.a"] = self.events.apply(
            lambda x: "home" if x.team_id == x.home_team_id else "away", axis=1
        )

        self.events["prevEvent"] = self.events.shift(1, fill_value=0)["type_name"]
        self.events["nextEvent"] = self.events.shift(-1, fill_value=0)["type_name"]
        self.events["nextTeamId"] = self.events.shift(-1, fill_value=0)["team_id"]

        self.events = utils.get_season_possessions(self.events)

        xgm = xG(self.events)
        self.events["xG"] = xgm.get_xg()

        self.events = pd.concat([self.events, utils.get_vaep(self.events)], axis=1)

    def _process_schedule(self):

        if self.get_dist == True:
            self.schedule = utils.get_distances(self.schedule)

        s3_europe = self.s3.get_object(
            Bucket=self.bucket,
            Key=f"European_Schedules/{self.season_id}_schedule.csv",
        )
        europe = pd.read_csv(StringIO(s3_europe["Body"].read().decode("utf-8")))

        self.schedule = pd.concat(
            [self.schedule, europe], ignore_index=True
        ).sort_values("start_time")

        self.schedule = self._get_rest_days()

    def _process_player_names(self):
        player_list = self.player_stats.player.unique()
        self.missing_players["player"] = self.missing_players.player.apply(
            lambda x: utils.best_name_match(x, player_list)
        )
        self.player_stats["player"] = self.player_stats.player.apply(
            lambda x: utils.best_name_match(x, player_list)
        )
        # self.events["player"] = self.events.player.apply(
        #     lambda x: utils.best_name_match(x, player_list)
        # )

    def _get_rest_days(self):
        schedule = self.schedule
        league = self.league_id

        # For filtering out european fixtures from schedule
        teams = schedule[schedule["league"] == league].home_team.unique()

        schedule["home_rest"] = np.nan
        schedule["away_rest"] = np.nan

        for team in teams:
            # Filter the schedule for the specific team (either home or away)
            team_sched = schedule[
                (schedule["home_team"] == team) | (schedule["away_team"] == team)
            ].sort_values("start_time", ascending=True)
            team_sched["previous_match_start"] = team_sched.shift(
                1, fill_value=team_sched.iloc[0].start_time
            )["start_time"]

            # Calculate the previous match start time
            schedule.loc[team_sched.index, "previous_match_start"] = team_sched[
                "start_time"
            ].shift(1, fill_value=team_sched.iloc[0].start_time)

            # Convert start_time to start_date and previous_match_start to previous_match_date
            schedule.loc[team_sched.index, "start_date"] = team_sched[
                "start_time"
            ].apply(lambda x: pd.to_datetime(str(x).split("T")[0]))
            schedule.loc[team_sched.index, "previous_match_date"] = team_sched[
                "previous_match_start"
            ].apply(lambda x: pd.to_datetime(str(x).split("T")[0]))

            home_condition = team_sched["home_team"] == team
            schedule.loc[team_sched.index[home_condition], "home_rest"] = (
                schedule.loc[team_sched.index[home_condition], "start_date"]
                - schedule.loc[team_sched.index[home_condition], "previous_match_date"]
            ).dt.days

            # Calculate away rest days where the team is the away team
            away_condition = team_sched["away_team"] == team
            schedule.loc[team_sched.index[away_condition], "away_rest"] = (
                schedule.loc[team_sched.index[away_condition], "start_date"]
                - schedule.loc[team_sched.index[away_condition], "previous_match_date"]
            ).dt.days

        schedule = schedule.reset_index(drop=True).sort_values(
            "start_time", ascending=True
        )
        schedule = schedule[schedule["league"] == league]

        return schedule

    def _load_data(self):
        s3_events = self.s3.get_object(
            Bucket=self.bucket,
            Key=f"{self.league_id}/{self.season_id}/events_spadl.csv",
        )
        events = pd.read_csv(StringIO(s3_events["Body"].read().decode("utf-8")))

        # s3_lineups = self.s3.get_object(
        #     Bucket=self.bucket,
        #     Key=f"{self.league_id}/{self.season_id}/lineups.csv",
        # )
        # lineups = pd.read_csv(StringIO(s3_lineups["Body"].read().decode("utf-8")))

        s3_missing_players = self.s3.get_object(
            Bucket=self.bucket,
            Key=f"{self.league_id}/{self.season_id}/missing_players.csv",
        )
        missing_players = pd.read_csv(
            StringIO(s3_missing_players["Body"].read().decode("utf-8"))
        )

        s3_odds = self.s3.get_object(
            Bucket=self.bucket,
            Key=f"{self.league_id}/{self.season_id}/odds.csv",
        )
        odds = pd.read_csv(StringIO(s3_odds["Body"].read().decode("utf-8")))

        non_odds = odds.columns[:6]
        odd_nums = [
            "B365>2.5",
            "B365<2.5",
            "P>2.5",
            "P<2.5",
            "B365H",
            "B365D",
            "B365A",
            "PSH",
            "PSD",
            "PSA",
            "Max>2.5",
            "Max<2.5",
            "Avg>2.5",
            "Avg<2.5",
            "MaxH",
            "MaxD",
            "MaxA",
            "AvgH",
            "AvgD",
            "AvgA",
        ]
        cols = list(non_odds) + odd_nums
        odds = odds[cols]

        s3_player_stats = self.s3.get_object(
            Bucket=self.bucket,
            Key=f"{self.league_id}/{self.season_id}/player_stats.csv",
        )
        player_stats = pd.read_csv(
            StringIO(s3_player_stats["Body"].read().decode("utf-8"))
        )

        s3_schedule = self.s3.get_object(
            Bucket=self.bucket,
            Key=f"{self.league_id}/{self.season_id}/schedule.csv",
        )
        schedule = pd.read_csv(StringIO(s3_schedule["Body"].read().decode("utf-8")))

        s3_team_stats = self.s3.get_object(
            Bucket=self.bucket,
            Key=f"{self.league_id}/{self.season_id}/team_stats.csv",
        )
        team_stats = pd.read_csv(StringIO(s3_team_stats["Body"].read().decode("utf-8")))

        return {
            "events": events,
            # "lineups": lineups,
            "missing_players": missing_players,
            "odds": odds,
            "player_stats": player_stats,
            "schedule": schedule,
            "team_stats": team_stats,
        }

    def _get_s3_agent(self):
        try:
            load_dotenv(self.env_path)
            aws_access_key = os.getenv("AWS_ACCESS_KEY")
            aws_secret_access = os.getenv("AWS_SECRET_ACCESS")
            aws_region = os.getenv("AWS_REGION")

            self.s3 = boto3.client(
                "s3",
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_access,
                region_name=aws_region,
            )
        except Exception as e:
            raise ConnectionError("Connection to AWS Failed. Check Credentials.") from e
