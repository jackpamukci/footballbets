import os
import pandas as pd
import boto3
from dotenv import load_dotenv
from io import StringIO
import socceraction.spadl as spadl
from data import utils
from data.xg import xG
import logging
from geopy.distance import geodesic


class Events:
    def __init__(
        self,
        league_id: str,
        season_id: int,
        env_path: str = None,
        bucket: str = "footballbets",
    ):

        self.league_id = league_id
        self.season_id = season_id
        self.bucket = bucket
        self.env_path = env_path
        self._get_s3_agent()

        logging.info("Loading Data Now from S3")
        _data = self._load_data()
        self.events = _data["events"]
        logging.info("Events Loaded")
        self.lineups = _data["lineups"]
        logging.info("Lineups Loaded")
        self.missing_players = _data["missing_players"]
        self.odds = _data["odds"]
        self.player_stats = _data["player_stats"]
        self.schedule = _data["schedule"]
        self.team_stats = _data["team_stats"]

        logging.info("Processing Event Data Now")
        # self._process_event_data()
        self._process_schedule(get_dist=True)
        self._process_player_names()

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

        self.events["prevEvent"] = self.events.shift(1, fill_value=0)["type_name"]
        self.events["nextEvent"] = self.events.shift(-1, fill_value=0)["type_name"]
        self.events["nextTeamId"] = self.events.shift(-1, fill_value=0)["team_id"]

        logging.info("Getting possessions from season")
        self.events = utils.get_season_possessions(self.events)

        logging.info("Getting xG")
        xgm = xG(self.events)
        self.events["xG"] = xgm.get_xg()

        logging.info("Getting VAEP")
        self.events = pd.concat([self.events, utils.get_vaep(self.events)], axis=1)

    def _process_schedule(self, get_dist=False):

        if get_dist == True:
            self.schedule = utils.get_distances(self.schedule)

        europe = utils.get_european_schedule(self.season_id)

        self.schedule = pd.concat(
            [self.schedule, europe], ignore_index=True
        ).sort_values("start_time")

    def _process_player_names(self):
        player_list = self.lineups.player.unique()
        self.missing_players["player"] = self.missing_players.player.apply(
            lambda x: utils.best_name_match(x, player_list)
        )
        self.player_stats["player"] = self.player_stats.player.apply(
            lambda x: utils.best_name_match(x, player_list)
        )
        # self.events["player"] = self.events.player.apply(
        #     lambda x: utils.best_name_match(x, player_list)
        # )

    def _load_data(self):
        s3_events = self.s3.get_object(
            Bucket=self.bucket,
            Key=f"{self.league_id}/{self.season_id}/events_spadl.csv",
        )
        events = pd.read_csv(StringIO(s3_events["Body"].read().decode("utf-8")))

        s3_lineups = self.s3.get_object(
            Bucket=self.bucket,
            Key=f"{self.league_id}/{self.season_id}/lineups.csv",
        )
        lineups = pd.read_csv(StringIO(s3_lineups["Body"].read().decode("utf-8")))

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
            "lineups": lineups,
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
