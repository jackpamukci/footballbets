import os
import pandas as pd
import boto3
from dotenv import load_dotenv
from io import StringIO
import socceraction.spadl as spadl


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

        _data = self._load_data()
        self.events = _data["events"]
        self.lineups = _data["lineups"]
        self.missing_players = _data["missing_players"]
        self.odds = _data["odds"]
        self.player_stats = _data["player_stats"]
        self.schedule = _data["schedule"]
        self.team_stats = _data["team_stats"]

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
