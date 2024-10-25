import soccerdata as sd
import pandas as pd
from tqdm import tqdm
import logging
import boto3
from dotenv import load_dotenv
from io import StringIO
import os
import curses
from utils import get_european_schedule
from pathlib import Path


supported_leagues = [
    "ESP-La Liga",
    # "FRA-Ligue 1",
    "GER-Bundesliga",
    # "ITA-Serie A",
    # "ENG-Premier League"
]


class HistoricData:
    def __init__(
        self,
        season_id: int,
        league_id: str,
        num_games: int = 380,
        env_path: str = None,
        bucket: str = "footballbets",
    ):

        self._create_session(env_path)

        self.bucket: str = bucket

        if league_id not in supported_leagues:
            raise ValueError(
                f"Currently only supporting top 5 European leagues. Please choose one of the folloiwing: {supported_leagues}"
            )

        self.season_id: int = season_id
        self.league_id: str = league_id

        self.ws = sd.WhoScored(leagues=self.league_id, seasons=self.season_id)

        self.mh = sd.MatchHistory(leagues=self.league_id, seasons=self.season_id)

        self.understat = sd.Understat(leagues=self.league_id, seasons=self.season_id)

        self.schedule = self._get_league_schedule()

        logging.info(len(self.schedule))

        self._load_event_data()
        self._load_player_ratings()
        self._load_player_match_stats()
        self._load_team_match_data()
        try:
            self._load_odds()
        except:
            with open("scraper_notes.txt", "a") as file:
                file.write(
                    f"{self.league_id} | {self.season_id} \n Betting Data Unavailable (must download manually) \n"
                )

    def _load_event_data(self):
        game_ids = list(self.schedule.ws_game_id)
        event_data = pd.DataFrame()

        for id in game_ids:
            try:
                match_data = self.ws.read_events(id, output_fmt="spadl")
                event_data = pd.concat([event_data, match_data])
            except:
                with open("scraper_notes.txt", "a") as file:
                    file.write(
                        f"{self.league_id} | {self.season_id} \n WhoScored Game ID {id} unable to load events \n"
                    )

        logging.info("SPADL Data Loaded")
        event_data.merge(
            self.schedule[["game", "home_team_id", "ws_game_id"]],
            how="left",
            left_on="game_id",
            right_on="ws_game_id",
        )
        spadl_buffer = StringIO()

        event_data.to_csv(spadl_buffer, index=True)
        self.s3.put_object(
            Bucket=self.bucket,
            Key=f"{self.league_id}/{self.season_id}/events_spadl.csv",
            Body=spadl_buffer.getvalue(),
        )

        logging.info("SPADL Data Into S3")

    def _load_player_ratings(self):
        game_ids = list(self.schedule.ws_game_id)
        player_ratings = pd.DataFrame()

        for id in game_ids:
            try:
                match_ratings = self.ws.read_player_ratings(id)
                player_ratings = pd.concat([player_ratings, match_ratings])
            except:
                with open("scraper_notes.txt", "a") as file:
                    file.write(
                        f"{self.league_id} | {self.season_id} \n WhoScored Game ID {id} unable to load ratings \n"
                    )

        logging.info("Player Ratings Loaded")

        ratings_buffer = StringIO()

        player_ratings.to_csv(ratings_buffer, index=True)
        self.s3.put_object(
            Bucket=self.bucket,
            Key=f"{self.league_id}/{self.season_id}/player_ratings.csv",
            Body=ratings_buffer.getvalue(),
        )

        logging.info("Ratings Data Into S3")

    def _load_missing_players(self):
        game_ids = list(self.schedule.ws_game_id)
        missing_players = pd.DataFrame()

        # self.ws.read_missing_players(list(self.schedule.ws_game_id))

        for id in game_ids:
            try:
                match_missing_players = self.ws.read_missing_players(id)
                missing_players = pd.concat([missing_players, match_missing_players])
            except:
                with open("scraper_notes.txt", "a") as file:
                    file.write(
                        f"{self.league_id} | {self.season_id} \n WhoScored Game ID {id} unable to load missing players \n"
                    )

        logging.info("Missing Player Data Loaded")

        players_buffer = StringIO()
        missing_players.to_csv(players_buffer, index=True)
        self.s3.put_object(
            Bucket=self.bucket,
            Key=f"{self.league_id}/{self.season_id}/missing_players.csv",
            Body=players_buffer.getvalue(),
        )

        logging.info("Missing Player Data Into S3")

    def _load_player_match_stats(self):

        player_match_data = self.understat.read_player_match_stats(
            list(self.schedule.und_game_id)
        )

        logging.info("Player Data Loaded")

        players_match = StringIO()
        player_match_data.to_csv(players_match, index=True)
        self.s3.put_object(
            Bucket=self.bucket,
            Key=f"{self.league_id}/{self.season_id}/player_stats.csv",
            Body=players_match.getvalue(),
        )

        logging.info("Player Data Into S3")

    def _load_team_match_data(self):
        team_match_data = self.understat.read_team_match_stats(
            list(self.schedule.und_game_id)
        )

        logging.info("Team Data Loaded")

        team_match = StringIO()
        team_match_data.to_csv(team_match, index=True)
        self.s3.put_object(
            Bucket=self.bucket,
            Key=f"{self.league_id}/{self.season_id}/team_stats.csv",
            Body=team_match.getvalue(),
        )

        logging.info("Team Data Into S3")

    def _load_odds(self):
        odds_data = self.mh.read_games()

        logging.info("Odds Data Loaded")

        odds_match = StringIO()
        odds_data.to_csv(odds_match, index=True)
        self.s3.put_object(
            Bucket=self.bucket,
            Key=f"{self.league_id}/{self.season_id}/odds.csv",
            Body=odds_match.getvalue(),
        )

        logging.info("Team Match Stats Into S3")

    def _get_league_schedule(self):

        epl_schedule = self.ws.read_schedule().reset_index()

        understat_schedule = self.understat.read_schedule().reset_index()

        european_schedule = get_european_schedule(self.season_id)

        master_schedule = epl_schedule.merge(
            understat_schedule[["game", "game_id"]],
            on="game",
            how="left",
        )

        master_schedule.rename(
            columns={
                "game_id_x": "ws_game_id",
                "game_id_y": "und_game_id",
            },
            inplace=True,
        )

        master_schedule = master_schedule.reset_index()

        with open("scraper_notes.txt", "a") as file:
            ws_differences = set(epl_schedule.home_team.unique()).difference(
                understat_schedule.home_team.unique()
            )
            us_differences = set(understat_schedule.home_team.unique()).difference(
                epl_schedule.home_team.unique()
            )
            file.write(
                f"{self.league_id} | {self.season_id} \n WhoScored Team Diff: {ws_differences} \n Understat Team Diff: {us_differences} \n"
            )

        schedule_match = StringIO()
        master_schedule.to_csv(schedule_match, index=True)
        self.s3.put_object(
            Bucket=self.bucket,
            Key=f"{self.league_id}/{self.season_id}/schedule.csv",
            Body=schedule_match.getvalue(),
        )

        europe = StringIO()
        european_schedule.to_csv(europe, index=True)
        self.s3.put_object(
            Bucket=self.bucket,
            Key=f"European_Schedules/{self.season_id}_schedule.csv",
            Body=europe.getvalue(),
        )

        return master_schedule

    def _create_session(self, env_path: str):

        load_dotenv(env_path)

        aws_access_key = os.getenv("AWS_ACCESS_KEY")
        aws_secret_access = os.getenv("AWS_SECRET_ACCESS")
        aws_region = os.getenv("AWS_REGION")

        try:
            self.s3 = boto3.client(
                "s3",
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_access,
                region_name=aws_region,
            )
        except Exception as e:
            raise ConnectionError(
                "Could not connect to AWS. Check credentials and .env file"
            ) from e


def main():

    # for league in supported_leagues:
    for season in [1718, 1819, 1920, 2021, 2122, 2223, 2324]:
        for league in supported_leagues:
            HistoricData(season_id=season, league_id=league)


if __name__ == "__main__":
    main()
