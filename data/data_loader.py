import soccerdata as sd
import pandas as pd
from tqdm import tqdm
import logging
import boto3
from dotenv import load_dotenv
from io import StringIO
import os
import curses

supported_leagues = [
    "ENG-Premier League",
    "ESP-La Liga",
    "FRA-Ligue 1",
    "GER-Bundesliga",
    "ITA-Serie A",
]


class HistoricData:
    def __init__(
        self,
        season_id: int,
        league_id: str,
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

        self.fbref = sd.FBref(leagues=self.league_id, seasons=self.season_id)

        self.mh = sd.MatchHistory(leagues=self.league_id, seasons=self.season_id)

        self.understat = sd.Understat(leagues=self.league_id, seasons=self.season_id)

        self._load_event_data()
        self._load_missing_players()
        self._load_player_match_stats()
        self._load_lineups()
        self._load_team_match_data()
        self._load_odds()

    def _load_event_data(self):
        event_data = self.ws.read_events(output_fmt="spadl")

        logging.info("SPADL Data Loaded")

        spadl_buffer = StringIO()

        event_data.to_csv(spadl_buffer, index=False)
        self.s3.put_object(
            Bucket=self.bucket,
            Key=f"{self.league_id}/{self.season_id}/events/events_spadl.csv",
            Body=spadl_buffer.getvalue(),
        )

        logging.info("SPADL Data Into S3")

    def _load_missing_players(self):
        missing_players = self.ws.read_missing_players()

        logging.info("Missing Player Data Loaded")

        players_buffer = StringIO()
        missing_players.to_csv(players_buffer, index=False)
        self.s3.put_object(
            Bucket=self.bucket,
            Key=f"{self.league_id}/{self.season_id}/missing_players/missing_players.csv",
            Body=players_buffer.getvalue(),
        )

        logging.info("Missing Player Data Into S3")

    def _load_player_match_stats(self):

        player_match_data = self.understat.read_player_match_stats()

        logging.info("Player Data Loaded")

        players_match = StringIO()
        player_match_data.to_csv(players_match, index=False)
        self.s3.put_object(
            Bucket=self.bucket,
            Key=f"{self.league_id}/{self.season_id}/player_match_stats/player_stats.csv",
            Body=players_match.getvalue(),
        )

        logging.info("Player Data Into S3")

    def _load_lineups(self):

        lineups = self.fbref.read_lineup()

        logging.info("Lineup Data Loaded")

        lineups_match = StringIO()
        lineups.to_csv(lineups_match, index=False)
        self.s3.put_object(
            Bucket=self.bucket,
            Key=f"{self.league_id}/{self.season_id}/lineups/lineups.csv",
            Body=lineups_match.getvalue(),
        )

        logging.info("Lineup Data Into S3")

    def _load_team_match_data(self):
        team_match_data = self.understat.read_team_match_stats()

        logging.info("Team Data Loaded")

        team_match = StringIO()
        team_match_data.to_csv(team_match, index=False)
        self.s3.put_object(
            Bucket=self.bucket,
            Key=f"{self.league_id}/{self.season_id}/team_match_stats/team_stats.csv",
            Body=team_match.getvalue(),
        )

        logging.info("Team Data Into S3")

    def _load_odds(self):
        odds_data = self.mh.read_games()

        logging.info("Odds Data Loaded")

        odds_match = StringIO()
        odds_data.to_csv(odds_match, index=False)
        self.s3.put_object(
            Bucket=self.bucket,
            Key=f"{self.league_id}/{self.season_id}/odds_data/odds.csv",
            Body=odds_match.getvalue(),
        )

        logging.info("Team Match Stats Into S3")

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


def main_menu(stdscr, options):
    # Turn off cursor and initialize colors
    curses.curs_set(0)
    curses.start_color()
    curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)

    current_row = 0

    while True:
        stdscr.clear()
        stdscr.addstr(0, 0, "Select an option for league ID from the following:")

        # Display the menu options with highlighting
        for idx, row in enumerate(options):
            if idx == current_row:
                stdscr.attron(curses.color_pair(1))
                stdscr.addstr(idx + 2, 0, f"* {row}")  # Move options down by one row
                stdscr.attroff(curses.color_pair(1))
            else:
                stdscr.addstr(idx + 2, 2, row)  # Move options down by one row

        stdscr.refresh()

        key = stdscr.getch()

        if key == curses.KEY_UP and current_row > 0:
            current_row -= 1
        elif key == curses.KEY_DOWN and current_row < len(options) - 1:
            current_row += 1
        elif key == curses.KEY_ENTER or key in [10, 13]:
            selected_option = options[current_row]
            stdscr.addstr(8, 0, f"You selected: {selected_option}")
            stdscr.refresh()

            # Ask for an integer input using curses
            stdscr.addstr(10, 0, "Enter an integer code: ")
            curses.echo()  # Enable echoing of typed characters
            int_code = stdscr.getstr(
                11, 0, 20
            )  # Get user input, max length of 20 characters
            int_code = int(int_code.decode("utf-8"))  # Decode and convert to integer

            stdscr.addstr(0, 0, f"You entered: {int_code}")
            return selected_option, int_code


def main():

    # Initialize curses and show the menu
    selected_league, selected_season = curses.wrapper(main_menu, supported_leagues)
    print(f"You selected: {selected_league}")

    # Ask for an integer input
    print(f"You entered: {selected_season}")

    print("Starting web scraper...")
    HistoricData(season_id=selected_season, league_id=selected_league)


if __name__ == "__main__":
    main()
