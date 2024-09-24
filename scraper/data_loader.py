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
    # "ENG-Premier League",
    # "ESP-La Liga",
    "FRA-Ligue 1",
    "GER-Bundesliga",
    "ITA-Serie A",
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

        self.schedule = self._get_league_schedule().iloc[:num_games]

        logging.info(len(self.schedule))

        self._load_event_data()
        # self._load_missing_players()
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

        # try:
        #     epl_schedule = self.ws.read_schedule().reset_index()
        # except:
        #     stage = self.ws.read_season_stages().iloc[0]
        #     region_id = stage.region_id
        #     stage_id = stage.stage_id
        #     league_id = stage.league_id
        #     season_id = stage.season_id

        #     stage_url = f"https://www.whoscored.com/Regions/{region_id}/Tournaments/{league_id}/Seasons/{season_id}/Stages/{stage_id}"
        #     calendar_filepath = (
        #         self.ws.data_dir / f"matches/{self.league_id}_{self.season_id}.html"
        #     )

        #     cal = self.ws.get(
        #         stage_url,
        #         calendar_filepath,
        #         var="wsCalendar",
        #     )
        #     epl_schedule = self.ws.read_schedule().reset_index()

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
            stdscr.refresh()
            try:
                stdscr.addstr(15, 0, "Enter number of games: ")
                curses.echo()  # Enable echoing of typed characters
                num_games = stdscr.getstr(
                    16, 0, 20
                )  # Get user input, max length of 20 characters
                num_games = int(num_games.decode("utf-8"))
            except ValueError:
                num_games = 380

            return selected_option, int_code, num_games


def main():

    # Initialize curses and show the menu
    # selected_league, selected_season, num_games = curses.wrapper(
    #     main_menu, supported_leagues
    # )
    # print(f"You selected: {selected_league}")

    # # Ask for an integer input
    # print(f"You entered: {selected_season}")

    # print("Starting web scraper...")
    for league in supported_leagues:
        for season in [1819, 1920, 2021, 2122, 2223, 2324]:

            if league == "FRA-Ligue 1" and season != 2324:
                continue
            else:
                HistoricData(season_id=season, league_id=league)


if __name__ == "__main__":
    main()
