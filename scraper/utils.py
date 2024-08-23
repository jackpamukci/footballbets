import os
import sys
import pandas as pd
import requests
import io
from datetime import datetime
import numpy as np

current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)
import _config


def get_european_schedule(season):
    TEAMNAME_REPLACEMENTS = _config.TEAMNAME_REPLACEMENTS
    SCHEDULE_COLUMNS = _config.SCHEDULE_COLUMNS

    season_id = f"20{str(season)[:2]}"
    europe = pd.DataFrame()

    for league in ["europa", "champions"]:

        if season_id in ["2017", "2018"] and league == "europa":
            continue

        headers = {
            "Cookies": "_ga_DTCKHDGKYF=GS1.1.1722868866.6.1.1722869089.0.0.0; _ga=GA1.2.1274569263.1721488882; ARRAffinity=3587c3b28f299ba120e848a3ba122803c40823fd58ac197de099244cf70e116d; ARRAffinitySameSite=3587c3b28f299ba120e848a3ba122803c40823fd58ac197de099244cf70e116d; _gid=GA1.2.1211098860.1722868867; Timezone=Eastern Standard Time",
            "Referer": f"https://fixturedownload.com/download/csv/{league}-league-{season_id}",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0",
        }

        csv = requests.get(
            f"https://fixturedownload.com/download/{league}-league-{season_id}-EasternStandardTime.csv",
            headers=headers,
        )
        temp = pd.read_csv(io.StringIO(csv.text))
        temp["league"] = "Europa League" if league == "europa" else "Champions League"
        europe = pd.concat([europe, temp])

    team_cols = ["Home Team", "Away Team"]
    europe[team_cols] = europe[team_cols].replace(TEAMNAME_REPLACEMENTS)

    europe["date"] = europe.Date.apply(lambda x: str(x).split(" ")[0])
    europe["date"] = europe.date.apply(
        lambda x: datetime.strptime(x, "%d/%m/%Y").date()
    )
    europe["time"] = europe.Date.apply(lambda x: str(x).split(" ")[1])
    europe["time"] = europe.time.apply(lambda x: datetime.strptime(x, "%H:%M").time())

    europe["season"] = season

    europe["game"] = europe.apply(
        lambda x: f"{x.date} {x['Home Team']}-{x['Away Team']}", axis=1
    )
    europe["start_time"] = europe.apply(lambda x: f"{x.date}T{x.time}", axis=1)
    europe = europe.rename(columns={"Home Team": "home_team", "Away Team": "away_team"})

    cols_to_keep = ["league", "season", "game", "start_time", "home_team", "away_team"]
    nul_cols = SCHEDULE_COLUMNS.difference(cols_to_keep)

    europe = europe.drop(europe.columns.difference(cols_to_keep), axis=1)
    europe[list(nul_cols)] = np.nan

    return europe
