import pandas as pd
from socceraction import spadl
from tqdm import tqdm
import xgboost
import socceraction.vaep.features as fs
import socceraction.vaep.labels as lab
import socceraction.vaep.formula as vaepformula
import sys
import os
import requests
import io
import numpy as np
from datetime import datetime

current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)
import _config


def get_vaep_data(spadldf):
    gamestates = fs.gamestates(spadldf, 3)
    xfns = [
        fs.actiontype_onehot,
        fs.result_onehot,
        fs.bodypart_onehot,
        fs.startlocation,
        fs.endlocation,
        fs.startpolar,
        fs.endpolar,
        fs.movement,
        fs.time_delta,
        fs.space_delta,
        fs.goalscore,
        fs.time,
    ]

    yfns = [lab.scores, lab.concedes]

    X = pd.concat([fn(gamestates) for fn in xfns], axis=1)
    y = pd.concat([fn(spadldf) for fn in yfns], axis=1)

    return X, y


def train_vaep(X, y):
    pscores = ProbabityModel(
        model=xgboost.XGBClassifier(
            n_estimators=50, max_depth=3, n_jobs=-3, verbosity=1
        ),
        model_type="classifier",
    )
    pscores.train(X, y[["scores"]])

    pconcedes = ProbabityModel(
        model=xgboost.XGBClassifier(
            n_estimators=50, max_depth=3, n_jobs=-3, verbosity=1
        ),
        model_type="classifier",
    )
    pconcedes.train(X, y[["concedes"]])

    return pscores, pconcedes


def get_vaep_values(spadldf, X, pscores, pconcedes):
    models = {"scores": pscores, "concedes": pconcedes}

    y_hat = pd.DataFrame(columns=["scores", "concedes"])

    for col in ["scores", "concedes"]:
        y_hat[col] = models[col].predict(X)

    return vaepformula.value(spadldf, y_hat["scores"], y_hat["concedes"])


def get_vaep(spadldf):
    print("Calculating Features for VAEP")
    X, y = get_vaep_data(spadldf)
    print("Training Model for VAEP")
    pscores, pconcedes = train_vaep(X, y)
    return get_vaep_values(spadldf, X, pscores, pconcedes)


def get_match_possessions(sample):
    next_event = sample.shift(-1, fill_value=0)
    sample["nextEvent"] = next_event["type_name"]
    sample["kickedOut"] = sample.apply(
        lambda x: 1 if x["nextEvent"] in ["throw_in", "goalkick"] else 0, axis=1
    )

    sample["nextTeamId"] = next_event["team_id"]
    chain_team = sample.iloc[0]["team_id"]
    period = sample.iloc[0]["period_id"]

    stop_criterion = 0
    chain = 0
    sample["possession_chain"] = 0
    sample["possession_chain_team"] = 0

    for i, row in sample.iterrows():
        sample.at[i, "possession_chain"] = chain
        sample.at[i, "possession_chain_team"] = chain_team

        if row.type_name in ["pass", "duel", "take_on", "dribble"]:

            if row["result_name"] == "fail":
                if row.nextEvent == "interception" or row.nextEvent == "tackle":
                    stop_criterion += 2
                else:
                    stop_criterion += 1

            if row.team_id != row.nextTeamId:
                if (
                    row.nextEvent == "dribble"
                    or row.nextEvent == "pass"
                    or row.nextEvent == "tackle"
                ):
                    stop_criterion += 2

        if row.type_name == "bad_touch" and row.team_id != row.nextTeamId:
            stop_criterion += 2

        if row.type_name in ["pass", "cross", "freekick_crossed", "corner_crossed"]:
            if row.result_name == "offside":
                stop_criterion += 2
        if row.type_name in ["shot", "foul", "clearance"]:
            stop_criterion += 2
        if row["kickedOut"] == 1:
            stop_criterion += 2

        if row["period_id"] != period:
            chain += 1
            stop_criterion = 0
            chain_team = row["team_id"]
            period = row["period_id"]
            sample.at[i, "possession_chain"] = chain
            sample.at[i, "possession_chain_team"] = chain_team

        if stop_criterion >= 2:
            chain += 1
            stop_criterion = 0
            chain_team = row["nextTeamId"]

    return sample


def get_season_possessions(spadldf):
    spadl_df = []
    possession_chain_counter = 0

    game_ids = spadldf.game_id.unique()
    for id in tqdm(game_ids):
        match_events = spadldf[spadldf["game_id"] == id]
        match_home_id = match_events.home_team_id.iloc[0]

        match_events = get_match_possessions(match_events)
        match_events["possession_chain"] += possession_chain_counter
        possession_chain_counter = match_events["possession_chain"].max() + 1

        match_events = spadl.play_left_to_right(match_events, match_home_id)

        spadl_df.append(match_events)

    return pd.concat(spadl_df).reset_index(drop=True)


def get_european_schedule(season):
    TEAMNAME_REPLACEMENTS = _config.TEAMNAME_REPLACEMENTS
    SCHEDULE_COLUMNS = _config.SCHEDULE_COLUMNS

    season_id = f"20{str(season)[:2]}"
    europe = pd.DataFrame()

    for league in ["europa", "champions"]:

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

    europe["season"] = 2223

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


class ProbabityModel:
    def __init__(
        self,
        model=xgboost.XGBClassifier(
            n_estimators=50, max_depth=3, n_jobs=-3, verbosity=1
        ),
        model_type="classifier",
    ):
        self.model = model
        self.model_type = model_type
        self.trained = False

    def train(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
    ):

        self.model.fit(X.values, y.values)
        self.trained = True

    def predict(
        self,
        X: pd.DataFrame,
    ):

        if not self.trained:
            raise ValueError("Model not trained")
        if self.model_type == "classifier":
            return [p[1] for p in self.model.predict_proba(X.values)]
        elif self.model_type == "regressor":
            return self.model.predict(X.values)
        else:
            raise ValueError("Model type not supported")
