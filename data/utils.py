import pandas as pd
from socceraction import spadl
from tqdm import tqdm
import xgboost
import socceraction.vaep.features as fs
import socceraction.vaep.labels as lab
import socceraction.vaep.formula as vaepformula
import sys
import os
import numpy as np
from geopy.distance import geodesic
import Levenshtein
from fuzzywuzzy import fuzz, process


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
    sample = sample.copy()  # To avoid modifying the original dataframe

    # Shift operations
    next_event = sample.shift(-1, fill_value=0)
    sample["nextEvent"] = next_event["type_name"]
    sample["nextTeamId"] = next_event["team_id"]

    # Determine 'kickedOut' column
    sample["kickedOut"] = sample["nextEvent"].isin(["throw_in", "goalkick"]).astype(int)

    # Convert columns to numpy arrays for faster access
    type_name = sample["type_name"].values
    result_name = sample["result_name"].values
    team_id = sample["team_id"].values
    next_event_type = sample["nextEvent"].values
    next_team_id = sample["nextTeamId"].values
    period_id = sample["period_id"].values
    kicked_out = sample["kickedOut"].values

    # Initialize arrays
    possession_chain = np.zeros(len(sample), dtype=int)
    possession_chain_team = np.zeros(len(sample), dtype=int)

    chain_team = team_id[0]
    period = period_id[0]

    stop_criterion = 0
    chain = 0

    for i in range(len(sample)):
        possession_chain[i] = chain
        possession_chain_team[i] = chain_team

        if type_name[i] in ["pass", "duel", "take_on", "dribble"]:
            if result_name[i] == "fail":
                if next_event_type[i] in ["interception", "tackle"]:
                    stop_criterion += 2
                else:
                    stop_criterion += 1

            if team_id[i] != next_team_id[i]:
                if next_event_type[i] in ["dribble", "pass", "tackle"]:
                    stop_criterion += 2

        if type_name[i] == "bad_touch" and team_id[i] != next_team_id[i]:
            stop_criterion += 2

        if (
            type_name[i] in ["pass", "cross", "freekick_crossed", "corner_crossed"]
            and result_name[i] == "offside"
        ):
            stop_criterion += 2
        if type_name[i] in ["shot", "foul", "clearance"]:
            stop_criterion += 2
        if kicked_out[i] == 1:
            stop_criterion += 2

        if period_id[i] != period:
            chain += 1
            stop_criterion = 0
            chain_team = team_id[i]
            period = period_id[i]
            possession_chain[i] = chain
            possession_chain_team[i] = chain_team

        if stop_criterion >= 2:
            chain += 1
            stop_criterion = 0
            chain_team = next_team_id[i]

    sample["possession_chain"] = possession_chain
    sample["possession_chain_team"] = possession_chain_team

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


def get_distances(schedule):
    STADIUM_LOCATIONS = _config.STADIUM_LOCATIONS
    schedule["distance"] = schedule.apply(
        lambda x: geodesic(
            tuple(STADIUM_LOCATIONS[x.home_team]), tuple(STADIUM_LOCATIONS[x.away_team])
        ).kilometers,
        axis=1,
    )

    return schedule


def levenshtein_similarity(s1, s2):
    distance = Levenshtein.distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return 1 - (distance / max_len)


def best_name_match(target, strings):
    best_score = -1
    best_string = None
    for s in strings:
        score = levenshtein_similarity(target, s)
        if score > best_score:
            best_score = score
            best_string = s
    return best_string


def fuzzy_match(query, choices):
    best_match, score = process.extractOne(query, choices, scorer=fuzz.token_set_ratio)
    return best_match


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
