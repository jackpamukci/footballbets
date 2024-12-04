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
import boto3
from dotenv import load_dotenv
from geopy.distance import geodesic
import Levenshtein
from fuzzywuzzy import fuzz, process
from sklearn.preprocessing import MinMaxScaler


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


def _get_s3_agent(env_path):
    try:
        load_dotenv(env_path)
        aws_access_key = os.getenv("AWS_ACCESS_KEY")
        aws_secret_access = os.getenv("AWS_SECRET_ACCESS")
        aws_region = os.getenv("AWS_REGION")

        s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_access,
            region_name=aws_region,
        )

        return s3
    except Exception as e:
        raise ConnectionError("Connection to AWS Failed. Check Credentials.") from e

def _fix_positions(features_df):
    position_rep = {
        "DEF": ["DC", "DR", "DL",  "DMC", "DMR", "DML"],
        "MID": ["AMC", "MC", "MR", "ML"],
        "FOR": ["FW", "FWR", "FWL", "AMR", "AML"],
        "GK": ["GK"],
        "SUB": ["Sub"],
    }

    pos = {}

    for position, to_replace_list in position_rep.items():
        for to_replace in to_replace_list:
            pos[to_replace] = position

    features_df.position = features_df.position.apply(
        lambda x: x.replace(x, pos[x])
    )

    return features_df


def _normalize_features(
    feats,
    use_diff: bool = False,
    normalize: bool = False,
    lookback: int = 4,
    feat_group: list = ["last_cols", "momentum", "venue", "general", "elo"],
):

    elo_metrics = [
        "elo_np_xg_season",
        "elo_np_xg_lookback",
        "elo_np_xg_conceded_season",
        "elo_np_xg_conceded_lookback",
        "elo_vaep_season",
        "elo_vaep_lookback",
        "elo_vaep_conceded_season",
        "elo_vaep_conceded_lookback",
        "elo_ppda_lookback",
        "elo_ppda_season",
        "elo_gen",
    ]

    config_cols = [
        "league",
        "season",
        "game",
        "date",
        "home_team",
        "away_team",
        "target",
        "matchday",
        "lookback",
    ]

    last_cols = [
        f"last_{lookback}_{x}_{y}"
        for x in ["home", "away"]
        for y in [
            "points",
            "np_xg",
            "np_xg_conceded",
            "vaep",
            "vaep_conceded",
            "ppda",
            "min_allocation",
            "player_rating",
        ]
    ]

    momentum = [
        f"last_{lookback}_{x}_{y}_{z}"
        for x in ["home", "away"]
        for y in ["np_xg", "np_xg_conceded", "vaep", "vaep_conceded"]
        for z in ["slope", "predicted"]
    ]

    venue = [
        f"{x}_{x}_{y}"
        for x in ["home", "away"]
        for y in [
            "np_xg",
            "np_xg_conceded",
            "vaep",
            "vaep_conceded",
            "ppda",
            "player_rating",
        ]
    ]
    general = [f"{x}_{y}" for x in ["home", "away"] for y in ["rest", "tot_points"]]

    col_list = [
        item
        for sublist in [x for x in [last_cols, momentum, venue, general]]
        for item in sublist
    ]

    elo = [f"{x}_{y}" for x in ["home", "away"] for y in elo_metrics]
    config = feats[config_cols]

    cols_dict = {
        "last_cols": last_cols,
        "momentum": momentum,
        "venue": venue,
        "general": general,
        "elo": elo,
    }

    # cols = self.feat_group
    if use_diff:
        last_cols_diff = pd.DataFrame(
            feats[[x for x in last_cols if x.split("_")[2] == "home"]].values
            - feats[[x for x in last_cols if x.split("_")[2] == "away"]].values,
            columns=[
                f"venue_diff_{x}"
                for x in [
                    "_".join(x.split("_")[2:])
                    for x in last_cols
                    if x.split("_")[2] == "home"
                ]
            ],
        )

        momentum_diff = pd.DataFrame(
            feats[[x for x in momentum if x.split("_")[2] == "home"]].values
            - feats[[x for x in momentum if x.split("_")[2] == "away"]].values,
            columns=[
                f"venue_diff_{x}"
                for x in [
                    "_".join(x.split("_")[2:])
                    for x in momentum
                    if x.split("_")[2] == "home"
                ]
            ],
        )

        venue_diff = pd.DataFrame(
            feats[[x for x in venue if x.split("_")[0] == "home"]].values
            - feats[[x for x in venue if x.split("_")[0] == "away"]].values,
            columns=[
                f"venue_diff_{x}"
                for x in [
                    "_".join(x.split("_")[2:])
                    for x in venue
                    if x.split("_")[0] == "home"
                ]
            ],
        )

        general_diff = pd.DataFrame(
            feats[[x for x in general if x.split("_")[0] == "home"]].values
            - feats[[x for x in general if x.split("_")[0] == "away"]].values,
            columns=[
                f"general_diff_{x}"
                for x in [
                    "_".join(x.split("_")[2:])
                    for x in general
                    if x.split("_")[0] == "home"
                ]
            ],
        )

        elo_diff = pd.DataFrame(
            feats[[x for x in elo if x.split("_")[0] == "home"]].values
            - feats[[x for x in elo if x.split("_")[0] == "away"]].values,
            columns=[
                f"elo_diff_{x}"
                for x in [
                    "_".join(x.split("_")[2:]) for x in elo if x.split("_")[0] == "home"
                ]
            ],
        )

        diff_dict = {
            "last_cols": last_cols_diff,
            "momentum": momentum_diff,
            "venue": venue_diff,
            "general": general_diff,
            "elo": elo_diff,
        }

        diff_data = pd.concat([diff_dict[x] for x in feat_group], axis=1)

        met_col_list = diff_data.columns
        features = pd.concat([config["matchday"], diff_data], axis=1)

        # MANDATORY NORMALIZING OF ELO
        features = (
                features.groupby("matchday", group_keys=False)
                .apply(lambda group: _scale_group(group, elo_diff.columns, use_diff))
            )

        if normalize:

            features = (
                features.groupby("matchday", group_keys=False)
                .apply(lambda group: _scale_group(group, met_col_list, use_diff))
                .drop("matchday", axis=1)
            )

        return pd.concat([config, features], axis=1)

    met_col_list = [item for x in feat_group for item in cols_dict[x]]

    # MANDATORY NORMALIZING OF ELO
    features = (
            features.groupby("matchday", group_keys=False)
            .apply(lambda group: _scale_group(group, elo, use_diff))
        )

    if normalize:
        features = feats[met_col_list + ["matchday"]]

        features = (
            features.groupby("matchday", group_keys=False)
            .apply(lambda group: _scale_group(group, met_col_list, use_diff))
            .drop("matchday", axis=1)
        )

        return pd.concat([config, features], axis=1)

    return pd.concat([config, feats[met_col_list]], axis=1)


def _scale_group(group, met_col_list, diff):

    scaler = MinMaxScaler(feature_range=(-1, 1)) if diff else MinMaxScaler()

    group[met_col_list] = scaler.fit_transform(group[met_col_list])
    return group


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
