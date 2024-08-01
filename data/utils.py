import pandas as pd
from socceraction import spadl
from tqdm import tqdm
import xgboost
import socceraction.vaep.features as fs
import socceraction.vaep.labels as lab
import socceraction.vaep.formula as vaepformula


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
