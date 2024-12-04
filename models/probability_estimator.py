import pandas as pd
from typing import Any, Union, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss
from models.models import PlayerCNN, ZIPoisson
from dataset.torch_dataset import PlayerDataset
from data import utils
from tqdm import tqdm

from data.utils import _normalize_features

from itertools import chain
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from models.utils import train_routine
from models.models import PlayerCNN
import numpy as np
from venn_abers import VennAbersCalibrator


class ProbabilityEstimator:
    def __init__(
        self,
        bookie_odds: pd.DataFrame = None,
        testing_data: pd.DataFrame = None,
        use_diff: bool = True,
        normalize: bool = False,
        lookback: int = 6,
        bookie: str = "pinnacle",
        feature_selection: str = "none",
        markets_to_play: list = ["1x2"],
        model_type: str = "class",
        model: Union[LogisticRegression, PlayerCNN] = None,
        training_data: pd.DataFrame = None,
        env_path: str = None,
    ):

        self.features = testing_data
        self.odds = bookie_odds
        self.markets = markets_to_play
        self.model_type = model_type
        self.model = model
        self.training_data = training_data
        self.env_path = env_path
        self.s3 = utils._get_s3_agent(self.env_path)

        if all(market not in ["1x2", "OU", "BTTS"] for market in self.markets):
            raise ValueError("Only available markets are '1x2', 'OU', and 'BTTS'")

        if self.model_type not in ["class", "zip", "player"]:
            raise ValueError("Only available models are 'class', 'zip', and 'player'")

        if feature_selection not in ["lasso", "forward", "none"]:
             raise ValueError("Only available feature selection methods are 'lasso', 'forward', and 'none'")

        self.feature_selection = feature_selection
        self.use_diff = use_diff
        self.normalize = normalize
        self.lookback = lookback

        self.team_feature_cols = [
            "venue_diff_home_points",
            "venue_diff_np_xg_conceded",
            "venue_diff_vaep_conceded",
            "venue_diff_ppda",
            "general_diff_",
            "general_diff_points",
            "elo_diff_np_xg_season",
            "elo_diff_np_xg_lookback",
            "elo_diff_np_xg_conceded_lookback",
            "elo_diff_vaep_season",
            "elo_diff_vaep_lookback",
            "elo_diff_ppda_lookback",
            "elo_diff_gen",
        ]

        self.config_cols = [
            "season",
            "game",
            "date",
            "home_team",
            "away_team",
            "matchday",
            "target",
        ]

        if self.odds is None:
            self._get_odds()

        bookie_cols = {
            "pinnacle": ["PSCD", "PSCH", "PSCA"],
            "b365": ["B365D", "B365H", "B365A"],
            "avg": ["AvgCD", "AvgCH", "AvgCA"],
            "max": ["MaxCD", "MaxCH", "MaxCA"],
        }

        self.bookie = bookie
        self.bet_df_cols = bookie_cols[self.bookie]

        if self.training_data is None:

            if self.model_type != "player":
                self._get_team_features()

                if self.model is None and self.model_type == "class":
                    self._train_team_model()
                elif self.model is None and self.model_type == "zip":
                    self.model = ZIPoisson(self.training_data, self.odds, self.lookback)

            if self.model_type == "player":
                self._get_player_features()
                if self.model is None:
                    self._train_player_model()

        self._get_pred_odds()
        self._extend_bets()

    def _get_pred_odds(self):

        bet_df_cols = [
            "league",
            "game",
        ] + self.bet_df_cols

        if self.model_type == "class":

            # TODO: errors on wrong columns
            data_to_predict = self.features[self.selected_features]
            # print(self.feature_selection)
            # print(set(data_to_predict.columns).difference(self.selected_features))

            config = self.features[self.config_cols].reset_index(drop=True)
            predictions = self.model.predict_proba(data_to_predict)
            pred_df = pd.DataFrame(
                predictions, columns=["draw_prob", "home_prob", "away_prob"]
            )
            pred_df = config.merge(
                pred_df, how="inner", left_index=True, right_index=True
            )
        elif self.model_type == "zip":
            data_to_predict = self.features

            config = data_to_predict[self.config_cols]

            data_to_predict = self.model.predict(data_to_predict)
            data_to_predict[["draw_prob", "home_prob", "away_prob"]] = (
                data_to_predict.goal_matrix.apply(lambda x: self._zip_matrix_parser(x))
            )

            config[["draw_prob", "home_prob", "away_prob"]] = data_to_predict[
                ["draw_prob", "home_prob", "away_prob"]
            ].values
            pred_df = config
            print(pred_df.shape)

        elif self.model_type == "player":
            player_preds = []
            match_configs = []
            targets = []

            test_loader = DataLoader(self.features, batch_size=32, shuffle=True)

            # print('getting preds')
            with torch.no_grad():
                for test_batch in test_loader:
                    test_inputs, test_labels, match_config = test_batch

                    test_outputs = self.model(test_inputs)
                    test_probabilities = F.softmax(test_outputs, dim=1)
                    targets.extend(test_labels.tolist())

                    player_preds.append(np.array(test_probabilities))

                    match_configs.append(match_config)

            result = [
                list(chain.from_iterable(inner_lists))
                for inner_lists in zip(*match_configs)
            ]

            config_df = pd.DataFrame(
                {
                    "league": result[0],
                    "season": result[1][0].item(),
                    "date": result[2],
                    "home_team": result[3],
                    "away_team": result[4],
                    "game": result[5],
                }
            )

            preds = pd.DataFrame(
                np.concatenate(player_preds, axis=0),
                columns=["draw_prob", "home_prob", "away_prob"],
            )
            pred_df = pd.concat([config_df, preds], axis=1)
            pred_df["target"] = pd.Series(targets)

            bet_df_cols = [col for col in bet_df_cols if col != "league"]

        pred_odds = pred_df.merge(
            self.odds[bet_df_cols].rename(
                columns={
                    self.bet_df_cols[0]: "draw_odds",
                    self.bet_df_cols[1]: "home_odds",
                    self.bet_df_cols[2]: "away_odds",
                }
            ),
            how="left",
            on="game",
        )

        self.pred_odds = pred_odds

    def _extend_bets(self):
        # Define columns and markets
        # markets = self.markets
        config_cols = ["league", "season", "date", "home_team", "away_team"]
        draw_cols = ["draw_prob", "draw_odds"]
        home_cols = ["home_prob", "home_odds"]
        away_cols = ["away_prob", "away_odds"]

        # Initialize list to store all bets
        all_bets_list = []

        # Iterate over each row in pred_odds
        for i, row in tqdm(self.pred_odds.iterrows(), total=len(self.pred_odds)):
            bets = []

            for market in self.markets:
                if market == "1x2":

                    total_odds = (
                        (1 / row.draw_odds) + (1 / row.home_odds) + (1 / row.away_odds)
                    )
                    # Prepare each bet row
                    draw = row[config_cols + draw_cols].to_dict()
                    draw["result"] = row.target == 0
                    draw["market"] = "1x2_draw"
                    draw["probability"] = draw.pop(draw_cols[0])
                    draw["odds"] = draw.pop(draw_cols[1])
                    draw["imp_odds"] = (1 / draw["odds"]) / total_odds
                    bets.append(draw)

                    home = row[config_cols + home_cols].to_dict()
                    home["result"] = row.target == 1
                    home["market"] = "1x2_home"
                    home["probability"] = home.pop(home_cols[0])
                    home["odds"] = home.pop(home_cols[1])
                    home["imp_odds"] = (1 / home["odds"]) / total_odds
                    bets.append(home)

                    away = row[config_cols + away_cols].to_dict()
                    away["result"] = row.target == 2
                    away["market"] = "1x2_away"
                    away["probability"] = away.pop(away_cols[0])
                    away["odds"] = away.pop(away_cols[1])
                    away["imp_odds"] = (1 / away["odds"]) / total_odds
                    bets.append(away)

            all_bets_list.extend(bets)

        # Convert list to DataFrame once at the end
        self.bets = pd.DataFrame(all_bets_list)

    def _get_team_features(self):

        key = f"season_pickles/team_feats_{self.lookback}.csv"

        mastercsv = self.s3.get_object(
            Bucket="footballbets",
            Key=key,
        )

        master_df = pd.read_csv(mastercsv["Body"])
        master_df.matchday.ffill(inplace=True)

        master_df = _normalize_features(
            master_df,
            self.use_diff,
            self.normalize,
            self.lookback,
            ["last_cols", "venue", "general", "momentum", "elo"],
        )

        self.training_data = master_df[
            (master_df["season"] != 2324) & (master_df["lookback"] != 1)
        ]

        if self.features is None:
            self.features = master_df[
                (master_df["season"] == 2324) & (master_df["lookback"] != 1)
            ]

    def _get_player_features(self):

        ### TARGET IS ASSIGNED AS 1x2 in schedule!!!

        player_feats = self.s3.get_object(
            Bucket="footballbets",
            Key=f"season_pickles/player_feats_{self.lookback}.csv",
        )
        master_player_feats = pd.read_csv(player_feats["Body"])
        master_player_feats = master_player_feats[master_player_feats["lookback"] != 1]

        s3schedule = self.s3.get_object(
            Bucket="footballbets", Key=f"season_pickles/master_schedule.csv"
        )
        master_schedule = pd.read_csv(s3schedule["Body"], index_col=0)
        master_schedule["target"] = master_schedule.apply(
            lambda x: (
                0
                if x.home_score == x.away_score
                else (1 if x.home_score > x.away_score else 2)
            ),
            axis=1,
        )

        train_feats = master_player_feats[
            (master_player_feats["season"] != 2324)
            & (master_player_feats["lookback"] != 1)
        ]
        train_schedule = master_schedule[master_schedule["season"] != 2324]

        self.training_data = PlayerDataset(train_feats, train_schedule)

        if self.features is None:
            test_feats = master_player_feats[
                (master_player_feats["season"] == 2324)
                & (master_player_feats["lookback"] != 1)
            ]
            test_schedule = master_schedule[master_schedule["season"] == 2324]
            self.features = PlayerDataset(test_feats, test_schedule)

    def _get_odds(self):
        oddscsv = self.s3.get_object(
            Bucket="footballbets",
            Key=f"season_pickles/masterodds.csv",
        )
        master_odds = pd.read_csv(oddscsv["Body"], index_col=0)

        self.odds = master_odds

    def _train_team_model(self):

        # TODO: get optimal feature set
        X_train = self.training_data.drop(self.config_cols + ["league", "lookback"], axis=1)
        
        y_train = self.training_data.target

        if self.feature_selection == "lasso":
            lasso = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
            param_grid = {'C': np.logspace(-3, 3, 10)}  # C is the inverse of lambda
            grid_search = GridSearchCV(lasso, param_grid, scoring='neg_log_loss', cv=6)

            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            self.selected_features = X_train.columns

        elif self.feature_selection == "forward":

            self.selected_features = ['venue_diff_home_points', 'venue_diff_home_np_xg',
            'venue_diff_home_np_xg_conceded', 'venue_diff_home_vaep',
            'venue_diff_home_vaep_conceded', 'venue_diff_home_min_allocation',
            'venue_diff_np_xg', 'venue_diff_np_xg_conceded', 'general_diff_points',
            'venue_diff_home_np_xg_slope', 'venue_diff_home_np_xg_predicted',
            'venue_diff_home_np_xg_conceded_slope',
            'venue_diff_home_np_xg_conceded_predicted',
            'venue_diff_home_vaep_predicted', 'venue_diff_home_vaep_conceded_slope',
            'elo_diff_np_xg_season', 'elo_diff_vaep_conceded_season']

            X_train_selected = X_train[self.selected_features]

            self.model = LogisticRegression(max_iter=1000)
            self.model.fit(X_train_selected, y_train)


        else:
            self.model = LogisticRegression(max_iter=1000)
            self.model.fit(X_train, y_train)
            self.selected_features = X_train.columns
            

    def _train_player_model(self):

        train_loader = DataLoader(self.training_data, batch_size=32, shuffle=True)

        learning_rate = 0.0001
        model = PlayerCNN(player_dim=14, feature_dim=31)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.model = train_routine(model, train_loader, optimizer, loss_fn, 50)

    def _zip_matrix_parser(self, goal_matrix):

        rows = len(goal_matrix)
        columns = len(goal_matrix[0])
        draw = np.trace(goal_matrix)
        away_win = 0.0
        home_win = 0.0
        over2 = 0.0
        under2 = 0.0
        btts = 0.0

        for away_goals in range(rows):
            for home_goals in range(columns):
                if home_goals > away_goals:
                    home_win += goal_matrix[home_goals, away_goals]
                if away_goals > home_goals:
                    away_win += goal_matrix[home_goals, away_goals]
                if away_goals + home_goals >= 3:
                    over2 += goal_matrix[home_goals, away_goals]
                if away_goals + home_goals < 3:
                    under2 += goal_matrix[home_goals, away_goals]
                if (away_goals != 0) and (home_goals != 0):
                    btts += goal_matrix[home_goals, away_goals]

        return pd.Series([draw, home_win, away_win])
