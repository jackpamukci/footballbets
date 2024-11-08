import pandas as pd
from typing import Any, Union, Optional
from sklearn.linear_model import LogisticRegression
from models.models import PlayerCNN
from dataset.torch_dataset import PlayerDataset
from data import utils
from tqdm import tqdm

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
        use_diff: bool = False,
        normalize: bool = False,
        lookback: int = 4,
        bookie: str = "pinnacle",
        best_set: bool = False,
        markets_to_play: list = ["1x2"],
        model_type: str = "team",
        model: Union[LogisticRegression, PlayerCNN] = None,
        training_data: pd.DataFrame = None,
        env_path: str = None,
    ):

        self.features = testing_data
        self.odds = bookie_odds

        if all(market not in ["1x2", "OU", "BTTS"] for market in markets_to_play):
            raise ValueError("Only available markets are '1x2', 'OU', and 'BTTS'")

        if model_type not in ["team", "player", "hybrid"]:
            raise ValueError(
                "Only available markets are 'team', 'player', and 'hybrid'"
            )

        self.markets = markets_to_play
        self.model_type = model_type
        self.model = model
        self.training_data = training_data
        self.env_path = env_path
        self.s3 = utils._get_s3_agent(self.env_path)

        self.best_set = best_set
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

        if self.training_data is None:

            if model_type == "team":
                self._get_team_features()
                if self.model is None:
                    clf = LogisticRegression(max_iter=10000)
                    self.model = VennAbersCalibrator(clf, inductive=False, n_splits=5)

                    X_train = (
                        self.training_data.drop(self.config_cols, axis=1).iloc[:, 4:]
                        if self.best_set
                        else self.training_data.drop(
                            self.config_cols + ["league", "lookback"], axis=1
                        )
                    )
                    y_train = self.training_data.target

                    print(X_train.isna().sum())
                    self.model.fit(X_train, y_train)

            if model_type == "player":
                self._get_player_features()
                if self.model is None:
                    self._train_player_model()

        if self.odds is None:
            self._get_odds()

        bookie_cols = {
            "pinnacle": ["PSCD", "PSCH", "PSCA"],
            "b365": ["B365CD", "B365CH", "B365CA"],
            "avg": ["AvgCD", "AvgCH", "AvgCA"],
            "max": ["MaxCD", "MaxCH", "MaxCA"],
        }

        self.bookie = bookie
        self.bet_df_cols = bookie_cols[self.bookie]

        self._get_pred_odds()
        self._extend_bets()

    def _get_pred_odds(self):

        bet_df_cols = [
            "league",
            "game",
        ] + self.bet_df_cols

        if self.model_type == "team":
            data_to_predict = (
                self.features.drop(self.config_cols, axis=1).iloc[:, 4:]
                if self.best_set
                else self.features.drop(
                    self.config_cols + ["league", "lookback"], axis=1
                )
            )
            config = self.features[self.config_cols].reset_index(drop=True)
            print(data_to_predict.columns)
            print(data_to_predict.isna().sum())
            predictions = self.model.predict_proba(data_to_predict)
            pred_df = pd.DataFrame(
                predictions, columns=["draw_prob", "home_prob", "away_prob"]
            )
            pred_df = config.merge(
                pred_df, how="inner", left_index=True, right_index=True
            )

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

        # print(pred_odds.columns)

        # pred_odds['matchday'].bfill(inplace=True)

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

        key = (
            "season_pickles/team_features_diff.csv"
            if self.best_set
            else f"season_pickles/team_feats_{self.lookback}.csv"
        )

        mastercsv = self.s3.get_object(
            Bucket="footballbets",
            Key=key,
        )

        master_df = pd.read_csv(mastercsv["Body"])

        if not self.best_set:
            master_df = utils._normalize_features(
                master_df,
                self.use_diff,
                self.normalize,
                self.lookback,
                ["last_cols", "venue", "general", "elo"],
            )

        # print(master_df.columns)

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
            Key=f"season_pickles/player_feats_4.csv",
        )
        master_player_feats = pd.read_csv(player_feats["Body"])
        master_player_feats = master_player_feats[master_player_feats["lookback"] != 1]

        s3schedule = self.s3.get_object(
            Bucket="footballbets", Key=f"season_pickles/team_feats_{self.lookback}.csv"
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

    def _train_player_model(self):

        train_loader = DataLoader(self.training_data, batch_size=32, shuffle=True)

        learning_rate = 0.0001
        model = PlayerCNN(player_dim=14, feature_dim=30)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.model = train_routine(model, train_loader, optimizer, loss_fn, 50)
