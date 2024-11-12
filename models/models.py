import torch
import torch.nn as nn
from torch import nn, Tensor
import torch.nn.functional as F

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from scipy.stats import poisson


class ZIPoisson:
    def __init__(
        self, training_data: pd.DataFrame, odds_data: pd.DataFrame, lookback: int
    ):
        self.training_data = training_data
        self.odds = odds_data
        self.lookback = lookback

        self.home_cols = [
            f"last_{self.lookback}_home_points",
            f"last_{self.lookback}_home_np_xg",
            f"last_{self.lookback}_home_vaep",
            f"last_{self.lookback}_home_ppda",
            f"last_{self.lookback}_home_min_allocation",
            f"last_{self.lookback}_home_player_rating",
            f"last_{self.lookback}_away_points",
            f"last_{self.lookback}_away_np_xg_conceded",
            f"last_{self.lookback}_away_vaep_conceded",
            f"last_{self.lookback}_away_ppda",
            f"last_{self.lookback}_away_player_rating",
            "home_home_np_xg",
            "home_home_vaep",
            "home_home_ppda",
            "home_home_player_rating",
            "away_away_np_xg_conceded",
            "away_away_vaep_conceded",
            "away_away_ppda",
            "away_away_player_rating",
            "home_elo_np_xg_lookback",
            "home_elo_vaep_lookback",
            "away_elo_np_xg_conceded_season",
            "away_elo_vaep_conceded_lookback",
        ]

        self.away_cols = [
            f"last_{self.lookback}_away_points",
            f"last_{self.lookback}_away_np_xg",
            f"last_{self.lookback}_away_vaep",
            f"last_{self.lookback}_away_ppda",
            f"last_{self.lookback}_away_min_allocation",
            f"last_{self.lookback}_away_player_rating",
            f"last_{self.lookback}_home_points",
            f"last_{self.lookback}_home_np_xg_conceded",
            f"last_{self.lookback}_home_vaep_conceded",
            f"last_{self.lookback}_home_ppda",
            f"last_{self.lookback}_home_player_rating",
            "away_away_np_xg",
            "away_away_vaep",
            "away_away_ppda",
            "away_away_player_rating",
            "home_home_np_xg_conceded",
            "home_home_vaep_conceded",
            "home_home_ppda",
            "home_home_player_rating",
            "away_elo_np_xg_lookback",
            "away_elo_vaep_lookback",
            "home_elo_np_xg_conceded_season",
            "home_elo_vaep_conceded_lookback",
        ]

        self.model_cols = [
            col.replace("_away", "")
            .replace("home_", "ven_")
            .replace("away_", "")
            .replace("_home", "ven_")
            for col in self.home_cols
        ]

        self.training_data = self.training_data.merge(
            self.odds[["game", "FTHG", "FTAG"]], how="inner", on="game"
        )

        print("Collecting data for Zero goals model")
        self._get_zero_data()
        self.zero_model = LogisticRegression()
        self.zero_model.fit(
            self.zero_training.drop("target", axis=1), self.zero_training.target
        )

        print("Collecting data for lambda goal model")
        self._get_lambda_data()
        lambda_x = sm.add_constant(self.lambda_training.drop("target", axis=1))
        self.lambda_model = sm.GLM(
            self.lambda_training.target, lambda_x, family=sm.families.Poisson()
        ).fit()

    def predict(self, matches_to_predict: pd.DataFrame):

        if "FTHG" not in matches_to_predict.columns:
            matches_to_predict = matches_to_predict.merge(
                self.odds[["game", "FTHG", "FTAG"]], how="inner", on="game"
            )

        def get_vector(game):
            home_vector = game[self.home_cols]
            home_vector["home"] = 1

            away_vector = game[self.away_cols]
            away_vector["home"] = 0

            return pd.DataFrame(
                [home_vector.values, away_vector.values],
                columns=self.model_cols + ["home"],
            )

        matches_to_predict[["home_goal_rate", "away_goal_rate"]] = (
            matches_to_predict.apply(
                lambda x: self.lambda_model.predict(
                    sm.add_constant(get_vector(x), has_constant="add")
                ),
                axis=1,
            )
        )
        matches_to_predict[["home_zero_rate", "away_zero_rate"]] = (
            matches_to_predict.apply(
                lambda x: pd.Series(
                    self.zero_model.predict_proba(get_vector(x))[:, 1:].flatten()
                ),
                axis=1,
            )
        )

        matches_to_predict["goal_matrix"] = matches_to_predict.apply(
            lambda x: self._get_goal_matrix(
                x.home_goal_rate, x.home_zero_rate, x.away_goal_rate, x.away_zero_rate
            ),
            axis=1,
        )

        return matches_to_predict

    def _zip_pmf(self, k, p_zero, poisson_lambda):
        """Compute the probability of observing `k` goals in a ZIP distribution."""
        if k == 0:
            return p_zero + (1 - p_zero) * poisson.pmf(0, poisson_lambda)
        else:
            return (1 - p_zero) * poisson.pmf(k, poisson_lambda)

    def _get_goal_matrix(
        self, lambda_home, zero_rate_home, lambda_away, zero_rate_away
    ):
        max_g = 7
        goal_values = np.arange(max_g + 1)

        poisson_home = np.zeros((len(goal_values), 1))
        poisson_away = np.zeros((1, len(goal_values)))
        poisson_array_list = []

        for goal in goal_values:
            poisson_home[goal, 0] = self._zip_pmf(
                goal, zero_rate_home, lambda_home
            ).item()
            poisson_away[0, goal] = self._zip_pmf(
                goal, zero_rate_away, lambda_away
            ).item()
        poisson_array = np.matmul(poisson_home, poisson_away)
        poisson_array_list.append(poisson_array)

        return poisson_array_list[0]

    def _get_zero_data(self):
        zero_log_df = []

        for i, row in self.training_data.iterrows():

            home_goals = row.FTHG
            away_goals = row.FTAG

            home_vector = row[self.home_cols]
            home_vector["home"] = 1
            home_vector["target"] = 0 if home_goals > 0 else 1

            away_vector = row[self.away_cols]
            away_vector["home"] = 0
            away_vector["target"] = 0 if away_goals > 0 else 1

            zero_log_df.append(home_vector.values)
            zero_log_df.append(away_vector.values)

        self.zero_training = pd.DataFrame(
            zero_log_df, columns=self.model_cols + ["home", "target"]
        )

    def _get_lambda_data(self):

        goal_rate_train_list = []

        for i, row in self.training_data.iterrows():

            home_goals = row.FTHG
            away_goals = row.FTAG

            home_vector = row[self.home_cols]
            home_vector["home"] = 1
            home_vector["target"] = home_goals

            away_vector = row[self.away_cols]
            away_vector["home"] = 0
            away_vector["target"] = away_goals

            if home_goals != 0:
                goal_rate_train_list.append(home_vector.values)
            if away_goals != 0:
                goal_rate_train_list.append(away_vector.values)

        self.lambda_training = pd.DataFrame(
            goal_rate_train_list, columns=self.model_cols + ["home", "target"]
        )


class PlayerCNN(torch.nn.Module):
    def __init__(
        self,
        player_dim: int,
        feature_dim: int,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=(1, feature_dim), bias=True
        )

        self.fc1 = nn.Linear(player_dim * 2, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 3)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor):
        home_team = x[:, 0].unsqueeze(1)
        away_team = x[:, 1].unsqueeze(1)

        home_features = F.tanh(self.conv(home_team)).squeeze(3)
        away_features = F.tanh(self.conv(away_team)).squeeze(3)

        home_features = home_features.view(home_features.size(0), -1)
        away_features = away_features.view(away_features.size(0), -1)

        combined_features = torch.cat((home_features, away_features), dim=1)
        x = F.tanh(self.fc1(combined_features))
        x = self.dropout(x)
        x = F.tanh(self.fc2(x))
        x = self.dropout(x)
        logits = self.fc3(x)
        return logits
