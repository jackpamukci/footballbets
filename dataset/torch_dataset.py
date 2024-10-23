from dataset.team_features import TeamFeatures
from dataset.player_features import PlayerFeatures
from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd


class ModelDataset(Dataset):
    def __init__(
        self,
        player_features: PlayerFeatures,
        device: torch.device = torch.device("cpu"),
    ):
        super(ModelDataset, self).__init__()
        self.features = player_features.features
        self.schedule = player_features.season.schedule
        self.lookback = player_features.lookback_matches
        self.player_config_cols = [
            "league",
            "season",
            "game",
            "team",
            "player",
            "h_a",
            "lookback",
            "matchday",
        ]

        self.features = self.features[~self.features["game"].isin(self.lookback)]

        player_feats = []
        player_config = []
        fixture = []
        season = []
        league = []

        groups = self.features.groupby("game")
        for game, group in groups:

            home_team = group[group["h_a"] == "home"].sort_values(
                "lookback_minutes", ascending=False
            )
            away_team = group[group["h_a"] == "away"].sort_values(
                "lookback_minutes", ascending=False
            )

            home_feats = (
                home_team.drop(self.player_config_cols, axis=1)
                .iloc[:14]
                .values.astype(float)
            )
            home_config = home_team[self.player_config_cols].iloc[:14]

            away_feats = (
                away_team.drop(self.player_config_cols, axis=1)
                .iloc[:14]
                .values.astype(float)
            )
            away_config = away_team[self.player_config_cols].iloc[:14]

            max_rows = max(home_feats.shape[0], away_feats.shape[0])
            max_cols = max(home_feats.shape[1], away_feats.shape[1])
            home_feats_padded = np.pad(
                home_feats,
                (
                    (0, max_rows - home_feats.shape[0]),
                    (0, max_cols - home_feats.shape[1]),
                ),
                mode="constant",
                constant_values=0,
            )
            away_feats_padded = np.pad(
                away_feats,
                (
                    (0, max_rows - away_feats.shape[0]),
                    (0, max_cols - away_feats.shape[1]),
                ),
                mode="constant",
                constant_values=0,
            )

            stacked_array = np.stack([home_feats_padded, away_feats_padded], axis=0)
            player_tensor = torch.from_numpy(stacked_array)

            player_config_df = pd.concat([home_config, away_config])
            sched_match = self.schedule[self.schedule["game"] == game].iloc[0]

            player_feats.append(player_tensor)
            player_config.append(player_config_df)
            fixture.append(game)
            season.append(sched_match.season)
            league.append(sched_match.league)

        self.data = {
            "player_features": player_feats,
            "player_configs": player_config,
            "fixtures": fixture,
            "seasons": season,
            "leagues": league,
        }

    def __len__(self):
        return len(self.data["player_features"])

    def __getitem__(self, idx: int):
        return (
            self.data["player_features"][idx],
            self.data["player_configs"][idx],
            self.data["fixtures"][idx],
            self.data["seasons"][idx],
            self.data["leagues"][idx],
        )
