from dataset.team_features import TeamFeatures
from dataset.player_features import PlayerFeatures
from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd


class PlayerDataset(Dataset):
    def __init__(
        self,
        player_features: pd.DataFrame,
        schedule: pd.DataFrame,
        device: torch.device = torch.device("cpu"),
    ):
        super(PlayerDataset, self).__init__()
        self.features = player_features
        self.schedule = schedule

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

        self.features = self.features[self.features["lookback"] != 1]

        player_feats = []
        targets = []
        configs = []

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

            # max_rows = max(home_feats.shape[0], away_feats.shape[0])
            # max_cols = max(home_feats.shape[1], away_feats.shape[1])
            if 14 - home_feats.shape[0] < 0 or 31 - home_feats.shape[1] < 0:
                print(game)
                print(14 - home_feats.shape[0])
                print(31 - home_feats.shape[1])

            home_feats_padded = np.pad(
                home_feats,
                (
                    (0, 14 - home_feats.shape[0]),
                    (0, 31 - home_feats.shape[1]),
                ),
                mode="constant",
                constant_values=0,
            )
            away_feats_padded = np.pad(
                away_feats,
                (
                    (0, 14 - away_feats.shape[0]),
                    (0, 31 - away_feats.shape[1]),
                ),
                mode="constant",
                constant_values=0,
            )

            stacked_array = np.stack([home_feats_padded, away_feats_padded], axis=0)
            player_tensor = torch.tensor(stacked_array, dtype=torch.float)

            player_config_df = pd.concat([home_config, away_config])
            sched_match = self.schedule[self.schedule["game"] == game].iloc[0]

            target = (
                0
                if sched_match.home_score == sched_match.away_score
                else (1 if sched_match.home_score > sched_match.away_score else 0)
            )

            player_feats.append(player_tensor)

            target = (
                0
                if sched_match.home_score == sched_match.away_score
                else (1 if sched_match.home_score > sched_match.away_score else 2)
            )

            targets.append(target)

            configs.append([
                sched_match.league,
                sched_match.season,
                sched_match.date,
                sched_match.home_team,
                sched_match.away_team,
                sched_match.game
            ])

        self.data = {
            "player_features": player_feats,
            "targets": targets,
            "config": configs
        }

    def __len__(self):
        return len(self.data["player_features"])

    def __getitem__(self, idx: int):
        return (
            self.data["player_features"][idx],
            self.data["targets"][idx],
            self.data["config"][idx],
        )


class HybridDataset(Dataset):
    def __init__(
        self,
        team_features: pd.DataFrame,
        player_features: pd.DataFrame,
        schedule: pd.DataFrame,
        device: torch.device = torch.device("cpu"),
    ):
        super(HybridDataset, self).__init__()
        self.player_features = player_features.features
        self.team_features = team_features.features
        self.schedule = schedule

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

        self.team_config_cols = [
            "season",
            "game",
            "date",
            "home_team",
            "away_team",
            "target",
            "matchday",
            "lookback",
        ]

        self.team_features = self.team_features[self.team_features["lookback"] != 1]
        self.player_features = self.player_features[
            self.player_features["lookback"] != 1
        ]

        ## TODO: assert that all the games in player dataset, event dataset and schedule match
        assert set(self.player_features.game.unique()) == set(
            self.team_features.game.unique()
        ), "Missing games from team and event features."

        player_feats = []
        team_feats = []
        targets = []
        player_config = []
        targets = []
        config = []

        groups = self.player_features.groupby("game")
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

            # max_rows = max(home_feats.shape[0], away_feats.shape[0])
            # max_cols = max(home_feats.shape[1], away_feats.shape[1])

            # if 14 - home_feats.shape[0] < 0 or 30 - home_feats.shape[1] < 0:
            print(game)
            print(14 - home_feats.shape[0])
            print(31 - home_feats.shape[1])
            
            home_feats_padded = np.pad(
                home_feats,
                (
                    (0, 14 - home_feats.shape[0]),
                    (0, 31 - home_feats.shape[1]),
                ),
                mode="constant",
                constant_values=0,
            )
            away_feats_padded = np.pad(
                away_feats,
                (
                    (0, 14 - away_feats.shape[0]),
                    (0, 31 - away_feats.shape[1]),
                ),
                mode="constant",
                constant_values=0,
            )

            stacked_array = np.stack([home_feats_padded, away_feats_padded], axis=0)
            player_tensor = torch.tensor(stacked_array, dtype=torch.float)

            player_config_df = pd.concat([home_config, away_config])
            team_vector = (
                self.team_features[self.team_features["game"] == game]
                .drop(self.team_config_cols, axis=1)
                .iloc[0]
            )
            sched_match = self.schedule[self.schedule["game"] == game].iloc[0]

            target = (
                0
                if sched_match.home_score == sched_match.away_score
                else (1 if sched_match.home_score > sched_match.away_score else 0)
            )

            player_feats.append(player_tensor)
            team_feats.append(torch.tensor(team_vector, dtype=torch.float))

            target = (
                0
                if sched_match.home_score == sched_match.away_score
                else (1 if sched_match.home_score > sched_match.away_score else 2)
            )

            targets.append(target)
            player_config.append(player_config_df)
            config.append(
                {
                    "fixture": game,
                    "season": sched_match.season,
                    "league": sched_match.league,
                }
            )

        self.data = {
            "player_features": player_feats,
            "team_features": team_feats,
            "targets": targets,
            "config": config,
        }

    def __len__(self):
        return len(self.data["player_features"])

    def __getitem__(self, idx: int):
        return (
            self.data["player_features"][idx],
            self.data["team_features"][idx],
            self.data["targets"][idx],
            # self.data["player_info"][idx],
            self.data["config"][idx],
        )
