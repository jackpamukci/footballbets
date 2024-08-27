import pandas as pd
from data.season import Season
from unidecode import unidecode
from data.utils import best_name_match, fuzzy_match
from tqdm import tqdm
import numpy as np


class PlayerFeatures:

    def __init__(self, season_data: Season):
        self.season = season_data
        self.events = self.season.events

        features_df = self.season.player_stats.copy()
        self.features = self._preprocess_features(features_df)
        self.features = self.features.sort_values(
            ["game", "h_a", "minutes"], ascending=[True, True, False]
        )

    def _preprocess_features(self, features_df):
        features_df = self._match_player_names(features_df)
        features_df = self._get_vaep(features_df)
        features_df["h_a"] = features_df.apply(
            lambda x: "home" if x.team == x.game[11:].split("-")[0] else "away", axis=1
        )
        features_df = self._fix_positions(features_df)
        return features_df

    def _get_vaep(self, features_df):
        # Precompute vaep_sum for all players across all fixtures
        sorted_events = self.events.sort_values(
            ["fixture", "period_id", "time_seconds"], ascending=True
        )
        vaep_sum = (
            sorted_events.groupby(["fixture", "player"])[
                ["vaep_value", "offensive_value", "defensive_value"]
            ]
            .sum()
            .reset_index()
        )

        # Merge the vaep_sum with features_df only once
        results = features_df.merge(
            vaep_sum,
            how="left",
            left_on=["game", "player"],
            right_on=["fixture", "player"],
        )

        # Drop the unnecessary 'fixture' column after the merge
        results = results.drop(columns=["fixture"])

        return results

    def _fix_positions(self, features_df):
        position_rep = {
            "CB": ["DC"],
            "FB": ["DR", "DL"],
            "DM": ["DMC", "DMR", "DML"],
            "CM": ["AMC", "MC", "MR", "ML"],
            "FW": ["FW", "FWR", "FWL", "AMR", "AML"],
            "GK": ["GK"],
            "Sub": ["Sub"],
        }

        pos = {}

        for position, to_replace_list in position_rep.items():
            for to_replace in to_replace_list:
                pos[to_replace] = position

        features_df.position = features_df.position.apply(
            lambda x: x.replace(x, pos[x])
        )
        return features_df

    def _match_player_names(self, features_df):
        features_df.player = features_df.player.apply(lambda x: unidecode(x))
        self.events.player = self.events.player.apply(lambda x: unidecode(x))

        team_match = self.events[["player", "team"]].drop_duplicates(subset="player")
        stat_match = features_df[["player", "team"]].drop_duplicates(subset="player")

        for i, row in tqdm(features_df.iterrows(), total=features_df.shape[0]):
            if (
                row.player
                not in team_match[team_match["team"] == row.team].player.unique()
            ):
                team_players = team_match[
                    (team_match["team"] == row.team)
                ].player.unique()
                stat_players = stat_match[
                    (stat_match["team"] == row.team)
                ].player.unique()
                features_df.at[i, "player"] = best_name_match(
                    row.player, set(team_players).difference(stat_players)
                )

        features_df["player"] = features_df.groupby("player_id")["player"].transform(
            "first"
        )

        duplicate_players = (
            features_df[["player", "player_id", "team"]]
            .drop_duplicates()  # Remove duplicate rows
            .groupby("player")  # Group by 'player'
            .filter(
                lambda x: x["player_id"].nunique() > 1
            )  # Filter groups with more than one unique 'player_id'
        )

        old_dups = self.season.player_stats[
            self.season.player_stats["player_id"].isin(
                list(duplicate_players.player_id)
            )
        ][["player", "player_id", "team"]].drop_duplicates()

        old_dups["new_name"] = old_dups.apply(
            lambda x: fuzzy_match(
                x.player, self.events[self.events["team"] == x.team].player.unique()
            ),
            axis=1,
        )

        for i, row in old_dups.iterrows():
            idx = features_df[features_df["player_id"] == row.player_id].index
            features_df.loc[idx, "player"] = row.new_name

        # print(old_dups.new_name)

        return features_df
