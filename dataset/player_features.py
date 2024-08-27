import pandas as pd
from data.season import Season
from unidecode import unidecode
from data.utils import best_name_match
from tqdm import tqdm


class PlayerFeatures:

    def __init__(self, season_data: Season):
        self.season = season_data
        self.events = self.season.events

        features_df = self.season.player_stats.copy()

        self.features = self._preprocess_features(features_df)

    def _preprocess_features(self, features_df):
        return self._match_player_names(features_df)

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

        return features_df
