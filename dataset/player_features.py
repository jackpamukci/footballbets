import pandas as pd
from data.season import Season
from data.utils import best_name_match, fuzzy_match
from tqdm import tqdm
import numpy as np
from unidecode import unidecode
import statistics


class PlayerFeatures:

    def __init__(self, season_data: Season, lookback=3):
        self.season = season_data
        self.events = self.season.events
        self.lookback = lookback

        features_df = self.season.player_stats.copy()
        self.features = self._preprocess_features(features_df)
        self.features = self.features.sort_values(
            ["game", "h_a", "minutes"], ascending=[True, True, False]
        ).reset_index(drop=True)

    def _preprocess_features(self, features_df):
        features_df = self._match_player_names(features_df)
        features_df = self._get_vaep(features_df)
        features_df["h_a"] = features_df.apply(
            lambda x: "home" if x.team == x.game[11:].split("-")[0] else "away", axis=1
        )

        # gets ratings from ratings table
        player_ratings = self.season.player_ratings
        features_df = features_df.merge(
            player_ratings[["player", "game", "team", "rating"]],
            how="left",
            on=["game", "team", "player"],
        )
        features_df.rating.fillna(6, inplace=True)

        features_df = self._fix_positions(features_df)
        features_df = self._calculate_features(features_df)

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

    def _calculate_features(self, features):
        features.dropna(subset=["player"], inplace=True)
        features.fillna(0, inplace=True)

        match_fixtures = features.groupby("game")
        player_performances = features.groupby("player")

        for fixture, data in tqdm(match_fixtures):
            home_team = fixture[11:].split("-")[0]
            home_fix = pd.Series(features[features.team == home_team].game.unique())
            home_match = home_fix[home_fix == fixture].index[0]
            lookback_home = home_fix.loc[home_match - self.lookback : home_match - 1]
            home_xg_table = lookback_home.to_frame(name="id").merge(
                self.season.team_stats, how="left", left_on="id", right_on="game"
            )

            away_team = fixture[11:].split("-")[1]
            away_fix = pd.Series(features[features.team == away_team].game.unique())
            away_match = away_fix[away_fix == fixture].index[0]
            lookback_away = away_fix.loc[away_match - self.lookback : away_match - 1]
            away_xg_table = lookback_away.to_frame(name="id").merge(
                self.season.team_stats, how="left", left_on="id", right_on="game"
            )

            home_xg_conceded = (
                statistics.mean(
                    [
                        row.home_np_xg if row.away_team == home_team else row.away_np_xg
                        for index, row in home_xg_table.iterrows()
                    ]
                )
                if len(home_xg_table) > 1
                else np.nan
            )
            away_xg_conceded = (
                statistics.mean(
                    [
                        row.home_np_xg if row.away_team == away_team else row.away_np_xg
                        for index, row in away_xg_table.iterrows()
                    ]
                )
                if len(away_xg_table) > 1
                else np.nan
            )

            for i, row in data.iterrows():
                player_perf = player_performances.get_group(row.player)
                seasonal_table = player_perf.loc[: i - 1]
                position = player_perf.index.get_loc(i)
                lookback_table = player_perf.iloc[position - 3 : position]

                if len(lookback_table) <= 1:
                    if len(seasonal_table) <= 1:
                        cons = 1
                        season_vaep = row.vaep_value
                        season_vaep_per90 = 0
                        season_xg = row.xg
                        season_xg_per90 = 0
                        season_goals = row.goals
                        season_rating = 0

                        lookback_vaep = row.vaep_value
                        lookback_vaep_per90 = 0
                        lookback_xg = row.xg
                        lookback_xg_per90 = 0
                        lookback_minutes = row.minutes
                        lookback_goals = row.goals
                        lookback_conceded = 0
                        lookback_rating = 0
                else:
                    cons = statistics.stdev(seasonal_table.minutes) / statistics.mean(
                        seasonal_table.minutes
                    )
                    season_vaep = statistics.mean(seasonal_table.vaep_value)
                    season_vaep_per90 = (sum(seasonal_table.vaep_value) * 90) / sum(
                        seasonal_table.minutes
                    )
                    season_xg = statistics.mean(seasonal_table.xg)
                    season_xg_per90 = (sum(seasonal_table.xg) * 90) / sum(
                        seasonal_table.minutes
                    )
                    season_goals = sum(seasonal_table.goals)
                    season_rating = statistics.mean(seasonal_table.rating)

                    lookback_vaep = statistics.mean(lookback_table.vaep_value)
                    lookback_vaep_per90 = (sum(lookback_table.vaep_value) * 90) / sum(
                        lookback_table.minutes
                    )
                    lookback_xg = statistics.mean(lookback_table.xg)
                    lookback_xg_per90 = (sum(lookback_table.xg) * 90) / sum(
                        lookback_table.minutes
                    )
                    lookback_goals = sum(lookback_table.goals)
                    lookback_rating = statistics.mean(lookback_table.rating)

                    if row.h_a == "home":
                        def_3 = lookback_home.to_frame(name="id").merge(
                            seasonal_table, how="left", left_on="id", right_on="game"
                        )
                        lookback_conceded = home_xg_conceded
                        def_3.minutes.fillna(0, inplace=True)
                        lookback_minutes = statistics.mean(def_3.minutes)
                    else:
                        def_3 = lookback_away.to_frame(name="id").merge(
                            seasonal_table, how="left", left_on="id", right_on="game"
                        )
                        lookback_conceded = away_xg_conceded
                        def_3.minutes.fillna(0, inplace=True)
                        lookback_minutes = statistics.mean(def_3.minutes)

                features.at[i, "CONS"] = cons
                features.at[i, "season_vaep"] = season_vaep
                features.at[i, "season_vaep_per90"] = season_vaep_per90
                features.at[i, "season_xg"] = season_xg
                features.at[i, "season_xg_per90"] = season_xg_per90
                features.at[i, "season_goals"] = season_goals
                features.at[i, "season_rating"] = season_rating

                features.at[i, "lookback_minutes"] = lookback_minutes
                features.at[i, "lookback_vaep"] = lookback_vaep
                features.at[i, "lookback_vaep_per90"] = lookback_vaep_per90
                features.at[i, "lookback_xg"] = lookback_xg
                features.at[i, "lookback_xg_per90"] = lookback_xg_per90
                features.at[i, "lookback_xg_conceded"] = lookback_conceded
                features.at[i, "lookback_goals"] = lookback_goals
                features.at[i, "lookback_rating"] = lookback_rating

        return features

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
        self.season.player_ratings.player = self.season.player_ratings.player.apply(
            lambda x: unidecode(x)
        )

        # self.season.missing_players.player = self.season.missing_players.player.apply(
        #     lambda x: unidecode(x)
        # )

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

        if old_dups.shape[0] != 0:

            oldies = old_dups.apply(
                lambda x: fuzzy_match(
                    x.player, self.events[self.events["team"] == x.team].player.unique()
                ),
                axis=1,
            )

            old_dups["new_name"] = oldies

            for i, row in old_dups.iterrows():
                idx = features_df[features_df["player_id"] == row.player_id].index
                features_df.loc[idx, "player"] = row.new_name

        return features_df


cols_to_drop = [
    "league_id",
    "season_id",
    "game_id",
    "team_id",
    "player_id",
    "position_id",
    "minutes",
    "goals",
    "own_goals",
    "shots",
    "xg",
    "xg_chain",
    "xg_buildup",
    "assists",
    "xa",
    "key_passes",
    "yellow_cards",
    "red_cards",
    "vaep_value",
    "offensive_value",
    "defensive_value",
]
