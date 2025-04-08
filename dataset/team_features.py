from data.season import Season
import pandas as pd
import numpy as np
from tqdm import tqdm
import statistics
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from data.utils import _fix_positions


class TeamFeatures:

    def __init__(
        self,
        season_data: Season,
        lookback: int = 3,
        k_rate: int = 20,
        use_diff: bool = False,
        use_dist: bool = False,
        normalize: bool = True,
        feat_group: list = ["last_cols", "player_rating", "momentum", "venue", "general", "elo"],
    ):

        self.met_col_list = None
        self.elo_col_list = None

        self.season = season_data
        self.lookback = lookback
        self.k_rate = k_rate
        self.metrics = metrics
        self.elo_metrics = elo_metrics
        self.use_dist = use_dist
        self.use_diff = use_diff
        self.normalize = normalize
        self.feat_group = feat_group

        self.metrics_calc = False

        features_df = self.season.team_stats.copy()

        if 'DEF' not in self.season.player_ratings.position.unique():
            self.season.player_ratings = _fix_positions(self.season.player_ratings)
                
        features_df = self._process_features(features_df)
        # self.features = self._calculate_lineup_strength(features_df, player_ratings)
        self.features = features_df

    def _process_features(self, features_df):

        # sanity check for missing data
        fixture_list = self.season.events.fixture.unique()
        features_df = features_df[features_df.game.isin(fixture_list)]

        features_df = self._get_season_points(features_df)
        features_df = self._get_vaep_shots_target(features_df)
        features_df = self._get_min_allocation(features_df)
        features_df = self._get_average_rating(features_df)
        features_df = self._calculate_metric_features(features_df)

        if self.metrics_calc:
            features_df = self._calculate_elo(features_df, self.k_rate)

        features_df = self._get_proper_cols(features_df)
        features_df = self._normalize_features(features_df)

        return features_df

    def _normalize_features(self, feats):

        last_cols = [
            f"last_{self.lookback}_{x}_{y}"
            for x in ["home", "away"]
            for y in [
                "points",
                "np_xg",
                "np_xg_conceded",
                "vaep",
                "vaep_conceded",
                "ppda",
                "min_allocation",
            ]
        ]

        player_rat = [
            f"last_{self.lookback}_{x}_{y}"
            for x in ["home", "away"]
            for y in [
                "pr_gen",
                "pr_gk",
                "pr_def",
                "pr_mid",
                "pr_fw"
            ]
        ]

        momentum = [
            f"last_{self.lookback}_{x}_{y}_{z}"
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


        last_config = [x for x in config_cols]
        if self.use_dist:
            last_config = [x for x in config_cols] + ['distance']

        config = feats[last_config]

        cols_dict = {
            "last_cols": last_cols,
            "momentum": momentum,
            "venue": venue,
            "general": general,
            "player_rating":player_rat,
            "elo": elo,
        }

        # cols = self.feat_group
        if self.use_diff:
            last_cols_diff = pd.DataFrame(
                feats[[x for x in last_cols if x.split("_")[2] == "home"]].values
                - feats[[x for x in last_cols if x.split("_")[2] == "away"]].values,
                columns=[
                    f"lookback_diff_{x}"
                    for x in [
                        "_".join(x.split("_")[2:])
                        for x in last_cols
                        if x.split("_")[2] == "home"
                    ]
                ],
            )

            player_rat_diff = pd.DataFrame(
                feats[[x for x in player_rat if x.split("_")[2] == "home"]].values
                - feats[[x for x in player_rat if x.split("_")[2] == "away"]].values,
                columns=[
                    f"rating_diff_{x}"
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
                    f"momentum_diff_{x}"
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
                        "_".join(x.split("_")[2:])
                        for x in elo
                        if x.split("_")[0] == "home"
                    ]
                ],
            )

            diff_dict = {
                "last_cols": last_cols_diff,
                "player_rating": player_rat_diff,
                "momentum": momentum_diff,
                "venue": venue_diff,
                "general": general_diff,
                "elo": elo_diff,
            }

            diff_data = pd.concat([diff_dict[x] for x in self.feat_group], axis=1)

            self.met_col_list = diff_data.columns
            features = pd.concat([config["matchday"], diff_data], axis=1)

            if self.normalize:

                features = (
                    features.groupby("matchday", group_keys=False)
                    .apply(self._scale_group)
                    .drop("matchday", axis=1)
                )

            return pd.concat([config, features], axis=1)

        self.met_col_list = [item for x in self.feat_group for item in cols_dict[x]]

        if self.normalize:
            features = feats[self.met_col_list + ["matchday"]]

            features = features.groupby("matchday", group_keys=False).apply(
                self._scale_group
            ).drop('matchday', axis=1)

            return pd.concat([config, features], axis=1)

        return pd.concat([config, feats[self.met_col_list]], axis=1)

    def _calculate_elo(self, feats, k_rate):
        """
        iterate through the matches.
            for the home team and the away team
                get all the games that team has played thus far
                for each metric
                    if the amount of games played is less than 3
                        set the specific metric elo for that row to 1500
                    else
                        get the previous match elo for  if not null, else 1500
        """
        # Create a cache of games played by each team
        team_games_cache = {}
        for team in feats["home_team"].unique():
            team_games_cache[team] = feats[
                (feats["home_team"] == team) | (feats["away_team"] == team)
            ]

        # Iterate over each row in the DataFrame
        for i, row in tqdm(feats.iterrows(), total=feats.shape[0]):
            for ind in ["home", "away"]:
                team = row[f"{ind}_team"]
                # Get all games the team has played up to the current index
                team_games = team_games_cache[team].loc[:i].iloc[:-1]
                games_played = len(team_games)
                opp_ind = "away" if ind == "home" else "home"

                for metric in self.elo_metrics:
                    column_name = f"{ind}_{metric}"
                    if column_name not in feats.columns:
                        feats[column_name] = 1500

                    if (games_played < 1) or (
                        games_played < 3 and metric.split("_")[-1] == "venue"
                    ):
                        continue

                    else:
                        # If enough games have been played, fetch the old Elo from the last game played
                        prev_matches = (
                            feats[
                                (feats["home_team"] == team)
                                | (feats["away_team"] == team)
                            ]
                            .loc[:i]
                            .iloc[:-1]
                        )
                        last_match = prev_matches.iloc[-1]
                        lm_ind = "home" if last_match.home_team == team else "away"
                        lm_opp_ind = "home" if lm_ind == "away" else "away"
                        old_elo = last_match[f"{lm_ind}_{metric}"]

                        # Determine the core metric based on the type
                        if metric.split("_")[1] == "gen":
                            core_metric = "gen"
                        elif metric.split("_")[2] == "xg":
                            core_metric = "np_xg"
                        elif metric.split("_")[1] == "vaep":
                            core_metric = "vaep"
                        elif metric.split("_")[1] == "ppda":
                            core_metric = "ppda"

                        # Determine expected and actual metrics
                        if metric.split("_")[-1] == "season":
                            if core_metric == "ppda":
                                actual_metric_name = f"{lm_ind}_{core_metric}"
                                expected_metric_name = (
                                    f"{lm_ind}_{lm_ind}_{core_metric}"
                                )
                            else:
                                if metric.split("_")[-2] == "conceded":
                                    expected_metric_name = (
                                        f"{lm_ind}_{lm_ind}_{core_metric}_conceded"
                                    )
                                    actual_metric_name = f"{lm_opp_ind}_{core_metric}"
                                else:
                                    expected_metric_name = f"{lm_opp_ind}_{lm_opp_ind}_{core_metric}_conceded"
                                    actual_metric_name = f"{lm_ind}_{core_metric}"

                        elif metric.split("_")[-1] == "lookback":
                            if core_metric == "ppda":
                                actual_metric_name = f"{lm_ind}_{core_metric}"
                                expected_metric_name = (
                                    f"last_{self.lookback}_{lm_ind}_{core_metric}"
                                )
                            else:
                                if metric.split("_")[-2] == "conceded":
                                    expected_metric_name = f"last_{self.lookback}_{lm_ind}_{core_metric}_conceded"
                                    actual_metric_name = f"{lm_opp_ind}_{core_metric}"
                                else:
                                    expected_metric_name = f"last_{self.lookback}_{lm_opp_ind}_{core_metric}_conceded"
                                    actual_metric_name = f"{lm_ind}_{core_metric}"

                        elif metric.split("_")[1] == "gen":
                            # opp_team = row[f"{opp_ind}_team"]

                            if metric.split("_")[-1] == "venue":
                                prev_ven_matches = prev_matches[
                                    prev_matches[f"{ind}_team"] == team
                                ].iloc[-1]
                                old_elo = prev_ven_matches[f"{ind}_{metric}"]

                                # opp_team = prev_ven_matches[f"{opp_ind}_team"]

                                # opp_lm = (
                                #     feats[(feats[f"{opp_ind}_team"] == opp_team)]
                                #     .loc[:i]
                                #     .iloc[:-1]
                                #     .iloc[-1]
                                # )
                                opp_old_elo = prev_ven_matches[f"{opp_ind}_{metric}"]

                                actual = (
                                    1
                                    if (
                                        prev_ven_matches["home_team"] == team
                                        and prev_ven_matches.target == 1
                                    )
                                    else (
                                        0
                                        if (
                                            prev_ven_matches[f"away_team"] == team
                                            and prev_ven_matches.target == 1
                                        )
                                        else 0.5
                                    )
                                )

                            else:

                                old_elo = last_match[f"{lm_ind}_{metric}"]
                                opp_old_elo = last_match[f"{lm_opp_ind}_{metric}"]

                                actual = (
                                    1
                                    if (
                                        last_match["home_team"] == team
                                        and last_match.target == 1
                                    )
                                    else (
                                        0
                                        if (
                                            last_match[f"away_team"] == team
                                            and last_match.target == 1
                                        )
                                        else 0.5
                                    )
                                )

                            # home field advantage
                            if ind == "home":
                                old_elo += 50
                            else:
                                opp_old_elo += 50

                            expected = 1 / (1 + 10 ** ((opp_old_elo - old_elo) / 400))

                            feats.at[i, f"{ind}_{metric}"] = old_elo + (
                                k_rate * (actual - expected)
                            )
                            continue

                        # Retrieve the expected and actual values
                        expected = last_match[expected_metric_name]
                        actual = last_match[actual_metric_name]

                        # Update the Elo rating based on the actual vs expected values
                        feats.at[i, f"{ind}_{metric}"] = old_elo + (
                            k_rate * (actual - expected)
                        )

        return feats

    def _calculate_metric_features(self, feats):
        new_ratings = self._get_player_ratings(self.season.player_ratings)
        player_groups = new_ratings.groupby('game')

        team_games_cache = {}
        for team in feats["home_team"].unique():
            team_games_cache[team] = feats[
                (feats["home_team"] == team) | (feats["away_team"] == team)
            ]

        # adding lookback marker
        feats["lookback"] = 0

        for i, row in tqdm(feats.iterrows(), total=feats.shape[0]):
            player_group = player_groups.get_group(row.game)

            for ind in ["home", "away"]:
                team = row[f"{ind}_team"]
                team_games = team_games_cache[team].loc[:i].iloc[:-1]
                games_played = len(team_games)

                # if in initial period, have lookback = 1
                if games_played < self.lookback:
                    feats.at[i, "lookback"] = 1

                if games_played <= 1:
                    continue

                for metric in self.metrics:
                    if (
                        metric.split("_")[0] in ["home", "away"]
                        and metric.split("_")[0] != ind
                    ):
                        continue

                    if metric.split("_")[0] == ind:
                        # Handle home/away specific metrics
                        ven_games = team_games[team_games[f"{ind}_team"] == team]

                        if len(ven_games) < 1:
                            continue
                        feats.at[i, f"{ind}_{metric}"] = (
                            self._calculate_team_performance(ven_games, metric, ind)
                        )
                    else:
                        # Handle general last 3 performances
                        lookback_matches = team_games.tail(self.lookback)
                        if metric.split("_")[-1] in ["slope", "predicted"]:
                            value = self._calculate_slope_metrics(
                                lookback_matches, metric, row, ind
                            )
                        # TODO
                        elif metric == 'player_rating':
                            
                            rating_feats = self._calculate_player_rating(lookback_matches, player_group, row, ind)
                            rating_cols = [f"last_{self.lookback}_{ind}_{x}" for x in ["pr_gen", "pr_gk", "pr_def", "pr_mid", "pr_fw"]]
                            feats.loc[i, rating_cols] = rating_feats
                        else:
                            value = self._calculate_last_performances(
                                lookback_matches, metric, row, ind
                            )
                        feats.at[i, f"last_{self.lookback}_{ind}_{metric}"] = value

        self.metrics_calc = True
        return feats.fillna(0)

    def _get_proper_cols(self, feats):

        cols = (
            ["game", "home_rest", "away_rest", "distance", "matchday"]
            if self.use_dist == True
            else ["game", "home_rest", "away_rest", "matchday"]
        )
        feats = feats.merge(
            self.season.schedule[cols],
            right_on="game",
            left_on="game",
            how="inner",
        )
        return feats.drop(cols_to_drop, axis=1)

    def _get_min_allocation(self, feats):
        lineups = self.season.player_stats
        grouped = lineups.groupby("game")

        for i, row in feats.iterrows():
            fixture = row.game
            lineups = grouped.get_group(fixture)

            home_lineups = (
                lineups[lineups["team"] == row.home_team]
                .sort_values("minutes", ascending=False)[:14]
                .minutes
            )
            away_lineups = (
                lineups[lineups["team"] == row.away_team]
                .sort_values("minutes", ascending=False)[:14]
                .minutes
            )

            home_minute_allocation = sum([(90 - x) ** 2 for x in home_lineups])
            away_minute_allocation = sum([(90 - x) ** 2 for x in away_lineups])

            feats.at[i, "home_min_allocation"] = home_minute_allocation
            feats.at[i, "away_min_allocation"] = away_minute_allocation

        return feats

    def _calculate_slope_metrics(self, lookback_matches, metric, row, ind):
        points = []
        first_date = datetime.strptime(
            lookback_matches.iloc[0].date, "%Y-%m-%d %H:%M:%S"
        )

        for _, match_row in lookback_matches.iterrows():
            date_diff = (
                datetime.strptime(match_row.date, "%Y-%m-%d %H:%M:%S") - first_date
            )
            x_val = abs(date_diff.days)

            if metric.split("_")[-2] == "conceded":
                indicator = (
                    "home" if row[f"{ind}_team"] == match_row.away_team else "away"
                )
                metric_name = "_".join(metric.split("_")[:-2])
            else:
                indicator = (
                    "home" if row[f"{ind}_team"] == match_row.home_team else "away"
                )
                metric_name = "_".join(metric.split("_")[:-1])
            y_val = match_row[f"{indicator}_{metric_name}"]

            points.append((x_val, y_val))

        x_values = np.array([x for x, y in points])
        y_values = np.array([y for x, y in points])
        slope, intercept = np.polyfit(x_values, y_values, 1)

        if metric.endswith("slope"):
            return slope
        else:
            day_diff = datetime.strptime(row.date, "%Y-%m-%d %H:%M:%S") - first_date
            x = abs(day_diff.days)
            return slope * x + intercept

    def _calculate_team_performance(self, ven_games, metric, ind):
        if metric.endswith("conceded"):

            indicator = "away" if ind == "home" else "home"
            metric_name = "_".join(metric.split("_")[1:-1])
            metric_perf = ven_games[f"{indicator}_{metric_name}"]

        else:
            metric_perf = ven_games[f'{ind}_{metric.split("_", 1)[1]}']

        return statistics.mean(metric_perf)

    def _calculate_last_performances(self, lookback_matches, metric, row, ind):
        metric_perf = []
        for _, match_row in lookback_matches.iterrows():
            if metric.endswith("conceded"):
                indicator = (
                    "home" if row[f"{ind}_team"] == match_row.away_team else "away"
                )
                metric_name = "_".join(metric.split("_")[:-1])
                metric_perf.append(match_row[f"{indicator}_{metric_name}"])
            else:
                indicator = (
                    "home" if row[f"{ind}_team"] == match_row.home_team else "away"
                )
                metric_perf.append(match_row[f"{indicator}_{metric}"])

        return statistics.mean(metric_perf)

    # TODO: define calculate player line up method to get not only lookback player rating but also lineup strengths
    def _calculate_player_rating(self, lookback_matches, match_players, row, ind):

        # for i, row in features.iterrows():
        # starting11 = match_players[match_players['position'] != 'SUB']
        # team_players = starting11[starting11['h_a'] == ind]

        denom_ratings = self.season.player_ratings[(self.season.player_ratings['game'].isin(lookback_matches.game.unique())) & (self.season.player_ratings['team'] == row[f'{ind}_team']) & (self.season.player_ratings['position'] != 'SUB')]

        gen_denom = denom_ratings.groupby('game').rating.mean().mean()
        gk_denom = denom_ratings[denom_ratings['position'] == 'GK'].groupby('game').rating.mean().mean()
        def_denom = denom_ratings[denom_ratings['position'] == 'DEF'].groupby('game').rating.mean().mean()
        mid_denom = denom_ratings[denom_ratings['position'] == 'MID'].groupby('game').rating.mean().mean()
        fw_denom = denom_ratings[denom_ratings['position'] == 'FOR'].groupby('game').rating.mean().mean()

        # denominator = ratings_by_match.rating.mean().max()
        num_ratings = self.season.player_ratings[(self.season.player_ratings['game'] == row.game) & (self.season.player_ratings['team'] == row[f'{ind}_team']) & (self.season.player_ratings['position'] != 'SUB')]

        gen_num = num_ratings.lookback_rating.mean()
        gk_num = num_ratings[num_ratings['position'] == 'GK'].lookback_rating.mean()
        def_num = num_ratings[num_ratings['position'] == 'DEF'].lookback_rating.mean()
        mid_num = num_ratings[num_ratings['position'] == 'MID'].lookback_rating.mean()
        fw_num = num_ratings[num_ratings['position'] == 'FOR'].lookback_rating.mean()

        # gk_max = []
        # def_max = []
        # mid_max = []
        # fw_max = []

        # print('match to predict: ', row.game)

        # for i, lb_row in lookback_matches.iterrows():
        #     indicator = (
        #         "home" if row[f"{ind}_team"] == lb_row.home_team else "away"
        #     )

        #     # print(f'lookback {i}: ', lb_row.game)

        #     gen_max.append(lb_row[f"{indicator}_player_rating"])
        #     # gk_max.append(lb_row[f"{indicator}_gk_player_rating"])
        #     # def_max.append(lb_row[f"{indicator}_def_player_rating"])
        #     # mid_max.append(lb_row[f"{indicator}_mid_player_rating"])
        #     # fw_max.append(lb_row[f"{indicator}_for_player_rating"])

        # gen_rating = numerator / denominator
        return [gen_num / gen_denom, gk_num / gk_denom, 
                def_num / def_denom, mid_num / mid_denom, 
                fw_num / fw_denom]
    ''',
        team_players[team_players['position'] == 'GK'].rating.mean() / max(gk_max),
        team_players[team_players['position'] == 'DEF'].rating.mean() / max(def_max),
        team_players[team_players['position'] == 'MID'].rating.mean() / max(mid_max),
        team_players[team_players['position'] == 'FOR'].rating.mean() / max(fw_max)]'''

    def _get_vaep_shots_target(self, feats):
        match_events = self.season.events.groupby("fixture")

        home_vaep = []
        away_vaep = []
        home_shots = []
        away_shots = []
        targets = []

        for i, row in tqdm(feats.iterrows(), total=feats.shape[0]):
            games_to_drop = []
            fixture = row.game
            try:
                events = match_events.get_group(fixture)
            except:
                games_to_drop.append(i)
                home_vaep.append(0)
                away_vaep.append(0)
                home_shots.append(0)
                away_shots.append(0)
                targets.append(0)
                continue
            events = events.sort_values(["period_id", "time_seconds"], ascending=True)
            events["ha"] = np.where(
                events.team_id == events.home_team_id, "home", "away"
            )

            vaep_sum = events.groupby("ha")["vaep_value"].sum()
            shots_count = events.groupby("ha")["type_name"].apply(
                lambda x: (x == "shot").sum()
            )

            home_vaep.append(vaep_sum.get("home", 0))
            away_vaep.append(vaep_sum.get("away", 0))
            home_shots.append(shots_count.get("home", 0))
            away_shots.append(shots_count.get("away", 0))
            targets.append(
                1 if row.home_points == 3 else (0 if row.home_points == 1 else 2)
            )

        feats["home_vaep"] = home_vaep
        feats["away_vaep"] = away_vaep
        feats["home_shots"] = home_shots
        feats["away_shots"] = away_shots
        feats["target"] = targets

        return feats.drop(games_to_drop).reset_index(drop=True)

    def _get_average_rating(self, feats):
        player_ratings = self.season.player_ratings
        # print(player_ratings.columns)
        player_ratings["h_a"] = player_ratings.apply(
            lambda x: "home" if x.team == x.game[11:].split("-")[0] else "away", axis=1
        )
        ratings_groups = player_ratings.groupby("game")

        for i, row in feats.iterrows():
            ratings = ratings_groups.get_group(row.game)

            for ind in ['home', 'away']:
                team_ratings = ratings[ratings["h_a"] == ind]

                feats.at[i, f"{ind}_player_rating"] = team_ratings.rating.mean()

                feats.at[i, f"{ind}_gk_player_rating"] = team_ratings[team_ratings['position'] == 'GK'].rating.mean()
                feats.at[i, f"{ind}_def_player_rating"] = team_ratings[team_ratings['position'] == 'DEF'].rating.mean()
                feats.at[i, f"{ind}_mid_player_rating"] = team_ratings[team_ratings['position'] == 'MID'].rating.mean()
                feats.at[i, f"{ind}_for_player_rating"] = team_ratings[team_ratings['position'] == 'FOR'].rating.mean()

        return feats
    
    def _get_season_points(self, schedule):
        schedule = schedule.copy()
        teams = schedule[schedule["league"] == self.season.league_id].home_team.unique()

        schedule["home_tot_points"] = np.nan
        schedule["away_tot_points"] = np.nan

        for team in teams:

            tot_points = 0
            team_sched = schedule[
                (schedule["home_team"] == team) | (schedule["away_team"] == team)
            ].sort_values("date", ascending=True)

            for i, fixture in team_sched.iterrows():
                h_a = "home" if fixture.home_team == team else "away"
                col = "home_tot_points" if h_a == "home" else "away_tot_points"

                schedule.loc[i, col] = tot_points

                if h_a == "home":
                    tot_points += fixture.home_points
                else:
                    tot_points += fixture.away_points

        return schedule

    def _scale_group(self, group):
        scaler = MinMaxScaler()
        group[self.met_col_list] = scaler.fit_transform(group[self.met_col_list])
        return group
            
    def _get_player_ratings(self, features):
        features.dropna(subset=["player"], inplace=True)
        features.fillna(0, inplace=True)

        match_fixtures = features.groupby("game")
        player_performances = features.groupby("player")

        for fixture, data in tqdm(match_fixtures):

            for i, row in data.iterrows():

                player_perf = player_performances.get_group(row.player)
                position = player_perf.index.get_loc(i)
                lookback_table = player_perf.iloc[max(0, position - self.lookback) : position]

                # assume that a player who hasn't made an appearance yet will perform below average
                # or esssentially 5.75 
                if len(lookback_table) < 1:
                    lookback_rating = 5.75
                else:
                    lookback_rating = lookback_table.rating.mean()

                # features.at[i, "season_rating"] = season_rating
                features.at[i, "lookback_rating"] = lookback_rating
                features.at[i, "lookback_len"] = len(lookback_table)

        return features



cols_to_drop = [
    "home_team_id",
    "away_team_id",
    "home_team_code",
    "away_team_code",
    "home_deep_completions",
    "away_deep_completions",
    "home_expected_points",
    "away_expected_points",
    "away_xg",
    "home_xg",
    "away_np_xg_difference",
    "home_np_xg_difference",
    "away_points",
    "home_points",
    "league_id",
    "season_id",
    "game_id",
    "away_goals",
    "home_goals",
    "home_np_xg",
    "home_ppda",
    "home_vaep",
    "away_vaep",
    "home_shots",
    "away_shots",
    "away_np_xg",
    "away_ppda",
    "home_min_allocation",
    "away_min_allocation",
    "home_player_rating",
    "home_gk_player_rating",
    "home_def_player_rating",
    "home_mid_player_rating",
    "home_for_player_rating",
    "away_player_rating",
    "away_gk_player_rating",
    "away_def_player_rating",
    "away_mid_player_rating",
    "away_for_player_rating",
]

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

metrics = [
    "np_xg",
    "np_xg_conceded",
    "home_np_xg",
    "home_np_xg_conceded",
    "away_np_xg",
    "away_np_xg_conceded",
    "np_xg_slope",
    "np_xg_conceded_slope",
    "np_xg_predicted",
    "np_xg_conceded_predicted",
    "vaep",
    "vaep_conceded",
    "home_vaep",
    "home_vaep_conceded",
    "away_vaep",
    "away_vaep_conceded",
    "vaep_slope",
    "vaep_conceded_slope",
    "vaep_predicted",
    "vaep_conceded_predicted",
    "ppda",
    "home_ppda",
    "away_ppda",
    "points",
    "min_allocation",
    "player_rating",
    "home_player_rating",
    "away_player_rating",
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


# venue_diff_cols = [
#     f"venue_diff_{x}"
#     for x in ["_".join(x.split("_")[2:]) for x in venue if x.split("_")[0] == "home"]
# ]
# general_diff_cols = [
#     f"gen_diff_{x}"
#     for x in ["_".join(x.split("_")[1:]) for x in general if x.split("_")[0] == "home"]
# ]
