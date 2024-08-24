from data.season import Season
import pandas as pd
import numpy as np
from tqdm import tqdm
import statistics
from datetime import datetime


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
    "league",
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
]

config_cols = ["season", "game", "date", "home_team", "away_team"]


class TeamFeatures:
    def __init__(
        self,
        season_data: Season,
        lookback: int = 3,
        metrics: list = [
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
        ],
    ):

        self.season = season_data
        self.lookback = lookback
        self.metrics = metrics
        self.use_dist = season_data.get_dist

        features_df = self.season.team_stats.copy()
        features_df = self._get_season_points(features_df)
        features_df = self._get_vaep_shots_target(features_df)
        features_df = self._get_min_allocation(features_df)
        features_df = self._calculate_metric_features(features_df)
        features_df = self._get_proper_cols(features_df)

        self.features = features_df

    def _calculate_metric_features(self, feats):
        team_games_cache = {}
        for team in feats["home_team"].unique():
            team_games_cache[team] = feats[
                (feats["home_team"] == team) | (feats["away_team"] == team)
            ]

        for i, row in tqdm(feats.iterrows(), total=feats.shape[0]):
            for ind in ["home", "away"]:
                team = row[f"{ind}_team"]
                team_games = team_games_cache[team].loc[:i].iloc[:-1]
                games_played = len(team_games)

                if games_played < self.lookback:
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
                        else:
                            value = self._calculate_last_performances(
                                lookback_matches, metric, row, ind
                            )
                        feats.at[i, f"last_{self.lookback}_{ind}_{metric}"] = value

        return feats

    def _get_proper_cols(self, feats):

        cols = (
            ["home_rest", "away_rest", "distance"]
            if self.use_dist == True
            else ["home_rest", "away_rest"]
        )
        feats = feats.merge(
            self.season.schedule[cols],
            right_index=True,
            left_index=True,
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

        return statistics.mean(metric_perf) if len(metric_perf) > 0 else 0

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

        return statistics.mean(metric_perf) if len(metric_perf) > 0 else 0

    def _get_vaep_shots_target(self, feats):
        match_events = self.season.events.groupby("fixture")

        home_vaep = []
        away_vaep = []
        home_shots = []
        away_shots = []
        targets = []

        for i, row in tqdm(feats.iterrows(), total=feats.shape[0]):
            fixture = row.game
            events = match_events.get_group(fixture)
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
                1 if row.home_points == 3 else (0 if row.home_points == 1 else -1)
            )

        feats["home_vaep"] = home_vaep
        feats["away_vaep"] = away_vaep
        feats["home_shots"] = home_shots
        feats["away_shots"] = away_shots
        feats["target"] = targets

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
