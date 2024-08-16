from data.season import Season
import pandas as pd
import numpy as np


class TeamFeatures:
    def __init__(self, season_data: Season):

        self.season = season_data

        self.season.schedule = self._get_rest_days()

        features_df = self.season.team_stats.copy()

        self.features = features_df

    def _get_vaep_dict(self, fixture, events):
        sample_match = events[events["fixture"] == fixture].sort_values(
            ["period_id", "time_seconds"], ascending=True
        )
        # sample_match["h.a"] = sample_match.apply(
        #     lambda x: "home" if x.team_id == x.home_team_id else "away", axis=1
        # )
        return dict(sample_match.groupby("h.a")["vaep_value"].sum())

    def get_game_shots(fixture, events):
        sample_match = events[events["fixture"] == fixture].sort_values(
            ["period_id", "time_seconds"], ascending=True
        )
        # sample_match["h.a"] = sample_match.apply(
        #     lambda x: "home" if x.team_id == x.home_team_id else "away", axis=1
        # )
        return dict(
            sample_match.groupby("h.a")["type_name"].apply(
                lambda x: (x == "shot").sum()
            )
        )

    def _get_rest_days(self):
        schedule = self.season.schedule
        league = self.season.league_id

        # For filtering out european fixtures from schedule
        teams = schedule[schedule["league"] == league].home_team.unique()

        schedule["home_rest"] = np.nan
        schedule["away_rest"] = np.nan

        for team in teams:
            # Filter the schedule for the specific team (either home or away)
            team_sched = schedule[
                (schedule["home_team"] == team) | (schedule["away_team"] == team)
            ].sort_values("start_time", ascending=True)
            team_sched["previous_match_start"] = team_sched.shift(
                1, fill_value=team_sched.iloc[0].start_time
            )["start_time"]

            # Calculate the previous match start time
            schedule.loc[team_sched.index, "previous_match_start"] = team_sched[
                "start_time"
            ].shift(1, fill_value=team_sched.iloc[0].start_time)

            # Convert start_time to start_date and previous_match_start to previous_match_date
            schedule.loc[team_sched.index, "start_date"] = team_sched[
                "start_time"
            ].apply(lambda x: pd.to_datetime(str(x).split("T")[0]))
            schedule.loc[team_sched.index, "previous_match_date"] = team_sched[
                "previous_match_start"
            ].apply(lambda x: pd.to_datetime(str(x).split("T")[0]))

            home_condition = team_sched["home_team"] == team
            schedule.loc[team_sched.index[home_condition], "home_rest"] = (
                schedule.loc[team_sched.index[home_condition], "start_date"]
                - schedule.loc[team_sched.index[home_condition], "previous_match_date"]
            ).dt.days

            # Calculate away rest days where the team is the away team
            away_condition = team_sched["away_team"] == team
            schedule.loc[team_sched.index[away_condition], "away_rest"] = (
                schedule.loc[team_sched.index[away_condition], "start_date"]
                - schedule.loc[team_sched.index[away_condition], "previous_match_date"]
            ).dt.days

        return schedule.reset_index(drop=True).sort_values("start_time", ascending=True)
