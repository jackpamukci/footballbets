from data.season import Season
import pandas as pd
import numpy as np
from tqdm import tqdm


cols_to_drop = ['home_team_id', 'away_team_id', 'home_team_code', 'away_team_code', 'home_deep_completions', 
                'away_deep_completions', 'home_expected_points', 'away_expected_points', 'away_xg', 'home_xg',
                'away_np_xg_difference', 'home_np_xg_difference', 'away_points', 'home_points', 'league', 
                'league_id', 'season_id', 'game_id']

config_cols = ['season', 'game', 'date', 'home_team', 'away_team']


class TeamFeatures:
    def __init__(self, season_data: Season):

        self.season = season_data

        self.season.schedule = self._get_rest_days()

        features_df = self.season.team_stats.copy()
        features_df = self._get_season_points(features_df)
        features_df = self._get_vaep_shots_target(features_df)
        features_df = self._get_proper_cols(features_df)

        self.features = features_df


    def _get_proper_cols(self, feats):
        feats = feats.merge(self.season.schedule[['home_rest', 'away_rest', 'distance']], right_index=True, left_index=True, how='inner')
        return feats.drop(cols_to_drop, axis=1)
    

    def _get_vaep_shots_target(self, feats):
        match_events = self.season.events.groupby('fixture')

        home_vaep = []
        away_vaep = []
        home_shots = []
        away_shots = []
        targets = []

        for i, row in tqdm(feats.iterrows(), total=feats.shape[0]):
            fixture = row.game
            events = match_events.get_group(fixture)
            events = events.sort_values(["period_id", "time_seconds"], ascending=True)
            events["ha"] = np.where(events.team_id == events.home_team_id, "home", "away")
            
            vaep_sum = events.groupby("ha")["vaep_value"].sum()
            shots_count = events.groupby("ha")["type_name"].apply(lambda x: (x == "shot").sum())
            
            home_vaep.append(vaep_sum.get('home', 0))
            away_vaep.append(vaep_sum.get('away', 0))
            home_shots.append(shots_count.get('home', 0))
            away_shots.append(shots_count.get('away', 0))
            targets.append(1 if row.home_points == 3 else (0 if row.home_points == 1 else -1))

        feats['home_vaep'] = home_vaep
        feats['away_vaep'] = away_vaep
        feats['home_shots'] = home_shots
        feats['away_shots'] = away_shots
        feats['target'] = targets

        return feats

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

        schedule = schedule.reset_index(drop=True).sort_values("start_time", ascending=True)
        schedule = schedule[schedule['league'] == self.season.league_id]

        return schedule
    
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
                h_a = 'home' if fixture.home_team == team else 'away'
                col = 'home_tot_points' if h_a == 'home' else 'away_tot_points'

                schedule.loc[i, col] = tot_points

                if h_a == 'home':
                    tot_points += fixture.home_points
                else:
                    tot_points += fixture.away_points

        return schedule
