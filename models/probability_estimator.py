import pandas as pd
from typing import Any, Union, Optional
from sklearn.linear_model import LogisticRegression
from models.models import PlayerCNN
from dataset.torch_dataset import PlayerDataset
from data import utils
from tqdm import tqdm

import torch.optim as optim
from torch import nn
from models.utils import train_routine
from models.models import PlayerCNN

class ProbabilityEstimator:
    def __init__(self,
                 testing_data: pd.DataFrame,
                 bookie_odds: pd.DataFrame,
                 markets_to_play: list = ['1x2'],
                 model_type: str = 'team',
                 model: Union[LogisticRegression, PlayerCNN] = None,
                 training_data: pd.DataFrame = None,
                 env_path: str = None):
        
        self.features = testing_data
        self.odds = bookie_odds

        if all(market not in ['1x2', 'OU', 'BTTS'] for market in markets_to_play):
            raise ValueError("Only available markets are '1x2', 'OU', and 'BTTS'")
        
        if model_type not in ['team', 'player', 'hybrid']:
            raise ValueError("Only available markets are 'team', 'player', and 'hybrid'")

        self.markets = markets_to_play
        self.model_type = model_type
        self.model = model
        self.training_data = training_data
        self.env_path = env_path
        self.s3 = utils._get_s3_agent(self.env_path)

        self.team_feature_cols = ['venue_diff_home_points', 'venue_diff_np_xg_conceded',
       'venue_diff_vaep_conceded', 'venue_diff_ppda', 'general_diff_',
       'general_diff_points', 'elo_diff_np_xg_season',
       'elo_diff_np_xg_lookback', 'elo_diff_np_xg_conceded_lookback',
       'elo_diff_vaep_season', 'elo_diff_vaep_lookback',
       'elo_diff_ppda_lookback', 'elo_diff_gen']
        
        self.config_cols = ['season', 'game', 'date', 'home_team', 'away_team', 'matchday', 'target']

        if self.model is None:
            if self.training_data is None:
            
                if model_type == 'team':
                    self.model = LogisticRegression(max_iter=10000)
                    train_data = self._get_team_features()
                    self.training_data = train_data[(train_data['season'] != 2324) & (train_data['lookback'] != 1)]

                    X_train = self.training_data[self.team_feature_cols]
                    y_train = self.training_data.target
                    self.model.fit(X_train, y_train)

                elif model_type == 'player':
                    return


        self._get_pred_odds()
        self._extend_bets()
        
    def _get_pred_odds(self):

        if self.model_type == 'team':
            data_to_predict = self.features[self.team_feature_cols]
            config = self.features[self.config_cols].reset_index(drop=True)
            predictions = self.model.predict_proba(data_to_predict)
            pred_df = pd.DataFrame(predictions, columns=['draw_prob', 'home_prob', 'away_prob'])
            pred_df = config.merge(pred_df, how='inner', left_index=True, right_index=True)

            
            pred_odds = pred_df.merge(self.odds[['league', 'game', 'PSCD', 'PSCH', 'PSCA', 'B365C>2.5', 'B365C<2.5']],
                          how='left', on='game')
            
            pred_odds['matchday'].bfill(inplace=True)
            
            self.pred_odds = pred_odds

    def _extend_bets(self):
        # Define columns and markets
        # markets = self.markets
        config_cols = ['league', 'season', 'date', 'home_team', 'away_team']
        draw_cols = ['draw_prob', 'PSCD']
        home_cols = ['home_prob', 'PSCH']
        away_cols = ['away_prob', 'PSCA']

        # Initialize list to store all bets
        all_bets_list = []

        # Iterate over each row in pred_odds
        for i, row in tqdm(self.pred_odds.iterrows(), total=len(self.pred_odds)):
            bets = []

            for market in self.markets:
                if market == '1x2':

                    total_odds = (1 / row.PSCD) + (1 / row.PSCH) + (1 / row.PSCA)
                    # Prepare each bet row
                    draw = row[config_cols + draw_cols].to_dict()
                    draw['result'] = row.target == 0
                    draw['market'] = '1x2_draw'
                    draw['probability'] = draw.pop(draw_cols[0])
                    draw['odds'] = draw.pop(draw_cols[1])
                    draw['imp_odds'] = (1 / draw['odds']) / total_odds
                    bets.append(draw)

                    home = row[config_cols + home_cols].to_dict()
                    home['result'] = row.target == 1
                    home['market'] = '1x2_home'
                    home['probability'] = home.pop(home_cols[0])
                    home['odds'] = home.pop(home_cols[1])
                    home['imp_odds'] = (1 / home['odds']) / total_odds
                    bets.append(home)

                    away = row[config_cols + away_cols].to_dict()
                    away['result'] = row.target == 2
                    away['market'] = '1x2_away'
                    away['probability'] = away.pop(away_cols[0])
                    away['odds'] = away.pop(away_cols[1])
                    away['imp_odds'] = (1 / away['odds']) / total_odds
                    bets.append(away)

            all_bets_list.extend(bets)

        # Convert list to DataFrame once at the end
        self.bets = pd.DataFrame(all_bets_list)



    
    def _get_team_features(self):

        mastercsv = self.s3.get_object(
              Bucket='footballbets',
              Key=f"season_pickles/team_features_diff.csv",
          )
        
        master_df = pd.read_csv(mastercsv['Body'], index_col=0)
        
        return master_df
    

    def _get_player_features(self):
        return

    
