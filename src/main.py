import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from tpot import TPOTRegressor

from models.individual_player_models import Modelling
from utilities import high_points_by_position

df = pd.read_csv("data/processed/data_with_odds.csv")
high_scoring_gks = high_points_by_position(df, 'GK', 120)

candidate_columns =['total_points_player',
                    'last_5_team_match_points',
                    'last_5_team_goals',
                    'last_5_player_goals',
                    'last_5_games_points_player',
                    'last_5_ind_goals_ratio_player',
                    'last_5_ind_assists_ratio_player',
                    'season_goals_per_match_player_team',
                    'last_5_team_match_points_opp_team',
                    'last_5_team_against_opp_team',
                    'season_goals_against_per_match_opp_team',
                    'season_points_per_match_opp_team',
                    'transfers_in_player',
                    'transfers_out_player',
                    'was_home_player',
                    'last_5_team_against',
                    'season_goals_against_per_match_player_team',
                    'last_5_team_goals_opp_team',
                    'match_draw_odds',
                    'player_team_win_odds',
                    'opponent_team_win_odds',
                    'player_team_win_odds',
                    'opponent_team_win_odds',
                    'over_2.5_goals_odds',
                    ]

gk_columns = ['total_points_player',
              'player_team_win_odds',
 #             'opponent_team_win_odds',
              'over_2.5_goals_odds',
              'last_5_games_points_player',
              'season_goals_against_per_match_player_team',
              'season_goals_per_match_opp_team',
              'last_5_team_against',
#              'was_home_player',
]

random_forest_params = {
                'model__n_estimators': [50, 100, 150],
                'model__max_features': ['log2', 'sqrt'],
                'model__max_depth': [None, 5, 10],
                'model__min_samples_split': [2, 5],
                'model__min_samples_leaf': [1, 2],
                'model__bootstrap': [True, False]
            }

ridge_params  = {
        'model__alpha': [0, 0.0001, 0.001, 0.01, 0.1],
    }

decision_tree_params = {
        'model__splitter':["best", "random"],
        'model__max_depth': [3,5,8],
    }

kneighbors_parameters = {
        'model__n_neighbors':[1,3,5,7,10]
    }

player_model = Modelling(df, high_scoring_gks, gk_columns, DecisionTreeRegressor(), decision_tree_params)
# player_model.position_specific_model_build(high_scoring_gks)
player_model.tpot_modelling(high_scoring_gks, gk_columns, 5)
