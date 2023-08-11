import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression

class Modelling:
    def __init__(self, df, player_list, feature_candidates, model, model_parameters=None):
        '''
        Initialize the RandomForestIndividualPlayer object.
        
        Parameters:
            df (pd.DataFrame): The main DataFrame containing player data.
            player_list (list): List of player names.
            feature_candidates (list): List of feature columns to consider.
        '''
        self.df = df.copy()
        self.player_list = player_list
        self.model = model
        self.model_parameters = model_parameters
        self.feature_candidates = feature_candidates
    
    def create_feature_count_dict(self):
        '''
        Creates a dictionary of column names with a value of 0 for counting feature occurrences.
        
        Returns:
            dict: Dictionary with feature names as keys and initial count as values.
        '''
        columns_dict = {column_name: 0 for column_name in self.feature_candidates}
        return columns_dict

    def player_model_preprocessing(self, df):
        '''
        Prepare the dataframe for a single player so it can be used in a model.
        
        Parameters:
            df (pd.DataFrame): DataFrame containing player data for preprocessing.
        
        Returns:
            pd.DataFrame: Preprocessed DataFrame for a single player.
        '''
        df = df.dropna()
    
        # Filter to only have games where players have played more than 15 minutes.
        # Also filter out hard to predict high scores.
        df = df[df['minutes_player'] > 15]
        df = df[df['total_points_player'] <= 10]
        df = df.loc[:, self.feature_candidates]
        return df
    
    def train_test_splitting(self, df):
        '''
        Perform a consistent train test split for ease of comparability between models
        
        Parameters:
            df (pd.DataFrame): DataFrame containing player data for modelling.
            
        Returns:
            pd.DataFrame: Train and test sets for X and y'''
        
        X = df.drop(columns=['total_points_player'])
        y = df['total_points_player']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    
    def build_model(self,
                  X_train, 
                  X_test, 
                  y_train, 
                  y_test,
                  model,
                  model_hyperparameters, 
                  num_feature_subsets=20):
        '''
        Perform random forest modeling for a given DataFrame.
        
        Parameters:
            X_train (pd.DataFrame): DataFrame containing player data for training.
            X_test (pd.DataFrame): DataFrame containing player data for testing.
            y_train (pd.DataFrame): DataFrame containing training target.
            y_test (pd.DataFrame): DataFrame containing test target.
            num_feature_subsets (int): Number of random feature subsets to consider.
        
        Returns:
            tuple: Best feature subset (list of feature names) and corresponding mean absolute error.
        '''
        best_mae = float('inf')
        best_feature_subset = None
        
        for _ in range(num_feature_subsets):

            selected_features = np.random.choice(X_train.columns, size=np.random.randint(6, len(X_train.columns) + 1), replace=False)
            X_train_subset = X_train[selected_features]
            X_test_subset = X_test[selected_features]
            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=model_hyperparameters,
                n_iter=5,
                scoring='neg_mean_absolute_error',
                cv=5,
                random_state=42,
                n_jobs=-1,
                verbose=0
            )

            random_search.fit(X_train_subset, y_train)
            best_model = random_search.best_estimator_
            y_pred = best_model.predict(X_test_subset)
            test_mae = mean_absolute_error(y_test, y_pred)

            if test_mae < best_mae:
                best_mae = test_mae
                best_feature_subset = selected_features
                best_model = random_search.best_estimator_

        return best_feature_subset, best_mae
    
    def optimise_feature_selection(self, num_feature_subsets=20):
        '''
        Optimize feature selection for multiple players and count the occurrences of selected features.
        
        Parameters:
            num_feature_subsets (int): Number of random feature subsets to consider for each player.
        
        Returns:
            tuple: A dictionary containing feature names as keys and occurrence counts as values,
                   and the number of players used for optimization.
        '''
        feature_dict = self.create_feature_count_dict()
        players_used = 0
        player_model_scores = {}
        
        for player in self.player_list:
            player_df = self.df[self.df['name_player'] == player].copy()
            player_df = self.player_model_preprocessing(player_df)
            X_train, X_test, y_train, y_test = self.train_test_splitting(player_df)
            best_features, mae_score = self.build_model(X_train,
                                                      X_test,
                                                      y_train,
                                                      y_test,
                                                      self.model,
                                                      self.model_parameters,
                                                      num_feature_subsets)

            player_model_scores[player] = mae_score    

            if mae_score < 2:
                players_used += 1
                for value in best_features:
                    if value in feature_dict:
                        feature_dict[value] += 1

        return feature_dict, players_used, player_model_scores
