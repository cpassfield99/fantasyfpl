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

class Modelling:
    def __init__(self, df, player_list, feature_candidates, model, model_hyperparameters=None):
        '''
        Initialize the Modelling object.
        
        Parameters:
            df (pd.DataFrame): The main DataFrame containing player data.
            player_list (list): List of player names.
            feature_candidates (list): List of feature columns to consider.
        '''
        self.df = df.copy()
        self.player_list = player_list
        self.model = model
        self.model_hyperparameters = model_hyperparameters
        self.feature_candidates = feature_candidates
    
    def create_feature_count_dict(self):
        '''
        Creates a dictionary of column names with a value of 0 for counting feature occurrences.
        
        Returns:
            dict: Dictionary with feature names as keys and initial count as values.
        '''
        columns_dict = {column_name: 0 for column_name in self.feature_candidates}
        return columns_dict

    def player_model_preprocessing(self, df, points_threshold=10, minutes_threshold=15):
        '''
        Prepare the dataframe for a single player so it can be used in a model.
        
        Parameters:
            df (pd.DataFrame): DataFrame containing player data for preprocessing.
        
        Returns:
            pd.DataFrame: Preprocessed DataFrame for a single player.
        '''
        df = df.copy()
        df = df.dropna()
    
        # Filter to only have games where players have played more than minutes_threshold minutes.
        # Also filter out hard to predict high scores.
        df = df[df['minutes_player'] > minutes_threshold]
        df = df[df['total_points_player'] <= points_threshold]
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
    
    def player_specific_model_build(self,
                                    X_train, 
                                    X_test, 
                                    y_train, 
                                    y_test,
                                    model,
                                    model_hyperparameters, 
                                    num_feature_subsets=20
                                    ):
        '''
        Perform player specifc modeling for a given DataFrame.
        
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

        pipeline = Pipeline(steps= [
            ('scaler', StandardScaler()),
            ('model', model)
        ])

        for _ in range(num_feature_subsets):

            selected_features = np.random.choice(X_train.columns, size=np.random.randint(5, len(X_train.columns) + 1), replace=False)
            X_train_subset = X_train[selected_features]
            X_test_subset = X_test[selected_features]
            random_search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=model_hyperparameters,
                n_iter=5,
                scoring='neg_mean_absolute_error',
                cv=5,
                random_state=42,
                n_jobs=-1,
                verbose=0
            )

            random_search.fit(X_train_subset, y_train)
            best_model_for_features = random_search.best_estimator_
            y_pred = best_model_for_features.predict(X_test_subset)
            test_mae = mean_absolute_error(y_test, y_pred)

            if test_mae < best_mae:
                best_mae = test_mae
                best_feature_subset = selected_features
                best_model = best_model_for_features

        return best_feature_subset, best_model, best_mae
    
    def position_specific_model_build(self,
                                      player_list,
                                      num_feature_subsets=100
                                      ):
        
        best_mae = float('inf')
        best_feature_subset = None

        for _ in range(num_feature_subsets):
            print(_)
            all_features = self.feature_candidates.copy()
            all_features.remove("total_points_player")
            selected_features = np.random.choice(all_features, size=np.random.randint(6, len(all_features) + 1), replace=False)
            mae_list = []

            for player in player_list:

                pipeline = Pipeline(steps= [
                    ('scaler', StandardScaler()),
                    ('model', self.model),
                    ])
                
                player_df = self.df[self.df['name_player'] == player].copy()
                player_df = self.player_model_preprocessing(player_df)
                X_train, X_test, y_train, y_test = self.train_test_splitting(player_df)
                X_train_subset, X_test_subset = X_train[selected_features], X_test[selected_features]

                random_search = RandomizedSearchCV( estimator=pipeline,
                                                    param_distributions=self.model_hyperparameters,
                                                    n_iter=5,
                                                    scoring='neg_mean_absolute_error',
                                                    cv=5,
                                                    random_state=42,
                                                    n_jobs=-1,
                                                    verbose=0
                                                    )
                random_search.fit(X_train_subset, y_train)
                best_model_for_features = random_search.best_estimator_
                y_pred = best_model_for_features.predict(X_test_subset)
                test_mae = mean_absolute_error(y_test, y_pred)
                mae_list.append(test_mae)

            mean_mae = sum(mae_list)/len(mae_list)
            
            if mean_mae < best_mae:
                best_mae = mean_mae
                best_feature_subset = selected_features
                print(f"The mae got updated to {best_mae}, using the parameters of \n {best_feature_subset}")

        print(f"The final mae is {best_mae}, using the parameters of \n {best_feature_subset}")
        return best_feature_subset, best_mae
    
    def tpot_modelling(self, players, feature_subset, generations=5):
        '''Using tpot to find an optimal model with the features that seem most reasonable using domain knowledge.
        Once an optimal model is found using TPOT, further feature selection could be done on this model.'''

        # Filter df to required players only
        df = self.df.copy()
        filtered_df = df[df['name_player'].isin(players)]
        filtered_df = self.player_model_preprocessing(filtered_df)
        filtered_df = filtered_df[feature_subset]
        X_train, X_test, y_train, y_test = self.train_test_splitting(filtered_df)
        tpot = TPOTRegressor(generations=generations, cv=5, random_state=42, verbosity=2, scoring="neg_mean_absolute_error", n_jobs=-1)
        tpot.fit(X_train, y_train)
        print(tpot.score(X_test, y_test))

    def optimise_feature_selection(self, num_feature_subsets):
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
            player_best_features, player_best_model, player_mae_score = self.player_specific_model_build(X_train,
                                                                                                        X_test,
                                                                                                        y_train,
                                                                                                        y_test,
                                                                                                        self.model,
                                                                                                        self.model_hyperparameters,
                                                                                                        num_feature_subsets
                                                                                          )

            player_model_scores[player] = player_mae_score   

            if player_mae_score < 2:
                players_used += 1
                for value in player_best_features:
                    if value in feature_dict:
                        feature_dict[value] += 1

        return feature_dict, players_used, player_model_scores, player_best_features, player_best_model