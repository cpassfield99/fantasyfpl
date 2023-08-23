import sys
import os
import pandas as pd
import numpy as np
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# Get the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory to the Python path
parent_dir = os.path.join(script_dir, "..")
sys.path.append(parent_dir)

from utilities import high_points_by_position

def prepare_data(position, high_score_threshold, relevant_columns):
    df = pd.read_csv("data/processed/data_with_odds.csv")
    position_df = df[df['position_player']==position]
    high_scorers = high_points_by_position(df, position, high_score_threshold)
    significant_players = position_df[position_df['name_player'].isin(high_scorers)]
    model_df = significant_players[relevant_columns]
    print(model_df.shape)
    model_df = model_df.dropna(axis=0)
    print(model_df.shape)
    return model_df

def logistic_regression_pipeline(df):
    X = df.drop('clean_sheets_player', axis=1)
    y = df['clean_sheets_player']

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

    # Create a pipeline with scaling and logistic regression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),        # StandardScaler for feature scaling
        ('logreg', LogisticRegression())     # Logistic Regression model
    ])

    # Perform cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=10)  # 5-fold cross-validation

    # Fit the pipeline on the entire training data
    pipeline.fit(X_train, y_train)

    # Evaluate the pipeline on the test data
    test_score = pipeline.score(X_test, y_test)
    y_pred = pipeline.predict(X_test)
    f1 = f1_score(y_test, y_pred)

    print("Cross-validation scores:", cv_scores)
    print("Mean CV score:", np.mean(cv_scores))
    print("Test score:", test_score)
    print("F1 score:", f1)
    print("Proportion of clean sheets:", y.mean())

def auto_ml_modelling(df, generations=5):
    '''Using tpot to find an optimal model with the features that seem most reasonable using domain knowledge.
    Once an optimal model is found using TPOT, further feature selection could be done on this model.'''
    # Filter df to required players only
    df = df.copy()
    X = df.drop('clean_sheets_player', axis=1)
    y = df['clean_sheets_player']

    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=41, stratify=y)

    tpot = TPOTClassifier(generations=generations, cv=5, random_state=42, verbosity=2, n_jobs=-1)
    tpot.fit(X_train, y_train)

    print("Proportion of clean sheets:", y.mean())
    print(tpot.score(X_test, y_test))


if __name__ == "__main__":
    gk_columns = [
        # "last_10_games_points_player",
         "player_win_chance", 
        # "player_lose_chance",
        # "match_draw_chance",
         "chance_of_over_2.5_goals",
        # "season_goals_against_per_match_player_team",
         "clean_sheets_player",
        # "last_10_team_goals_opp_team",
        # "last_10_team_against"
        ]
    
    df = prepare_data('GK', 120, gk_columns)
    auto_ml_modelling(df)
