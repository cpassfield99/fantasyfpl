import sys
import os

# Get the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory to the Python path
parent_dir = os.path.join(script_dir, "..")
sys.path.append(parent_dir)

import pandas as pd
from utilities import high_points_by_position

from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def prepare_data(position, high_score_threshold, relevant_columns):
    df = pd.read_csv("data/processed/data_with_odds.csv")
    position_df = df[df['position_player']==position]
    high_scorers = high_points_by_position(df, position, high_score_threshold)
    significant_players = position_df[position_df['name_player'].isin(high_scorers)]
    model_df = significant_players[relevant_columns]
    return model_df

def logistic_regression_pipeline(df):
    X = data.drop('target_column', axis=1)
    y = data['target_column']

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a pipeline with scaling and logistic regression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),        # StandardScaler for feature scaling
        ('logreg', LogisticRegression())     # Logistic Regression model
    ])

    # Perform cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)  # 5-fold cross-validation

    # Fit the pipeline on the entire training data
    pipeline.fit(X_train, y_train)

    # Evaluate the pipeline on the test data
    test_score = pipeline.score(X_test, y_test)

    print("Cross-validation scores:", cv_scores)
    print("Mean CV score:", np.mean(cv_scores))
    print("Test score:", test_score)


if __name__ == "__main__":
    gk_columns = ["last_5_games_points_player",
                  "player_win_chance", 
                  "player_lose_chance",
                  "match_draw_chance",
                  "chance_of_over_2.5_goals",
                  "season_goals_against_per_match_player_team",
                  "clean_sheets_player",
                  ]
    
    df = prepare_data('GK', 100, gk_columns)
    print(df)
