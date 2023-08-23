import pandas as pd

from data_preprocessing.preprocessing import PreprocessData, AddMatchOdds, generate_fixture_mapping, add_processed_odds
from data_preprocessing.player_stats import PlayerStats, TeamStats, CombineTeamAndPlayers

# Combine data from multiple seasons together
loaded_data = PreprocessData('data/raw/2020-21/teams.csv', 
                      'data/raw/2021-22/teams.csv', 
                      'data/raw/2022-23/teams.csv', 
                      'data/raw/2020-21/gws/merged_gw.csv', 
                      'data/raw/2021-22/gws/merged_gw.csv', 
                      'data/raw/2022-23/gws/merged_gw.csv')
team_form = 10
preprocessed_data = loaded_data.get_preprocessed_data()

preprocessed_data.to_csv("data/interim/all_seasons_player_data.csv", index=False)

# Add player/team stats to complete df
player_stats = PlayerStats(preprocessed_data)
players_df = player_stats.process_data(team_form)

team_stats = TeamStats(players_df)
team_df = team_stats.calculate_team_df(team_form)

combine_team_and_player = CombineTeamAndPlayers(players_df, team_df, team_form)
player_data_with_stats = combine_team_and_player.combine()

player_data_with_stats.to_csv("data/interim/all_seasons_player_stats.csv", index=False)

# Generate fixture id mapping
fixture_id_mapping = generate_fixture_mapping(player_data_with_stats)

fixture_id_mapping.to_csv("data/processed/fixture_id_mapping.csv", index=False)

# Process match odds data
match_odds = AddMatchOdds("data/raw/2020-21/odds/E0.csv",
                          "data/raw/2021-22/odds/E0.csv",
                          "data/raw/2022-23/odds/E0.csv",
                          "data/processed/fixture_id_mapping.csv")

combined_odds = match_odds.get_preprocessed_odds()

combined_odds.to_csv("data/interim/all_match_odds.csv", index=False)

fixture_mapping = pd.read_csv("data/processed/fixture_id_mapping.csv")
player_data_with_stats = pd.read_csv("data/interim/all_seasons_player_stats.csv")
player_data_with_odds = add_processed_odds(combined_odds, fixture_mapping, player_data_with_stats)

player_data_with_odds.to_csv("data/processed/data_with_odds.csv", index=False)


