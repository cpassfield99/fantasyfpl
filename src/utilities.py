import pandas as pd
import numpy as np

def high_points_by_position(df, position, points_threshold):
    '''Creates a datafame of the players in a specified position that scored over the points threshold in a season.'''
    df = df.copy()
    df = df[df['position_player'] == position]
    grouped_df = df.groupby(['name_player', 'season'])['total_points_player'].sum().reset_index()
    high_points_df = grouped_df[grouped_df['total_points_player']>points_threshold]
    player_names = high_points_df['name_player'].unique()
    return player_names

def generate_fixture_mapping(df, save_location):
    df = df.copy()
    df = df[['team_player', 'was_home_player', 'opponent_team', 'fixture_id_player', 'season']]

    def map_home_team(row):
        if row['was_home_player']:
            return row['team_player']
        else:
            return row['opponent_team']
    def map_away_team(row):
        if row['was_home_player']:
            return row['opponent_team']
        else:
            return row['team_player']
        
    df['home_team'] = df.apply(map_home_team, axis=1)
    df['away_team'] = df.apply(map_away_team, axis=1)
    df = df.drop(columns = ['team_player', 'was_home_player', 'opponent_team'])
    df = df.drop_duplicates()
    df.to_csv(save_location, index=False)

def add_processed_odds(odds, fixture_mapping, player_data):
    fixtures_with_odds = fixture_mapping.merge(odds,
                                                left_on = ['home_team','away_team','season'], 
                                                right_on = ['HomeTeam', 'AwayTeam', 'season'],
                                                how='inner',
                                                )
    fixtures_with_odds = fixtures_with_odds.drop(columns = ['HomeTeam','AwayTeam', 'season'])
    all_data_with_odds = player_data.merge(fixtures_with_odds, on = 'fixture_id_player', how= 'inner')
    all_data_with_odds['match_draw_odds'] = all_data_with_odds['B365D']
    all_data_with_odds['player_team_win_odds'] = np.where(all_data_with_odds['was_home_player'], all_data_with_odds['B365H'], all_data_with_odds['B365A'])
    all_data_with_odds['opponent_team_win_odds'] = np.where(all_data_with_odds['was_home_player'], all_data_with_odds['B365A'], all_data_with_odds['B365H'])
    all_data_with_odds = all_data_with_odds.drop(columns = ['home_team', 'away_team', 'B365H', 'B365D', 'B365A'])
    return all_data_with_odds