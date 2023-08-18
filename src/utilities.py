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