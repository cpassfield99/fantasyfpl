import pandas as pd

class PlayerStats:
    def __init__(self, df):
        self.df = df

    def drop_irrelevant_cols(self):
        cols_to_drop = ['xP', 'element', 'transfers_in', 'transfers_out','ict_index','own_goals','round', 'opponent_team_id','fixture']
        return self.df.drop(columns=cols_to_drop, axis=1, inplace=False)

    def match_points(self, row):
        if row['was_home'] == True:
            if row['team_h_score'] > row['team_a_score']:
                return 3
            elif row['team_h_score'] == row['team_a_score']:
                return 1
            else:
                return 0
        else:
            if row['team_h_score'] < row['team_a_score']:
                return 3
            elif row['team_h_score'] == row['team_a_score']:
                return 1
            else:
                return 0
            
    def team_goals(self,row):
        if row['was_home'] == True:
            return row['team_h_score']
        else:
            return row['team_a_score']
        
    def team_goals_against(self,row):
        if row['was_home'] == True:
            return row['team_a_score']
        else:
            return row['team_h_score']

    def team_points_form(self, group, num_matches):
        df = group.sort_values(by='order', ascending=True)
        rolling_sum = df['team_points'].rolling(window=num_matches, min_periods=1).sum().shift(1)
        rolling_count = df['team_points'].rolling(window=num_matches, min_periods=1).count().shift(1)
        df[f'last_{num_matches}_team_match_points'] = ((rolling_sum / rolling_count)*num_matches).fillna(0)
        return df
    
    def team_goals_form(self, group, num_matches):
        df = group.sort_values(by='order', ascending=True)
        rolling_sum = df['team_goals'].rolling(window=num_matches, min_periods=1).sum().shift(1)
        rolling_count = df['team_goals'].rolling(window=num_matches, min_periods=1).count().shift(1)
        df[f'last_{num_matches}_team_goals'] = ((rolling_sum / rolling_count)*num_matches).fillna(0)
        return df
    
    def team_goals_against_form(self, group, num_matches):
        df = group.sort_values(by='order', ascending=True)
        rolling_sum = df['team_goals_against'].rolling(window=num_matches, min_periods=1).sum().shift(1)
        rolling_count = df['team_goals_against'].rolling(window=num_matches, min_periods=1).count().shift(1)
        df[f'last_{num_matches}_team_against'] = ((rolling_sum / rolling_count)*num_matches).fillna(0)
        return df
    
    def individual_goals_form(self, group, num_matches):
        df = group.sort_values(by='order', ascending=True)
        rolling_sum = df['goals_scored'].rolling(window=num_matches, min_periods=1).sum().shift(1)
        rolling_count = df['goals_scored'].rolling(window=num_matches, min_periods=1).count().shift(1)
        df[f'last_{num_matches}_games_goals'] = ((rolling_sum / rolling_count)*num_matches).fillna(0)
        return df
    
    def individual_points_form(self, group, num_matches):
        df = group.sort_values(by='order', ascending=True)
        rolling_sum = df['total_points'].rolling(window=num_matches, min_periods=1).sum().shift(1)
        rolling_count = df['total_points'].rolling(window=num_matches, min_periods=1).count().shift(1)
        df[f'last_{num_matches}_games_points'] = ((rolling_sum / rolling_count)*num_matches).fillna(0)
        return df
    
    def individual_goals_ratio(self, group, num_matches):
        df = group.sort_values(by='order', ascending=True)
        df[f'last_{num_matches}_ind_goals_ratio'] = df['goals_scored'].rolling(window=num_matches, min_periods=1).mean().shift(1) / df['team_goals'].rolling(window=num_matches, min_periods=1).mean().shift(1)
        df[f'last_{num_matches}_ind_goals_ratio'] = df[f'last_{num_matches}_ind_goals_ratio'].fillna(0)
        return df
    
    def individual_assists_ratio(self, group, num_matches):
        df = group.sort_values(by='order', ascending=True)
        df[f'last_{num_matches}_ind_assists_ratio'] = df['assists'].rolling(window=num_matches, min_periods=1).mean().shift(1) / df['team_goals'].rolling(window=num_matches, min_periods=1).mean().shift(1)
        df[f'last_{num_matches}_ind_assists_ratio'] = df[f'last_{num_matches}_ind_assists_ratio'].fillna(0)
        return df

    def process_data(self, num_matches):
        df_copy = self.df.copy()
        df_copy['team_points'] = df_copy.apply(self.match_points, axis=1)
        df_copy['team_goals'] = df_copy.apply(self.team_goals, axis=1)
        df_copy['team_goals_against'] = df_copy.apply(self.team_goals_against, axis=1)
        df_copy['order'] = df_copy.groupby('name')['kickoff_time'].rank(ascending=True, method='dense')
        df_copy = df_copy.groupby(['name'], group_keys=False).apply(self.team_points_form, num_matches).reset_index(drop=True)
        df_copy = df_copy.groupby(['name'], group_keys=False).apply(self.team_goals_form, num_matches).reset_index(drop=True)
        df_copy = df_copy.groupby(['name'], group_keys=False).apply(self.individual_goals_form, num_matches).reset_index(drop=True)
        df_copy = df_copy.groupby(['name'], group_keys=False).apply(self.individual_points_form, num_matches).reset_index(drop=True)
        df_copy = df_copy.groupby(['name'], group_keys=False).apply(self.team_goals_against_form,num_matches).reset_index(drop=True)
        df_copy = df_copy.groupby(['name'], group_keys=False).apply(self.individual_goals_ratio,num_matches).reset_index(drop=True)
        df_copy = df_copy.groupby(['name'], group_keys=False).apply(self.individual_assists_ratio,num_matches).reset_index(drop=True)

        return df_copy


class TeamStats:
    def __init__(self, match_df):
        self.match_df = match_df
        self.team_df = None
    
    def calculate_team_df(self, num_matches):

        team_df_copy = self.match_df.copy()
        team_df_copy = team_df_copy.groupby(['season','GW','fixture','team']).agg({
            'team_goals': 'first',
            'team_goals_against': 'first',
            'was_home': 'first',
            'opponent_team': 'first',
            'team_points': 'first',
            'order': 'first'
        }).reset_index()

        team_df_copy['total_points_season'] = team_df_copy.groupby(['season','team'])['team_points'].cumsum() - team_df_copy['team_points']
        team_df_copy['total_goals_season'] = team_df_copy.groupby(['season','team'])['team_goals'].cumsum() - team_df_copy['team_goals']
        team_df_copy['total_goals_against_season'] = team_df_copy.groupby(['season','team'])['team_goals_against'].cumsum() - team_df_copy['team_goals_against']
        team_df_copy['goal_difference'] = team_df_copy['total_goals_season'] - team_df_copy['total_goals_against_season']
        team_df_copy['season_matches_played'] = team_df_copy.groupby(['season','team'])['fixture'].cumcount()

        team_df_copy = team_df_copy.groupby(['team'], group_keys=False).apply(self.team_points_form, num_matches).reset_index(drop=True)
        team_df_copy = team_df_copy.groupby(['team'], group_keys=False).apply(self.team_goals_form, num_matches).reset_index(drop=True)
        team_df_copy = team_df_copy.groupby(['team'], group_keys=False).apply(self.team_goals_against_form,num_matches).reset_index(drop=True)

        team_df_copy['season_goals_per_match'] = team_df_copy['total_goals_season'] / team_df_copy['season_matches_played']
        team_df_copy['season_goals_against_per_match'] = team_df_copy['total_goals_against_season'] / team_df_copy['season_matches_played']
        team_df_copy['season_goal_difference_per_match'] = team_df_copy['goal_difference'] / team_df_copy['season_matches_played']
        team_df_copy['season_points_per_match'] = team_df_copy['total_points_season'] / team_df_copy['season_matches_played']


        self.team_df = team_df_copy

        return team_df_copy

    def _rolling_position_form(self, group, num_matches):
        df = group.sort_values(by='order', ascending=True)
        rolling_sum = df['league_position'].rolling(window=num_matches, min_periods=1).sum().shift(1)
        rolling_count = df['league_position'].rolling(window=num_matches, min_periods=1).count().shift(1)
        df[f'last_{num_matches}_games_league_position'] = ((rolling_sum / rolling_count)).fillna(0)
        return df
    
    def team_points_form(self, group, num_matches):
        df = group.sort_values(by='order', ascending=True)
        rolling_sum = df['team_points'].rolling(window=num_matches, min_periods=1).sum().shift(1)
        rolling_count = df['team_points'].rolling(window=num_matches, min_periods=1).count().shift(1)
        df[f'last_{num_matches}_team_match_points'] = ((rolling_sum / rolling_count)*num_matches).fillna(0)
        return df
    
    def team_goals_form(self, group, num_matches):
        df = group.sort_values(by='order', ascending=True)
        rolling_sum = df['team_goals'].rolling(window=num_matches, min_periods=1).sum().shift(1)
        rolling_count = df['team_goals'].rolling(window=num_matches, min_periods=1).count().shift(1)
        df[f'last_{num_matches}_team_goals'] = ((rolling_sum / rolling_count)*num_matches).fillna(0)
        return df
    
    def team_goals_against_form(self, group, num_matches):
        df = group.sort_values(by='order', ascending=True)
        rolling_sum = df['team_goals_against'].rolling(window=num_matches, min_periods=1).sum().shift(1)
        rolling_count = df['team_goals_against'].rolling(window=num_matches, min_periods=1).count().shift(1)
        df[f'last_{num_matches}_team_against'] = ((rolling_sum / rolling_count)*num_matches).fillna(0)
        return df


class CombineTeamAndPlayers():
    def __init__(self, player_data, team_data, num_matches):
        self.player_data = player_data
        self.team_data = team_data

        self.columns_to_drop = ['xP_player',
                                'element_player',
                                'opponent_team_id_player',
                                'team_a_score_player',
                                'team_h_score_player',
                                'order_player',
                                'order_player_team',
                                f'last_{num_matches}_team_match_points_player_team',
                                f'last_{num_matches}_team_goals_player_team',
                                f'last_{num_matches}_team_against_player_team',
                                'was_home_opp_team',
                                'team_points_opp_team',
                                'order_opp_team',
                                'team_points_player',
                                'round_player'] 
        
        self.rename_columns = {'kickoff_time_player': 'kickoff_time',
                               'season_player': 'season',
                               'opponent_team_player': 'opponent_team',
                               'team_goals_player': 'team_season_goals_scored',
                               'team_goals_against_player': 'team_season_goals_against',
                               f'last_{num_matches}_team_match_points_player':f'last_{num_matches}_team_match_points',
                               f'last_{num_matches}_team_goals_player':f'last_{num_matches}_team_goals',
                               f'last_{num_matches}_games_goals_player':f'last_{num_matches}_player_goals',
                               f'last_{num_matches}_team_against_player':f'last_{num_matches}_team_against',
                               'goal_difference_player_team':'season_goal_difference_player_team',
                               'goal_difference_opp_team':'season_goal_difference_opp_team'}
                

    def combine(self):
        
        player_data_copy = self.player_data.copy()
        team_data_copy = self.team_data.copy()

        player_data_copy = player_data_copy.add_suffix('_player')
        player_team_data_copy = team_data_copy.add_suffix('_player_team')

        df = pd.merge( player_data_copy, 
                       player_team_data_copy,
                       left_on=['team_player','season_player','GW_player','fixture_player'],
                       right_on=['team_player_team','season_player_team','GW_player_team','fixture_player_team'],
                       how='inner')
        
        opponent_team_data = team_data_copy.add_suffix('_opp_team')
        
        df = pd.merge(df,
                       opponent_team_data,
                       left_on=['season_player', 'fixture_player', 'opponent_team_player'], 
                       right_on=['season_opp_team', 'fixture_opp_team', 'team_opp_team'], 
                       how='inner')
        df = df.loc[:,~df.T.duplicated(keep='first')]

        df = df.drop(columns=self.columns_to_drop, axis=1)
        df = df.rename(columns=self.rename_columns)

        df['player_season_fixture_number'] = df['season_matches_played_player_team'] +1

        return df

        


