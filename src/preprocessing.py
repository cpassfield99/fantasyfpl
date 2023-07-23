import pandas as pd

class PreprocessData:
    def __init__(self, teams2021_path, teams2122_path, teams2223_path, gws2021_path, gws2122_path, gws2223_path):
        self.teams2021 = pd.read_csv(teams2021_path)
        self.teams2122 = pd.read_csv(teams2122_path)
        self.teams2223 = pd.read_csv(teams2223_path)
        self.gws2021 = pd.read_csv(gws2021_path)
        self.gws2122 = pd.read_csv(gws2122_path)
        self.gws2223 = pd.read_csv(gws2223_path)

        self.gws2223 = self.gws2223.dropna(subset=['GW'])

        self.teams2021_pp = self.prepro_team(self.teams2021)
        self.teams2122_pp = self.prepro_team(self.teams2122)
        self.teams2223_pp = self.prepro_team(self.teams2223)

        self.pp_2021 = self.concat_prepro(self.gws2021, '2020-21')
        self.pp_2122 = self.concat_prepro(self.gws2122, '2021-22')
        self.pp_2223 = self.concat_prepro(self.gws2223, '2022-23')

        self.data = self.concat_data(self.pp_2021, self.pp_2122, self.pp_2223)
    
    def prepro_team(self,df):
        df = df[['id','name']]
        return df
    
    def concat_prepro(self,df, season):
        df['season'] = season
        if season == '2022-23':
            df = df.merge(self.teams2223_pp, left_on='opponent_team', right_on='id')
        elif season == '2021-22':
            df = df.merge(self.teams2122_pp, left_on='opponent_team', right_on='id')
        elif season == '2020-21':
            df = df.merge(self.teams2021_pp, left_on='opponent_team', right_on='id')
        df = df.rename(columns={'id_x':'id', 'name_x':'name','opponent_team':'opponent_team_id', 'name_y':'opponent_team'})
        df = df.drop(columns=['id'])
        return df

    def concat_data(self, *dfs):
        '''concat dfs and drop columns with nulls'''
        df = pd.concat(dfs)
        df = df.dropna(axis=1)
        return df
        
    def get_preprocessed_data(self):
        return self.data
         

