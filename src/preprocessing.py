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

        self.teams2021_pp = self.preprocess_team(self.teams2021)
        self.teams2122_pp = self.preprocess_team(self.teams2122)
        self.teams2223_pp = self.preprocess_team(self.teams2223)

        self.pp_2021 = self.concat_preprocess(self.gws2021, '2020-21')
        self.pp_2122 = self.concat_preprocess(self.gws2122, '2021-22')
        self.pp_2223 = self.concat_preprocess(self.gws2223, '2022-23')

        self.data = self.concat_data(self.pp_2021, self.pp_2122, self.pp_2223)

        self.data = self.generate_fixture_id(self.data)
        
        self.data = self.name_mapping(self.data)
    
    def preprocess_team(self,df):
        df = df[['id','name']]
        return df
    
    def concat_preprocess(self,df, season):
        df['season'] = season
        team_pp_dict = {'2022-23': self.teams2223_pp, '2021-22': self.teams2122_pp, '2020-21': self.teams2021_pp}
        df = df.merge(team_pp_dict[season], left_on='opponent_team', right_on='id') \
            .rename(columns={'id_x': 'id', 'name_x': 'name', 'opponent_team': 'opponent_team_id', 'name_y': 'opponent_team'}) \
            .drop(columns=['id'])
        return df

    def concat_data(self, *dfs):
        '''concat dfs and drop columns with nulls'''
        df = pd.concat(dfs)
        df = df.dropna(axis=1)
        return df
    
    def generate_fixture_id(self, df):
        df['season_temp'] = df['season'].apply(lambda x: x.split('-')[0][-2:] + x.split('-')[1]).astype(str)
        df['fixture_string'] = df['fixture'].astype(str)
        df['fixture_id'] = df['season_temp'] + df['fixture_string']
        df = df.drop(columns = ['fixture_string', 'season_temp'])
        return df
    
    def name_mapping(self,df):
        '''Only a temporary fix, need to figure out longer term string matching solution'''
        string_mapping = {
            'Benjamin Chilwell' : 'Ben Chilwell',
            'Bernardo Fernandes Da Silva Junior':'Bernardo Fernandes da Silva Junior',
            'Bernardo Mota Veiga de Carvalho e Silva':'Bernardo Veiga de Carvalho e Silva',
            'Björn Engels':'Bjorn Engels',
            'Bobby Decordova-Reid':'Bobby De Cordova-Reid',
            'Bruno André Cavaco Jordão':'Bruno André Cavaco Jordao',
            'Bruno Miguel Borges Fernandes':'Bruno Borges Fernandes',
            'Carlos Vinícius Alves Morais':'Carlos Vinicius Alves Morais',
            'José Diogo Dalot Teixeira':'Diogo Dalot Teixeira',
            'Emerson Aparecido Leite de Souza Junior':'Emerson Leite de Souza Junior',
            'Emiliano Martínez':'Emiliano Martínez Romero',
            'Gabriel Teodoro Martinelli Silva':'Gabriel Martinelli Silva',
            'Junior Firpo Adames':'Héctor Junior Firpo Adames',
            'Ivan Neves Abreu Cavaleiro':'Ivan Ricardo Neves Abreu Cavaleiro',
            'Joseph Willock':'Joe Willock',
            'Joshua Wilson-Esbrand':'Josh Wilson-Esbrand',
            'Lyanco Silveira Neves Vojnovic':'Lyanco Evangelista Silveira Neves Vojnovic',
            'Matija Šarkić':'Matija Šarkic',
            'Matija Šarkic':'Matija Sarkic',
            'André Filipe Tavares Gomes':'André Tavares Gomes',
            'Mattéo Guendouzi':'Matteo Guendouzi',  
            'Mbwana Samatta':'Mbwana Ally Samatta',
            'Willian Borges Da Silva':'Willian Borges da Silva',
            'William Smallbone' : 'Will Smallbone',
            'Thomas McGill':'Tom McGill',
            'Rúben Santos Gato Alves Dias':'Rúben Gato Alves Dias',
            'Roméo Lavia':'Romeo Lavia',
            'Rúben da Silva Neves':'Rúben Diogo da Silva Neves',
            'Rayan Ait Nouri':'Rayan Aït-Nouri',
            'Ricardo Barbosa Pereira':'Ricardo Domingos Barbosa Pereira',
            'Finley Stevens':'Fin Stevens',
            'Matt Clarke':'Matthew Clarke'
            }

        df['name'] = df['name'].replace(string_mapping)
        return df

    
    def get_preprocessed_data(self):
        return self.data
         

