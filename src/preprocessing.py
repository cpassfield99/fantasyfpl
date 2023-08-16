import os
import pandas as pd

class PreprocessData:
    def __init__(self, teams2021_path, teams2122_path, teams2223_path, gws2021_path, gws2122_path, gws2223_path):
        # Validate CSV file paths before reading
        self._validate_file_paths(teams2021_path, teams2122_path, teams2223_path, gws2021_path, gws2122_path, gws2223_path)

        self.teams2021 = pd.read_csv(teams2021_path)
        self.teams2122 = pd.read_csv(teams2122_path)
        self.teams2223 = pd.read_csv(teams2223_path)
        self.gws2021 = pd.read_csv(gws2021_path)
        self.gws2122 = pd.read_csv(gws2122_path)
        self.gws2223 = pd.read_csv(gws2223_path)

        self.gws2223 = self.gws2223.dropna(subset=['GW'])

        self.teams2021_pp = self._preprocess_team(self.teams2021)
        self.teams2122_pp = self._preprocess_team(self.teams2122)
        self.teams2223_pp = self._preprocess_team(self.teams2223)

        self.pp_2021 = self._concat_preprocess(self.gws2021, '2020-21')
        self.pp_2122 = self._concat_preprocess(self.gws2122, '2021-22')
        self.pp_2223 = self._concat_preprocess(self.gws2223, '2022-23')

        self.data = self._concat_data(self.pp_2021, self.pp_2122, self.pp_2223)

        self.data = self._generate_fixture_id(self.data)
        self.data = self._name_mapping(self.data)
        self.data = self._kickout_time_conversion(self.data)


    def _validate_file_paths(self, *file_paths):
        """
        Validates the existence of provided file paths.

        Parameters:
        *file_paths (str): One or more file paths to be validated.

        Raises:
        FileNotFoundError: If any of the provided file paths do not exist.
        """
        for file_path in file_paths:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

    def _preprocess_team(self, df):
        """
        Preprocesses team data by selecting specific columns.

        This method takes a DataFrame containing team data and performs the following preprocessing steps:
        1. Selects the 'id' and 'name' columns from the input DataFrame.
        2. Returns a new DataFrame with only the selected columns.

        Parameters:
        df (DataFrame): The input DataFrame containing team data.

        Returns:
        DataFrame: A preprocessed DataFrame with only the 'id' and 'name' columns.
        """
        df = df[['id','name']]
        return df
    
    def _concat_preprocess(self, df, season):
        """
        Preprocess and concatenate data based on the specified season.

        This function takes a DataFrame and a season, and performs the following steps:
        1. Validates the provided season against a list of valid seasons.
        2. Adds a 'season' column to the DataFrame.
        3. Merges the DataFrame with preprocessed team data based on the given season.
        4. Renames columns and drops unnecessary columns.

        Parameters:
        df (DataFrame): The input DataFrame containing data to be preprocessed and concatenated.
        season (str): The season for which data should be preprocessed. Valid values are: '2022-23', '2021-22', '2020-21'.

        Returns:
        DataFrame: The preprocessed and concatenated DataFrame.

        Raises:
        ValueError: If an invalid season is provided.

        Note:
        The function assumes that the class instance has attributes 'teams2223_pp', 'teams2122_pp', and 'teams2021_pp',
        representing preprocessed team data for the respective seasons.
        """
        valid_seasons = ['2022-23', '2021-22', '2020-21']
        
        if season not in valid_seasons:
            raise ValueError("Invalid season provided. Valid seasons are: {}".format(", ".join(valid_seasons)))
        
        df['season'] = season
        
        season_pp_data = {
            '2022-23': self.teams2223_pp,
            '2021-22': self.teams2122_pp,
            '2020-21': self.teams2021_pp,
        }
                
        merged_df = df.merge(season_pp_data[season], left_on='opponent_team', right_on='id')
        merged_df = merged_df.rename(columns={
            'id_x': 'id', 'name_x': 'name', 'opponent_team': 'opponent_team_id', 'name_y': 'opponent_team'
        }).drop(columns=['id'])
        
        return merged_df
    
    def _concat_data(self, *dfs, axis=0, drop_null_columns=True):
        """
        Concatenate multiple DataFrames and optionally drop columns with null values.

        Parameters:
        *dfs (DataFrame): One or more DataFrames to concatenate.
        axis (int, optional): Concatenation axis. Use 0 for rows (default), 1 for columns.
        drop_null_columns (bool, optional): Whether to drop columns with null values (default=True).

        Returns:
        DataFrame: Concatenated DataFrame.
        """
        if not all(isinstance(df, pd.DataFrame) for df in dfs):
            raise ValueError("All arguments must be DataFrames.")

        concatenated_df = pd.concat(dfs, axis=axis)

        if drop_null_columns:
            concatenated_df = concatenated_df.dropna(axis=1)

        return concatenated_df
    
    def _generate_fixture_id(self, df):
        """
        Generates unique fixture IDs based on season and fixture number.

        This method takes a DataFrame containing match fixture data and generates unique fixture IDs
        by combining the season and fixture number.

        Parameters:
        df (DataFrame): The input DataFrame containing match fixture data with 'season' and 'fixture' columns.

        Returns:
        DataFrame: A modified DataFrame with an additional 'fixture_id' column that represents unique fixture IDs.
        """
        df['season_temp'] = df['season'].apply(lambda x: x.split('-')[0][-2:] + x.split('-')[1]).astype(str)
        df['fixture_string'] = df['fixture'].astype(str)
        df['fixture_id'] = df['season_temp'] + df['fixture_string']
        df = df.drop(columns = ['fixture_string', 'season_temp'])
        return df
    
    def _name_mapping(self, df):
        '''
        Only a temporary fix, need to figure out longer term string matching solution
        '''
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
    
    def _kickout_time_conversion(self, df):
        df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])
        return df


    def get_preprocessed_data(self):
        return self.data
