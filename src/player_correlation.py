import pandas as pd
import plotly.express as px

# Function to compare correlation of overall points between two players
def player_correlation (df, player1, player2, correlation_attribute):
    '''Calculates correlation coefficient between two players for a certain attribute and generates a scatter plot with a trend line for visualisation'''
    df = df.copy()
    df_player1 = df[df['name_player']==player1][[correlation_attribute, 'fixture_id_player']]
    df_player2 = df[df['name_player']==player2][[correlation_attribute, 'fixture_id_player']]
    
    merged_df = pd.merge(df_player1, df_player2, on='fixture_id_player', suffixes=('_' + player1,'_' + player2))
    
    correlation = merged_df[correlation_attribute + '_' + player1].corr(merged_df[correlation_attribute + '_' + player2])
    fig = px.scatter(merged_df, merged_df[correlation_attribute + '_' + player1], merged_df[correlation_attribute + '_' + player2], trendline='ols')
    fig.show()
    
    return correlation, fig