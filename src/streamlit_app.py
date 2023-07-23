import streamlit as st
import plotly.express as px
import pandas as pd

# load the data
final_df = pd.read_csv('../data/processed_data/played_df.csv')

# define the dropdown menus
players = final_df['name_player'].unique()
selected_player = st.selectbox('Select player:', players)
seasons = st.multiselect('Select season: ', final_df['season'].unique())

# filter the data
selected_seasons = seasons if seasons else final_df['season'].unique()
filtered_df = final_df[(final_df['name_player'] == selected_player) & (final_df['season'].isin(selected_seasons))]

# create the scatter plot
fig = px.scatter(filtered_df, x='player_season_fixture_number', y='value_player', color='season')

# show the plot
st.plotly_chart(fig)
