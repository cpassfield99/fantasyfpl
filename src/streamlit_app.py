import streamlit as st
import plotly.express as px
import pandas as pd
from player_correlation import player_correlation

# load the data
data = pd.read_csv('data/processed_data/complete_df.csv')

# # define the dropdown menus
# players = final_df['name_player'].unique()
# selected_player = st.selectbox('Select player:', players)
# seasons = st.multiselect('Select season: ', final_df['season'].unique())

# # filter the data
# selected_seasons = seasons if seasons else final_df['season'].unique()
# filtered_df = final_df[(final_df['name_player'] == selected_player) & (final_df['season'].isin(selected_seasons))]

# # create the scatter plot
# fig = px.scatter(filtered_df, x='player_season_fixture_number', y='value_player', color='season')
st.set_option('client.showErrorDetails', False)

players = data['name_player'].unique()
player1 = st.selectbox('Select Player 1 for Correlation:', ['', *players])
player2 = st.selectbox('Select Player 2 for Correlation:', ['', *players])

numeric_columns = []
for column in data.columns:
    if pd.api.types.is_numeric_dtype(data[column]):
        numeric_columns.append(column)

correlation_attribute = st.selectbox('Select the attribute you want to assess correlation of:', numeric_columns)

correlation_score, correlation_plot = player_correlation(data, player1, player2, correlation_attribute)

st.subheader("Correlation Information:")
st.write(f"The correlation between {player1} and {player2} for {correlation_attribute} is {correlation_score:.2f}.")


st.plotly_chart(correlation_plot)
# player_correlation(df)

# show the plot
#st.plotly_chart(fig)
