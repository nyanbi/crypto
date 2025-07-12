import streamlit as st
import pandas as pd
import glob






data = '../data/coin_*.csv'
csv_files = glob.glob(data)
dfs = [pd.read_csv(f) for f in csv_files]
all_coins = pd.concat(dfs, ignore_index=True)
all_coins = pd.DataFrame(all_coins)
# Center the text using HTML and Streamlit's markdown with unsafe_allow_html
st.markdown(
    """
    <div style="text-align: center;">
        <h1>WebApp AITradePredict</h1>
        <h3>All Coins Data</h3>
        <p>This page displays all the coins data from the CSV files.</p>
    </div>
    """,
    unsafe_allow_html=True
)
st.write(all_coins)

st.title('Leaderboard of All Coin Prices')

# Ensure 'Date' column is in datetime format
all_coins['Date'] = pd.to_datetime(all_coins['Date'])

# Get the latest date available in the data
latest_date = all_coins['Date'].max()
latest_data = all_coins[all_coins['Date'] == latest_date]

# Select relevant columns and sort by Close, High, Low
leaderboard = latest_data[['Name', 'Close', 'High', 'Low']].copy()
leaderboard = leaderboard.sort_values(by=['Close', 'High', 'Low'], ascending=[False, False, False])

# Reset index for display
leaderboard = leaderboard.reset_index(drop=True)

st.dataframe(leaderboard)

st.title('what can we do?')
st.write('''
- Predict the price of a coin
- Show the chart of a coin
- Chat with the bot
''')