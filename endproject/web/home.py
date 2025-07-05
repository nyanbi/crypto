import streamlit as st
import pandas as pd
import glob



st.title("WebApp AITradePredict")


data = '../data/coin_*.csv'
csv_files = glob.glob(data)
dfs = [pd.read_csv(f) for f in csv_files]
all_coins = pd.concat(dfs, ignore_index=True)
all_coins = pd.DataFrame(all_coins)
st.write("All Coins Data")
st.write("This page displays all the coins data from the CSV files.")
st.write(all_coins)

st.title('what can we do?')
st.write('''
- Predict the price of a coin
- Show the chart of a coin
- Chat with the bot
''')