import streamlit as st
import pandas as pd
import glob
st.title('Chart page')

data = '../data/coin_*.csv'
csv_files = glob.glob(data)
dfs = [pd.read_csv(f) for f in csv_files]
all_coins = pd.concat(dfs, ignore_index=True)

all_coins = pd.DataFrame(all_coins)
asd = list(all_coins['Name'])
asd = list(dict.fromkeys(asd))
asd.insert(0,'')

st.title('Select Coin and Chart Type')
selected_coins = st.multiselect('Select coins to compare', asd[1:])  # Exclude empty string
b = st.selectbox('Select a chart type', ['Line Chart', 'Bar Chart', 'Area Chart'])
c = st.selectbox('Select a type of price', ['Open', 'High', 'Low', 'Close'])
st.write(f'You selected: {selected_coins} with chart type: {b} and price type: {c}')


st.title('Chart of Coin Price')


if selected_coins:
    combined_data = all_coins[all_coins['Name'].isin(selected_coins)]
    combined_data = combined_data.sort_values(['Name', 'Date'])
    chart_df = pd.DataFrame(combined_data.pivot(index='Date', columns='Name', values=c))
    if b == 'Line Chart':
        st.line_chart(chart_df, use_container_width=True)
    elif b == 'Bar Chart':
        st.bar_chart(chart_df, use_container_width=True)
    elif b == 'Area Chart':
        st.area_chart(chart_df, use_container_width=True)



