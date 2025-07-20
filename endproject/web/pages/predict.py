import streamlit as st
import pandas as pd
import glob

st.markdown("<h1 style='text-align: center;'>Predict Coin Price and Investment Prediction</h1>", unsafe_allow_html=True)

data = '../data/coin_*.csv'
csv_files = glob.glob(data)
dfs = [pd.read_csv(f) for f in csv_files]
all_coins = pd.concat(dfs, ignore_index=True)

all_coins = pd.DataFrame(all_coins)
adfd = pd.read_csv('../coin_history.csv', on_bad_lines='skip')

asd = list(all_coins['Name'])
asd = list(dict.fromkeys(asd))
asd.insert(0,'')

coin_invest = 0



from sklearn.model_selection import train_test_split
X = all_coins[['High', 'Low', 'Open', 'Close', 'Volume', 'Marketcap']].values
y = all_coins['Name'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from sklearn.preprocessing import LabelEncoder

    # Encode labels
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)
num_classes = len(le.classes_)

    # Build CNN model for classification
model = Sequential([
        
        Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
st.title('Train Model first')

c_btn = st.button('Train Model')
if c_btn:
    st.write("Training the model, please wait...")
    # model.summary()
    epochs = 10
    batch_size = 32
    # history = model.fit(
    #     X_train, y_train_enc,
    #     epochs=epochs,
    #     batch_size=batch_size,
    #     validation_data=(X_test, y_test_enc)
    # )

    with st.spinner('Training model...'):
            history = model.fit(
                X_train, y_train_enc,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test_enc)
            )
    st.success('Model predict completed!')

st.title('Predict Coin Price')
a = st.selectbox('Select a coin to predict its price', asd)
coin_input = st.number_input('Enter your quantity ', min_value=0, value=0, step=1000)
b = st.button('Test')
st.write(f'You selected: {a}')
# train model cnn
if a != '' and b:
    
    max_close_prices1 = all_coins.groupby('Name')['Close'].max()

    # Filter test samples for the selected coin
    coin_idx = (y_test == a)  # Replace 'AveCoin' with the desired coin name
    X_coin = X_test[coin_idx]

    # Predict class probabilities for the selected coin samples
    coin_pred_probs = model.predict(X_coin)
    coin_pred_classes = coin_pred_probs.argmax(axis=1)
    coin_pred_labels = le.inverse_transform(coin_pred_classes)

    # Show the corresponding Close prices for the selected coin test samples
    coin_close_prices = X_coin[:, 3]
    # Convert Close prices to USD format and print
    coin_close_prices_usd = ["{:,.2f}".format(price) for price in coin_close_prices]
    st.write(f"Close prices for {a} test samples (USD):", "$",max(coin_close_prices_usd))


    import numpy as np
    price_diff_pred = np.diff(coin_close_prices)
    trend_pred = np.where(price_diff_pred > 0, 1, np.where(price_diff_pred < 0, -1, 0))
    num_up_pred = np.sum(trend_pred == 1)
    num_down_pred = np.sum(trend_pred == -1)
    num_same_pred = np.sum(trend_pred == 0)
    if price_diff_pred[-1] > 0:
        st.markdown("<span style='color:green; font-size:23px'>UP</span>", unsafe_allow_html=True)
    elif price_diff_pred[-1] < 0:
        st.markdown("<span style='color:red; font-size:23px'>DOWN</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='color:white; font-size:23px'>HOLD</span>", unsafe_allow_html=True)
        
    # import matplotlib.pyplot as plt

    # fig, ax = plt.subplots(figsize=(14, 6))
    # ax.plot(range(len(coin_close_prices)), coin_close_prices, label='Predict Close Price')
    # ax.set_title(f'Predicted {a} Close Prices (Test Samples)')
    # ax.set_xlabel('Sample Index')
    # ax.set_ylabel('Close Price (USD)')
    # ax.legend()
    # ax.grid(True)

    # st.pyplot(fig)
    
    st.line_chart(pd.DataFrame(coin_close_prices, columns=['Close Price']))
    

    # max_price_predict = float(max(coin_close_prices_usd))
    max_price_predict = max((price.replace(',', '')) for price in coin_close_prices_usd)
    max_price_predict = float(max_price_predict)

    coin_invest = max_price_predict
    
    # Calculate the max close price for each coin in the dataset
    max_close_prices = all_coins.groupby('Name')['Close'].max()

    # Display the max close price for the selected coin
    
    st.write(f"Max close price for {a}: ${max_close_prices[a]:.2f}")
    
    coin_max = max_close_prices[a]
    
    coin_amount = coin_input * coin_max
    # st.write("You can buy", coin_amount, "coins of", a, "with your investment amount of $", coin_input)
    st.write(f"With {coin_input} units of {a}, the current total value at the max close price (${coin_max:.2f}) is: ${coin_amount:.2f}")

    coin_amount_predict = coin_input * coin_invest
    st.write(f"With {coin_input} units of {a}, the predicted future total value at the max close price (${coin_invest:.2f}) is: ${coin_amount_predict:.2f}")
    
    
    earn = coin_amount_predict - coin_amount
    if earn < 0:
        st.markdown(f"<span style='color:red; font-size:23px'>You can lose ${abs(earn):.2f} in the future</span>", unsafe_allow_html=True)
    elif earn > 0:
        
        st.markdown(f"<span style='color:green; font-size:23px'>You can earn ${earn:.2f} in the future</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='color:white; font-size:23px'>You will not earn or lose money in the future</span>", unsafe_allow_html=True)
# predict if user input a coin name and input thier investment amount how much they can earn

# Calculate the max close price for each coin in the dataset
# Calculate and display the max close price for the selected coin

# st.write(coin_invest)
# Set coin_invest1 to the predicted max close price for the selected coin
