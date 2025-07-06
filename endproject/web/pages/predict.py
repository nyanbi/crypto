import streamlit as st
import pandas as pd
import glob

st.title('Predict Coin Price')

data = '../data/coin_*.csv'
csv_files = glob.glob(data)
dfs = [pd.read_csv(f) for f in csv_files]
all_coins = pd.concat(dfs, ignore_index=True)

all_coins = pd.DataFrame(all_coins)
adfd = pd.read_csv('../coin_history.csv', on_bad_lines='skip')

asd = list(all_coins['Name'])
asd = list(dict.fromkeys(asd))
asd.insert(0,'')

a = st.selectbox('Select a coin to predict its price', asd)
b = st.button('Predict')
st.write(f'You selected: {a}')


# train model cnn
if a != '' and b:
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
    coin_close_prices_usd = ["${:,.2f}".format(price) for price in coin_close_prices]
    st.write(f"Close prices for {a} test samples (USD):", max(coin_close_prices_usd))


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
