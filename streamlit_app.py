import streamlit as st
import pandas as pd
import requests
from dateutil.relativedelta import relativedelta
from datetime import datetime
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

# GitHub repository information
username = 'sahil-gidwani'
repository = 'Stock-Price-Prediction-LSTM'
folder_path = '/data'  # Relative path within the repository

# Get a list of files in the folder from the GitHub API
api_url = f'https://api.github.com/repos/{username}/{repository}/contents/{folder_path}'
response = requests.get(api_url)
data = response.json()

stock_names = []

# Loop through the files and extract CSV URLs and names
for item in data:
    if item['type'] == 'file' and item['name'].endswith('.csv'):
        stock_names.append(item['name'])

# Define a function to load your CSV data for a selected stock
def load_selected_stock_data(selected_stock_url):
    # Read the selected stock's data into a DataFrame
    selected_df = pd.read_csv(selected_stock_url, index_col=0)
    
    # Convert the index to datetime objects
    selected_df.index = pd.to_datetime(selected_df.index)
    
    return selected_df

# Define a function to build an LSTM model
def build_lstm_model(x_train, y_train):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    return model

# Create a Streamlit app
st.title("Stock Price Prediction App")

# Sidebar with stock selection
selected_stock = st.sidebar.selectbox("Select a Stock", stock_names)

# Get the selected stock's CSV URL
selected_stock_url = None
for item in data:
    if item['name'] == selected_stock:
        selected_stock_url = item['download_url']
        break

# Check if the selected stock was found
if selected_stock_url is not None:
    # Load the selected stock's data
    selected_df = load_selected_stock_data(selected_stock_url)
    
    # Display selected stock data
    st.write(f"Selected Stock: {selected_stock}")
    st.dataframe(selected_df)

    # Date range selection
    st.title("Select Date Range")
    today = datetime.today()
    one_year_ago = today - relativedelta(years=1)
    date_range = st.date_input("Select a start date", one_year_ago), st.date_input("Select an end date", today)

    # Convert date range to datetime objects
    start_date, end_date = map(lambda x: datetime.strptime(str(x), '%Y-%m-%d'), date_range)

    # Filter DataFrame based on selected date range
    selected_df_filtered = selected_df[(selected_df.index >= start_date) & (selected_df.index <= end_date)]

    # Line chart for the selected date range
    st.title("Stock Price Line Chart")
    st.line_chart(selected_df_filtered[["Adj Close", "Open"]])

    # Bar chart for volume
    st.title("Stock Volume Bar Chart")
    st.bar_chart(selected_df_filtered["Volume"])

    # Split the data and build the LSTM model
    train_data = selected_df_filtered['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(train_data)

    x_train = []
    y_train = []
    for i in range(60, len(scaled_data)):
        x_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    lstm_model = build_lstm_model(x_train, y_train)

    # Button to predict stock price
    if st.button("Predict Stock Price"):
        # Prepare input data for prediction
        # Prepare input data for prediction
        x_test = []
        y_test = []

        test_data_start = len(selected_df_filtered) - 60
        test_data_end = len(selected_df_filtered)

        test_data = scaled_data[test_data_start:test_data_end, :]
        for i in range(60, len(test_data)):
            x_test.append(test_data[i - 60:i, 0])
            y_test.append(test_data[i, 0])

        x_test = np.array(x_test)
        # Reshape x_test to have dimensions (batch_size, timesteps, features)
        # Reshape x_test to have dimensions (1, x_test.shape[0], x_test.shape[1])
        x_test = np.reshape(x_test, (1, x_test.shape[0], x_test.shape[1]))

        # Make predictions
        predictions = lstm_model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        # Create a DataFrame for predictions
        predicted_df = pd.DataFrame({'Predicted Close': predictions[:, 0]})

        # Display the predictions
        st.title("Stock Price Predictions")
        st.dataframe(predicted_df)
else:
    st.write("Selected stock data not found.")
