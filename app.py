#libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import streamlit as st

from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential




start = '2012-01-01'
end = '2022-06-01'
#stock = 'GOOG'


import streamlit as st


st.title("Stock Trend Prediction")
st.markdown("<style>h1{color: Orange;}</style>", unsafe_allow_html=True)
st.markdown("<style>input{color: grey !important;}</style>", unsafe_allow_html=True)

# Create the text input field
stock = st.text_input('Enter stock ticker', 'GOOG', key='text-input')
data = yf.download(stock, start, end)



st.markdown('<h2 style="color: orange;">Fetched Data</h2>', unsafe_allow_html=True)
st.write(data.head())


#preprocessing
print("Null values in  dataset:")
print(data.isnull().sum())

# Remove rows with null values
data.dropna(inplace=True)

# Check for and remove duplicate rows
duplicate_rows = data[data.duplicated()]
if not duplicate_rows.empty:
    print("Duplicate rows found and removed.")
    data.drop_duplicates(inplace=True)
else:
    print("No duplicate rows found.")

    

# Calculate z-score for 'Close' column
data['Close_zscore'] = zscore(data['Close'])
threshold = 3
data_no_outliers = data[np.abs(data['Close_zscore']) < threshold]
data_no_outliers = data_no_outliers.drop(columns=['Close_zscore'])

# Print the shape of the original and filtered data
print("Original data shape:", data.shape)
print("Filtered data shape:", data_no_outliers.shape)

def create_dataset(data, time_steps=1):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), 0])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)

# Scale the data using MinMaxScaler with the same feature range(Not needed)
#scaler = MinMaxScaler(feature_range=(0, 1))
#scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

train_size = int(len(data) * 0.10)
test_size = len(data) - train_size
train_data, test_data = data[:train_size], data[train_size:]


train_scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled_data = train_scaler.fit_transform(train_data['Close'].values.reshape(-1, 1))


X_train_scaled, y_train_scaled = create_dataset(train_scaled_data)

 
model_ann = Sequential()
model_ann.add(Dense(units=64, activation='relu', input_shape=(X_train_scaled.shape[1],)))  # Input shape directly specified
model_ann.add(Dense(units=64, activation='relu'))
model_ann.add(Dense(units=1))

# Compile the ANN model
model_ann.compile(optimizer='adam', loss='mean_squared_error')
model_ann.fit(X_train_scaled, y_train_scaled, epochs=100, batch_size=32, verbose=1)
test_scaled_data = train_scaler.transform(test_data['Close'].values.reshape(-1, 1))
X_test_scaled, y_test_scaled = create_dataset(test_scaled_data)

# Predict using ANN
y_pred_ann = model_ann.predict(X_test_scaled)
y_pred_ann = train_scaler.inverse_transform(y_pred_ann.reshape(-1, 1))  # Reshape the prediction
y_test_ann = train_scaler.inverse_transform(y_test_scaled.reshape(-1, 1))  # Reshape the true values  

import plotly.graph_objs as go

# Ensure 'Date' column is set as index
#data = data.set_index('Date')


trace_actual_ann = go.Scatter(x=data.index, y=y_test_ann.flatten(), mode='lines', name='Actual', line=dict(color='blue'))
trace_predicted_ann = go.Scatter(x=data.index, y=y_pred_ann.flatten(), mode='lines', name='Predicted (ANN)', line=dict(color='orange'))

# Create Plotly figure for ANN plot
fig_ann = go.Figure(data=[trace_actual_ann, trace_predicted_ann])
fig_ann.update_layout(
    title='Actual vs Predicted Prices (ANN)',
    title_font_color='orange',
    xaxis=dict(title='Time'),
    yaxis=dict(title='Price'),
    hovermode='x unified'
)

# Show ANN plot
#st.pyplot(fig_ann)
st.plotly_chart(fig_ann)

import datetime

#data&time 
start_date = pd.to_datetime(start)
end_date = pd.to_datetime(end)
st.markdown('<h2 style="color: orange;">Fetch by date</h2>', unsafe_allow_html=True)

prediction_date = st.date_input("Enter a date for prediction", min_value=start_date.date(), max_value=end_date.date(), value=None)

prediction_date = pd.Timestamp(prediction_date)

if start_date <= prediction_date <=end_date:
  
    predicted_price = y_pred_ann[data.index.get_loc(prediction_date)]
    st.write("Predicted Price on", prediction_date, ":", predicted_price)
else:
    st.write("Please enter a date between", start_date.date(), "and", end_date.date())


# Calculate ANN metrics
#mse_ann = mean_squared_error(y_test_ann, y_pred_ann)
#rmse_ann = np.sqrt(mse_ann)
#mae_ann = mean_absolute_error(y_test_ann, y_pred_ann)
r2_ann = r2_score(y_test_ann, y_pred_ann)

# Print ANN metrics
print("ANN Metrics:")
#print("Mean Squared Error (MSE):", mse_ann)
#print("Root Mean Squared Error (RMSE):", rmse_ann)
#print("Mean Absolute Error (MAE):", mae_ann)
print("Coefficient of Determination (R²):", r2_ann)





# Fetch additional stock metrics from Yahoo Finance API
info = yf.Ticker(stock).info
ticker = yf.Ticker(stock)

stock_info = ticker.info


outstanding_shares = stock_info.get('sharesOutstanding', 100000000)

# Calculate market capitalization
data['Market_Cap'] = data['Close'] * outstanding_shares
st.markdown('<h2 style="color: orange;">Analyze Market Capitalization</h2>', unsafe_allow_html=True)
st.line_chart(data['Market_Cap'])

st.markdown('<h2 style="color: orange;">Analyze Market Volume</h2>', unsafe_allow_html=True)
#Fetch historical volume data
volume_data = data['Volume']

# Plot historical volume
st.line_chart(volume_data)


st.markdown('<h2 style="color: orange;">When to invest🤔</h2>', unsafe_allow_html=True)

def fetch_market_cap_data(stock, start_date, end_date):
    ticker = yf.Ticker(stock)
    hist_data = ticker.history(start=start_date, end=end_date)
    outstanding_shares = ticker.info['sharesOutstanding']
    hist_data['Market_Cap'] = hist_data['Close'] * outstanding_shares
    
    return hist_data

# Function to suggest ideal investment time
def suggest_investment_time(data):
    max_market_cap_date = data['Market_Cap'].idxmax()
    max_market_cap_year = max_market_cap_date.year
    return max_market_cap_date, max_market_cap_year


try:
    market_cap_data = fetch_market_cap_data(stock, start, end)
    st.write("Historical Market Capitalization Data:")
    st.write(market_cap_data.head())

    # Suggest ideal investment time
    ideal_investment_date, ideal_investment_year = suggest_investment_time(market_cap_data)
    st.write("Ideal Investment Date:", ideal_investment_date)
    st.write("Ideal Investment Year:", ideal_investment_year)

except Exception as e:
    st.error("Error fetching data:", e)
