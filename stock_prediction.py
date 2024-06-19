import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Define the stock ticker and the period for historical data
ticker = "TSLA"
start_date = "2015-01-01"
end_date = "2024-01-01"

# Download the historical stock data
data = yf.download(ticker, start=start_date, end=end_date)

# Feature engineering: add moving averages and other technical indicators
data['Date'] = data.index
data['Date'] = pd.to_datetime(data['Date'])
data['Date'] = data['Date'].map(pd.Timestamp.timestamp)

# Moving averages
data['MA10'] = data['Close'].rolling(window=10).mean()
data['MA50'] = data['Close'].rolling(window=50).mean()
data['MA200'] = data['Close'].rolling(window=200).mean()

# Momentum
data['Momentum'] = data['Close'] / data['Close'].shift(5)

# Volatility
data['Volatility'] = data['Close'].rolling(window=10).std()

# Drop rows with NaN values
data = data.dropna()

# Features and target variable
X = data[['Date', 'MA10', 'MA50', 'MA200', 'Momentum', 'Volatility']]
y = data['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model with hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Calculate Mean Squared Error and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Plot the actual vs predicted stock prices
plt.figure(figsize=(10, 6))
plt.plot(X_test.index, y_test, color='blue', label='Actual Prices')
plt.plot(X_test.index, y_pred, color='red', linestyle='--', label='Predicted Prices')
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
