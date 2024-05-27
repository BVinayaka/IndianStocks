import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots

app = Flask(__name__)

def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df.reset_index(inplace=True)
    return df

# Supertrend Calculation
def calculate_supertrend(df, period=7, multiplier=4):
    df['TR'] = df[['High', 'Low', 'Close']].apply(
        lambda x: max(x['High'] - x['Low'], abs(x['High'] - x['Close']), abs(x['Low'] - x['Close'])),
        axis=1
    )
    df['ATR'] = df['TR'].rolling(window=period).mean()
    df['Basic Upper Band'] = (df['High'] + df['Low']) / 2 + multiplier * df['ATR']
    df['Basic Lower Band'] = (df['High'] + df['Low']) / 2 - multiplier * df['ATR']
    df['Final Upper Band'] = df['Basic Upper Band']
    df['Final Lower Band'] = df['Basic Lower Band']
    
    for i in range(1, len(df)):
        if df['Close'].iloc[i - 1] <= df['Final Upper Band'].iloc[i - 1]:
            df.at[i, 'Final Upper Band'] = min(df['Basic Upper Band'].iloc[i], df['Final Upper Band'].iloc[i - 1])
        else:
            df.at[i, 'Final Upper Band'] = df['Basic Upper Band'].iloc[i]
            
        if df['Close'].iloc[i - 1] >= df['Final Lower Band'].iloc[i - 1]:
            df.at[i, 'Final Lower Band'] = max(df['Basic Lower Band'].iloc[i], df['Final Lower Band'].iloc[i - 1])
        else:
            df.at[i, 'Final Lower Band'] = df['Basic Lower Band'].iloc[i]
    
    df['Supertrend'] = 0.0
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Final Upper Band'].iloc[i - 1]:
            df.at[i, 'Supertrend'] = df['Final Lower Band'].iloc[i]
        elif df['Close'].iloc[i] < df['Final Lower Band'].iloc[i - 1]:
            df.at[i, 'Supertrend'] = df['Final Upper Band'].iloc[i]
        else:
            df.at[i, 'Supertrend'] = df['Supertrend'].iloc[i - 1]
            
    return df

# Prepare the Data for Machine Learning
def prepare_data(df):
    df['Signal'] = 0
    df.loc[df['Close'] > df['Supertrend'], 'Signal'] = 1
    df.loc[df['Close'] < df['Supertrend'], 'Signal'] = -1
    df.dropna(inplace=True)
    X = df[['Close', 'High', 'Low', 'ATR']].values
    y = df['Signal'].values
    return X, y

# Lorentzian Classifier
class LorentzianClassifier:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.scaler = StandardScaler()
    
    def fit(self, X, y):
        # Feature Scaling
        X = self.scaler.fit_transform(X)
        
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        
        for epoch in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)
            errors = y - y_predicted
            
            d_weights = -(1 / X.shape[0]) * np.dot(X.T, errors * self._lorentzian_derivative(y_predicted))
            d_bias = -(1 / X.shape[0]) * np.sum(errors * self._lorentzian_derivative(y_predicted))
            
            self.weights -= self.learning_rate * d_weights
            self.bias -= self.learning_rate * d_bias
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Weights: {self.weights}, Bias: {self.bias}")
    
    def predict(self, X):
        # Feature Scaling
        X = self.scaler.transform(X)
        
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        return np.where(y_predicted >= 0.5, 1, -1)
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def _lorentzian_derivative(self, y):
        return 1 / (1 + y**2)

def plot_results(df, y_pred, title):
    df['Prediction'] = 0
    df['Prediction'].iloc[-len(y_pred):] = y_pred

    buy_signals = df[df['Prediction'] == 1]
    sell_signals = df[df['Prediction'] == -1]
    
    fig = make_subplots(rows=1, cols=1)

    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Candlestick'
    ))

    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Supertrend'],
        mode='lines',
        name='Supertrend',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=buy_signals['Date'],
        y=buy_signals['Close'],
        mode='markers',
        name='Buy Signal',
        marker=dict(color='green', symbol='triangle-up', size=10)
    ))

    fig.add_trace(go.Scatter(
        x=sell_signals['Date'],
        y=sell_signals['Close'],
        mode='markers',
        name='Sell Signal',
        marker=dict(color='red', symbol='triangle-down', size=10)
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False
    )

    return fig.to_json()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_data', methods=['POST'])
def get_data():
    data = request.json
    ticker = data['ticker']
    start_date = data['start_date']
    end_date = datetime.today().strftime('%Y-%m-%d')

    df = fetch_data(ticker, start_date, end_date)
    df = calculate_supertrend(df)
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    log_reg_model = LogisticRegression()
    log_reg_model.fit(X_train, y_train)
    lorentz_model = LorentzianClassifier(learning_rate=0.01, epochs=1000)
    lorentz_model.fit(X_train, y_train)

    y_pred_log_reg = log_reg_model.predict(X_test)
    y_pred_lorentz = lorentz_model.predict(X_test)

    log_reg_accuracy = accuracy_score(y_test, y_pred_log_reg)
    lorentz_accuracy = accuracy_score(y_test, y_pred_lorentz)

    # Plotting results
    plot1 = plot_results(df, y_pred_log_reg, "Logistic Regression Buy/Sell Signals")
    plot2 = plot_results(df, y_pred_lorentz, "Lorentzian Classifier Buy/Sell Signals")

    return jsonify({
        'log_reg_accuracy': log_reg_accuracy,
        'lorentz_accuracy': lorentz_accuracy,
        'plot1': plot1,
        'plot2': plot2
    })

if __name__ == '__main__':
    app.run(debug=True)
