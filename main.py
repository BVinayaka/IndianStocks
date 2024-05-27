from curses import flash
import string
import bcrypt
import numpy as np
from pandas_datareader import data as pdr
from datetime import datetime, timedelta
from pymongo import MongoClient
import requests
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from flask import Flask, redirect, render_template, jsonify, request, session
from flask_cors import CORS
import json
from flask_socketio import SocketIO, emit 
import threading
import time
from flask import Flask, render_template, jsonify, request
from yaml import emit
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objs as go 
import plotly.io as pio
from flask_cors import CORS
from yahoo_fin.stock_info import get_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
socketio = SocketIO(app, mode='threading')
CORS(app)
import subprocess
headers = {
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiYjk0NzQyNTctOWY4Yy00Y2I0LWE5Y2UtMjNhM2Q5ZDM4ZTRhIiwidHlwZSI6ImFwaV90b2tlbiJ9.Hc9nYSzzt5thtOWuWq5Kr14wD0nixZdO7ODBheS3lLE"
}

app.secret_key = ''
url = "https://api.edenai.run/v2/text/question_answer"
client = MongoClient('mongodb://localhost:27017')
db = client['STOCK']
users_collection = db['users']   # Collection for session management


def fetch_historical_data(symbol, period="7d", interval="1m"):
    data = yf.download(tickers=symbol, period=period, interval=interval)
    return data



def calculate_indicator(data, indicator='', length=14):
    if indicator == 'SMA':
        data[indicator] = ta.sma(data['Close'], length=length)
    if indicator == 'RSI':
        data[indicator] = ta.rsi(data['Close'], length=length)
    if indicator == 'EMA':
        data[indicator] = ta.ema(data['Close'], length=length)


def plot_indicator(data, indicator, model='linear', filename='plot.png', symbol=''):
    history = data[indicator].tail(100).reset_index()
    history['index'] = history.index

    if model == 'linear':
        reg = LinearRegression()
    elif model == 'svr':
        reg = SVR(kernel='rbf', C=1e3, gamma=0.1)
        scaler = StandardScaler()
        history[indicator] = scaler.fit_transform(history[[indicator]])

    reg.fit(history[['index']], history[indicator])
    history[f'{model}_pred'] = reg.predict(history[['index']])

    sns.set(style="whitegrid")
    plt.figure(figsize=(20, 16))
    sns.lineplot(x='index', y=indicator, data=history)
    sns.lineplot(x='index', y=f'{model}_pred', data=history, color='red', label=f'{model} regression')
    plt.title(f"{indicator} with {model.capitalize()} Regression for {symbol}")
    plt.xlabel("Date and Time")
    plt.ylabel(indicator)
    plt.xticks(np.arange(0, len(history), 5), history['Datetime'].iloc[::5].dt.strftime('%a.- (%d- %b- %Y) %H:%M'), rotation=45)
    plt.legend()

    plt.savefig(filename)
    plt.close()

    return filename


@app.route('/technical', methods=['GET'])
def plot_technical():
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({"error": "Symbol parameter is required"}), 400

    data = fetch_historical_data(symbol)
    if data.empty:
        return jsonify({"error": "No data available for the given symbol"}), 404
    
    calculate_indicator(data, 'RSI')
    calculate_indicator(data, 'SMA')
    calculate_indicator(data, 'EMA')
    plot_files = [
        plot_indicator(data, 'SMA', filename=f'static/{symbol}_SMA_plot.png', symbol=symbol),
        plot_indicator(data, 'RSI', filename=f'static/{symbol}_RSI_plot.png', symbol=symbol),
        plot_indicator(data, 'EMA', filename=f'static/{symbol}_EMA_plot.png', symbol=symbol)
    ]
    # Return the plot file paths as JSON
    return jsonify(SMA_plot=plot_files[0], RSI_plot=plot_files[1],EMA_plot=plot_files[2])


@app.route('/')
def index():
    return render_template('index.html')




@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['username']
        email = request.form['email']
        mobile = request.form['mobile']
        password1 = request.form['password']
        password2 = request.form['confirm_password']

        # Check if passwords match
        if password1 != password2:
            return render_template('signup.html', message='Passwords do not match')

        # Check if password meets criteria
        if not (len(password1) >= 8 and any(c.isupper() for c in password1)
                and any(c.islower() for c in password1) and any(c.isdigit() for c in password1)
                and any(c in string.punctuation for c in password1)):
            return render_template('signup.html', message='Password criteria not met')

        hashed_password = bcrypt.hashpw(password1.encode('utf-8'), bcrypt.gensalt())

        # Check if email already exists
        existing_user = users_collection.find_one({'email': email})
        if existing_user:
            return render_template('signup.html', message='Email already exists. If you already have an account, please log in.')

        # Insert new user into users_collection
        user_data = {
            'name': name,
            'email': email,
            'mobile': mobile,
            'password': hashed_password
        }
        users_collection.insert_one(user_data)

        return redirect('/')

    return render_template('signup.html')


@app.route('/logout')
def logout():
    # Remove user from session
    session.pop('user', None)
    
    return redirect('/')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Find user by email
        user = users_collection.find_one({'email': email})

        if user:
            # Convert ObjectId to string
            user['_id'] = str(user['_id'])

            # Check if password matches
            if bcrypt.checkpw(password.encode('utf-8'), user['password']):
                # Store user data in session
                session['user'] = user
                return redirect('/')
            else:
                return render_template('login.html', message='Incorrect password')
        else:
            return render_template('login.html', message='User not found')

    return render_template('login.html')

def get_exchange_rate():
    # Use an appropriate exchange rate API
    api_url = "https://open.er-api.com/v6/latest/USD"
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        return data['rates']['INR']
    except requests.RequestException as e:
        raise ValueError("Error fetching exchange rate") from e

def convert_usd_to_inr(usd_value, exchange_rate):
    if isinstance(usd_value, (int, float)):
        return f"₹{round(usd_value * exchange_rate, 2):,.2f}"
    return usd_value

@app.route('/about', methods=['GET'])
def get_about_info():
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({"error": "Symbol parameter is required"}), 400

    try:
        exchange_rate = get_exchange_rate()

        # Fetch stock information using yfinance
        ticker = yf.Ticker(symbol)
        info = ticker.info

        # Extract required fields from the info dictionary
        about_info = {
            "Name": info.get('longName', ''),
            "Symbol": info.get('symbol', ''),
            "Description": info.get('longBusinessSummary', ''),
            "AssetType": info.get('quoteType', ''),
            "Exchange": info.get('exchange', ''),
            "Industry": info.get('industry', ''),
            "Country": info.get('country', ''),
            "Currency": "INR",
            "MarketCapitalization": convert_usd_to_inr(info.get('marketCap', ''), exchange_rate),
            "SharesOutstanding": info.get('sharesOutstanding', ''),
            "CIK": info.get('CIK', ''),
            "Sector": info.get('sector', ''),
            "Address": info.get('address1', '') + ', ' + info.get('city', '') + ', ' + info.get('state', '') + ', ' + info.get('zip', ''),
            "FiscalYearEnd": info.get('lastFiscalYearEnd', ''),
            "LatestQuarter": info.get('latestQuarter', ''),
            "EBITDA": convert_usd_to_inr(info.get('EBITDA', ''), exchange_rate),
            "PERatio": info.get('trailingPE', ''),
            "PEGRatio": info.get('pegRatio', ''),
            "BookValue": convert_usd_to_inr(info.get('bookValue', ''), exchange_rate),
            "DividendPerShare": convert_usd_to_inr(info.get('dividendRate', ''), exchange_rate),
            "DividendYield": info.get('dividendYield', ''),
            "EPS": convert_usd_to_inr(info.get('trailingEps', ''), exchange_rate),
            "RevenuePerShareTTM": convert_usd_to_inr(info.get('revenuePerShare', ''), exchange_rate),
            "ProfitMargin": info.get('profitMargins', ''),
            "OperatingMarginTTM": info.get('operatingMargins', ''),
            "ReturnOnAssetsTTM": info.get('returnOnAssets', ''),
            "ReturnOnEquityTTM": info.get('returnOnEquity', ''),
            "RevenueTTM": convert_usd_to_inr(info.get('totalRevenue', ''), exchange_rate),
            "GrossProfitTTM": convert_usd_to_inr(info.get('grossProfit', ''), exchange_rate),
            "DilutedEPSTTM": convert_usd_to_inr(info.get('dilutedEPS', ''), exchange_rate),
            "QuarterlyEarningsGrowthYOY": info.get('earningsQuarterlyGrowth', ''),
            "QuarterlyRevenueGrowthYOY": info.get('revenueGrowth', ''),
            "AnalystTargetPrice": convert_usd_to_inr(info.get('targetMeanPrice', ''), exchange_rate),
            "AnalystRatingStrongBuy": info.get('recommendationKey', '') == 'buy',
            "AnalystRatingBuy": info.get('recommendationKey', '') == 'overweight',
            "AnalystRatingHold": info.get('recommendationKey', '') == 'hold',
            "AnalystRatingSell": info.get('recommendationKey', '') == 'underweight',
            "AnalystRatingStrongSell": info.get('recommendationKey', '') == 'sell',
            "TrailingPE": info.get('trailingPE', ''),
            "ForwardPE": info.get('forwardPE', ''),
            "PriceToSalesRatioTTM": info.get('priceToSalesTrailing12Months', ''),
            "PriceToBookRatio": info.get('priceToBook', ''),
            "EVToRevenue": convert_usd_to_inr(info.get('enterpriseToRevenue', ''), exchange_rate),
            "EVToEBITDA": convert_usd_to_inr(info.get('enterpriseToEbitda', ''), exchange_rate),
            "Beta": info.get('beta', ''),
            "52WeekHigh": convert_usd_to_inr(info.get('fiftyTwoWeekHigh', ''), exchange_rate),
            "52WeekLow": convert_usd_to_inr(info.get('fiftyTwoWeekLow', ''), exchange_rate),
            "50DayMovingAverage": convert_usd_to_inr(info.get('fiftyDayAverage', ''), exchange_rate),
            "200DayMovingAverage": convert_usd_to_inr(info.get('twoHundredDayAverage', ''), exchange_rate),
            "DividendDate": info.get('dividendDate', ''),
            "ExDividendDate": info.get('exDividendDate', '')
        }

        return jsonify(about_info)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


def emit_live_data(symbol):
    while True:
        df = yf.download(tickers=symbol, period='1d', interval='1m')
        
        if not df.empty:
            fig = create_candlestick_chart(df, symbol)
            div = pio.to_json(fig)
            
            emit('update_live_data', {'div': div})
        
        time.sleep(30)  # Sleep for 30 seconds before sending the next update


@app.route('/live_data_stream', methods=['GET'])
def live_data_stream():
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({"error": "Stock symbol is missing."}), 400

    threading.Thread(target=emit_live_data, args=(symbol,)).start()
    
    return jsonify({"message": "Live data stream started."})


@app.route('/live_data', methods=['GET'])
def live_data():
    stock = request.args.get('symbol')
    counter = 0 

    if not stock:
        return jsonify({"error": "Stock symbol is missing."}), 400
    
    while counter < 10:  # Limit the loop to 10 seconds
        df = yf.download(tickers=stock, period='1d', interval='1m')
        
        if df.empty:
            return jsonify({"error": "No data available for the given symbol."}), 404

        fig = create_candlestick_chart(df, stock)
        div = pio.to_json(fig)

        time.sleep(1)  # Sleep for 30 seconds before fetching the next data
        counter += 10  # Increment the counter by 30 seconds

    return jsonify({"div": div})


def create_candlestick_chart(df, stock):
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'], name='market data'))

    fig.update_layout(
        title=f"{stock} Live Share Price:",
        yaxis_title='Stock Price (USD per Shares)')

    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=15, label="15m", step="minute", stepmode="backward"),
                dict(count=45, label="45m", step="minute", stepmode="backward"),
                dict(count=1, label="HTD", step="hour", stepmode="todate"),
                dict(count=3, label="3h", step="hour", stepmode="backward"),
                dict(step="all")
            ])
        )
    )

    return fig


@app.route('/run_code', methods=['POST', 'GET'])
def run_code():
    symbol = request.args.get('symbol')
    yf.pdr_override()
    start = pd.Timestamp.now() - pd.DateOffset(years=8)
    end = pd.Timestamp.now()
    df = yf.download(symbol, start=start, end=end)

    data = df.filter(['Close'])
    dataset = data.values
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    training_data_len = int(np.ceil(len(dataset) * .70))
    train_data = scaled_data[0:int(training_data_len), :]

    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    test_data = scaled_data[training_data_len - 60:, :]
    x_test = []
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions).flatten()

    valid = data[training_data_len:]
    
    # Ensure predictions align with valid data
    predictions = predictions[-len(valid):]

    volume_path, daily_return_path, adj_close_path, prediction_path = generate_plots(df, symbol, valid, predictions)

    valid_json = valid.to_json(orient='index')
    
    # Prepare the data for the table
    actual_data = valid['Close'].values.tolist()
    predicted_data = predictions.tolist()
    date_index = valid.index.strftime('%Y-%m-%d').tolist()

    return jsonify({
        "status": "success",
        "volume_plot": volume_path,
        "daily_return_histogram": daily_return_path,
        "adjusted_close_price": adj_close_path,
        "predicted_vs_actual_plot": prediction_path,
        "data": json.loads(valid_json),
        "table_data": {
            "date": date_index,
            "actual": actual_data,
            "predicted": predicted_data
        }
    })


def generate_plots(df, symbol, valid, predictions):
    # Plot volume
    plt.figure(figsize=(10, 6))
    df['Volume'].plot()
    plt.ylabel('Volume')
    plt.title('Volume')
    volume_path = f'static/{symbol}_volume_plot.png'
    plt.savefig(volume_path)
    plt.close()

    # Plot daily return histogram
    plt.figure(figsize=(10, 6))
    df['Daily Return'] = df['Adj Close'].pct_change()
    df['Daily Return'].hist(bins=50, color='green', alpha=0.7)
    plt.xlabel('Daily Return')
    plt.title('Daily Return Histogram')
    daily_return_path = f'static/{symbol}_daily_return_histogram.png'
    plt.savefig(daily_return_path)
    plt.close()

    # Plot adjusted close price with moving averages
    plt.figure(figsize=(10, 6))
    
    # Plot actual adjusted close price
    plt.plot(df.index, df['Adj Close'], label='Actual Close Price', color='black')
    
    # Calculate and plot moving averages
    df['MA10'] = df['Adj Close'].rolling(window=10).mean()
    df['MA20'] = df['Adj Close'].rolling(window=20).mean()
    df['MA50'] = df['Adj Close'].rolling(window=50).mean()
    
    plt.plot(df.index, df['MA10'], label='MA10', color='blue')
    plt.plot(df.index, df['MA20'], label='MA20', color='green')
    plt.plot(df.index, df['MA50'], label='MA50', color='orange')
    
    plt.ylabel('Adj Close')
    plt.title('Adjusted Close Price with Moving Averages')
    plt.legend()
    adj_close_path = f'static/{symbol}_adjusted_close_price.png'
    plt.savefig(adj_close_path)
    plt.close()

    # Plot for Predicted vs Actual
    plt.figure(figsize=(10, 6))
    plt.plot(valid.index, valid['Close'], label='Actual Close Price', color='green')
    plt.plot(valid.index, predictions, label='Predicted Close Price', color='red')
    plt.title('Predicted vs Actual Close Price')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    prediction_path = f'static/{symbol}_predicted_vs_actual.png'
    plt.savefig(prediction_path)
    plt.close()

    return volume_path, daily_return_path, adj_close_path, prediction_path


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('question')

    text1 = "i need information related to stocks"
    text2 = "please give me related to stocks only"

    payload = {
        "providers": "openai",
        "texts": [text1, text2],
        "question": question,
        "examples_context": "In 2017, U.S. life expectancy was 78.6 years.",
        "examples": [["What is human life expectancy in the United States?", "78 years."]],
        "fallback_providers": ""
    }

    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        answer = result['openai']['answers'][0]
    else:
        answer = "Sorry, something went wrong."

    return jsonify({'answer': answer})



@app.route('/topstock')
def get_top_stocks():
    strategy = request.args.get('strategy')
    
    current_date = datetime.now()
    dates = {
        '1 Day': (current_date - timedelta(days=3)).strftime('%Y-%m-%d'),
        '30 Days': (current_date - timedelta(days=30)).strftime('%Y-%m-%d'),
        '150 Days': (current_date - timedelta(days=150)).strftime('%Y-%m-%d')
    }

    indian_tickers = [
        'INFY.NS', 'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'HDFC.NS', 
        'ITC.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'LT.NS', 'AXISBANK.NS', 
        'SBIN.NS', 'BHARTIARTL.NS', 'ASIANPAINT.NS', 'HINDUNILVR.NS', 
        'MARUTI.NS', 'WIPRO.NS', 'BAJFINANCE.NS', 'ONGC.NS', 'SUNPHARMA.NS', 
        'TECHM.NS', 'NESTLEIND.NS', 'ULTRACEMCO.NS', 'IOC.NS', 'POWERGRID.NS', 
        'TITAN.NS', 'DRREDDY.NS', 'NTPC.NS', 'GAIL.NS', 'CIPLA.NS', 
        'GRASIM.NS', 'SHREECEM.NS', 'COALINDIA.NS', 'HCLTECH.NS', 'BAJAJFINSV.NS',
        'BAJAJ-AUTO.NS', 'JSWSTEEL.NS', 'BRITANNIA.NS', 'HEROMOTOCO.NS', 
        'INDUSINDBK.NS', 'ADANIPORTS.NS', 'UBL.NS', 'EICHERMOT.NS', 'BAJAJHLDNG.NS',
        'BPCL.NS', 'HINDALCO.NS', 'M&M.NS', 'VEDL.NS', 'HDFCLIFE.NS', 'MRF.NS',
        'DIVISLAB.NS', 'AUROPHARMA.NS', 'HDFCAMC.NS', 'BANDHANBNK.NS', 'DLF.NS',
        'YESBANK.NS', 'RAMCOCEM.NS', 'TATASTEEL.NS', 'CADILAHC.NS', 'PNB.NS',
        'LTI.NS', 'ADANIGREEN.NS', 'BIOCON.NS', 'UBL.NS', 'TORNTPOWER.NS'
    ]

    price_changes_data = {}
    for period, date in dates.items():
        hist_data = yf.download(indian_tickers, start=date, end=current_date.strftime('%Y-%m-%d'))['Close']
        
        price_changes = {}
        for ticker in hist_data.columns:
            if hist_data[ticker].dropna().size > 0:
                price_change = ((hist_data[ticker].iloc[-1] - hist_data[ticker].iloc[0]) / hist_data[ticker].iloc[0]) * 100
                price_changes[ticker] = price_change
        
        sorted_price_changes = dict(sorted(price_changes.items(), key=lambda item: item[1], reverse=True)[:5])
        price_changes_data[period] = sorted_price_changes

    if strategy in price_changes_data:
        return jsonify(price_changes_data[strategy])
    else:
        return jsonify({})


def fetch_top_market_cap_stocks():
    # List of Indian stock tickers
    indian_tickers = [
        'INFY.NS', 'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'HDFC.NS', 
        'ITC.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'LT.NS', 'AXISBANK.NS', 
        'SBIN.NS', 'BHARTIARTL.NS', 'ASIANPAINT.NS', 'HINDUNILVR.NS', 
        'MARUTI.NS', 'WIPRO.NS', 'BAJFINANCE.NS', 'ONGC.NS', 'SUNPHARMA.NS', 
        'TECHM.NS', 'NESTLEIND.NS', 'ULTRACEMCO.NS', 'IOC.NS', 'POWERGRID.NS', 
        'TITAN.NS', 'DRREDDY.NS', 'NTPC.NS', 'GAIL.NS', 'CIPLA.NS', 
        'GRASIM.NS', 'SHREECEM.NS', 'COALINDIA.NS', 'HCLTECH.NS', 'BAJAJFINSV.NS',
        'BAJAJ-AUTO.NS', 'JSWSTEEL.NS', 'BRITANNIA.NS', 'HEROMOTOCO.NS', 
        'INDUSINDBK.NS', 'ADANIPORTS.NS', 'UBL.NS', 'EICHERMOT.NS', 'BAJAJHLDNG.NS',
        'BPCL.NS', 'HINDALCO.NS', 'M&M.NS', 'VEDL.NS', 'HDFCLIFE.NS', 'MRF.NS',
        'DIVISLAB.NS', 'AUROPHARMA.NS', 'HDFCAMC.NS', 'BANDHANBNK.NS', 'DLF.NS',
        'YESBANK.NS', 'RAMCOCEM.NS', 'TATASTEEL.NS', 'CADILAHC.NS', 'PNB.NS',
        'LTI.NS', 'ADANIGREEN.NS', 'BIOCON.NS', 'UBL.NS', 'TORNTPOWER.NS'
    ]

    # Fetch market cap data using yfinance
    market_cap_data = {}
    for ticker in indian_tickers:
        try:
            # Get ticker data
            ticker_data = yf.Ticker(ticker)
            
            # Get market cap
            market_cap = ticker_data.info.get('marketCap', None)
            
            # If market cap is available, add to dictionary
            if market_cap:
                market_cap_data[ticker] = market_cap
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

    # Convert to DataFrame and sort by market cap
    df = pd.DataFrame(list(market_cap_data.items()), columns=['Symbol', 'MarketCap'])
    df_sorted = df.sort_values(by='MarketCap', ascending=False)
    
    # Get top 5 stocks by market cap
    top_5_stocks = df_sorted.head(5).to_dict(orient='records')
    
    return top_5_stocks


@app.route('/api/getTopMarketCap', methods=['GET'])
def get_top_market_cap():
    top_5_stocks = fetch_top_market_cap_stocks()
    return jsonify(top_5_stocks)


# Function to determine buy or sell signal
def detect_trend(data):
    if data.empty:
        return "Hold"  # If there's no data, return "Hold" by default
    
    # Check the color of the last 10 candlesticks
    last_10_close = data['close'].iloc[-10:]
    last_10_open = data['open'].iloc[-10:]

    buy_count = sum(last_10_close > last_10_open)
    sell_count = sum(last_10_close < last_10_open)
    
    if buy_count > sell_count:
        return "Buy"
    elif sell_count > buy_count:
        return "Sell"
    else:
        return "Hold"
# def detect_trend(data):
#     # Check the color of the last candlestick
#     last_close = data['close'].iloc[-1]
#     last_open = data['open'].iloc[-1]

#     if last_close > last_open:
#         return "Buy"
#     elif last_close < last_open:
#         return "Sell"
#     else:
#         return "Hold"
@app.route('/get_trend', methods=['GET'])
def get_trend():
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({"error": "Stock symbol is missing"}), 400
    
    try:
        # Get live data for the provided stock symbol
        stock_data = get_data(symbol)
        # Call the function to detect trend
        trend = detect_trend(stock_data)
        return jsonify({"trend": trend})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

class StockSuggestion:
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def load_data(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = eval(file.read())
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return {}

    def get_suggestions(self, prefix):
        suggestions = []
        if self.data is None:
            print("Data not loaded.")
            return suggestions

        for symbol, name in self.data.items():
            if name is not None and (symbol.startswith(prefix.upper()) or name.lower().startswith(prefix.lower())):
                suggestions.append({"symbol": symbol, "name": name})
        return suggestions[:10]


@app.route('/search', methods=['POST'])
def search():
    prefix = request.json['prefix']
    stock_suggestion = StockSuggestion('yahoo_symbols.txt')
    suggestions = stock_suggestion.get_suggestions(prefix)
    return jsonify(suggestions)


# Fetch real financial data

# Lorentzian Classifier


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
    socketio.run(app, debug=True)
