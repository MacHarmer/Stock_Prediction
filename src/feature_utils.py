
import numpy as np
import pandas as pd
import datetime
import yfinance as yf
import pandas_datareader.data as web
import requests
import os
import sys

def extract_features():
    return_period = 5
    
    # 1. Define Dates and Tickers
    START_DATE = (datetime.date.today() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
    END_DATE = datetime.date.today().strftime("%Y-%m-%d")
    
    stk_tickers = ['AAPL', 'AAL', 'DAL', 'TSLA'] # Updated for your selection
    ccy_tickers = ['DEXJPUS', 'DEXUSUK', 'DEXCHUS']
    idx_tickers = ['SP500', 'DJIA', 'VIXCLS']
    
    # 2. Download Data
    stk_data = yf.download(stk_tickers, start=START_DATE, end=END_DATE, auto_adjust=False)
    ccy_data = web.DataReader(ccy_tickers, 'fred', start=START_DATE, end=END_DATE)
    idx_data = web.DataReader(idx_tickers, 'fred', start=START_DATE, end=END_DATE)

    # 3. Create Target (Y) - 5-day future returns for AAPL
    Y = np.log(stk_data.loc[:, ('Adj Close', 'AAPL')]).diff(return_period).shift(-return_period)
    Y.name = 'AAPL_Future'
    
    # 4. Create Feature Sets (X1, X2, X3)
    X1 = np.log(stk_data.loc[:, ('Adj Close', ('AAL', 'DAL', 'TSLA'))]).diff(return_period)
    X1.columns = X1.columns.droplevel()
    X2 = np.log(ccy_data).diff(return_period)
    X3 = np.log(idx_data).diff(return_period)

    # 5. Create Technical Indicator Features (X4) - Mandatory for Assignment
    # Momentum
    X4_1 = np.log(stk_data.loc[:, ('Adj Close', 'AAPL')]).diff(5)
    # Volatility
    X4_2 = (stk_data.loc[:, ('High', 'AAPL')] - stk_data.loc[:, ('Low', 'AAPL')])
    # Price Gap
    X4_3 = stk_data.loc[:, ('Open', 'AAPL')] - stk_data.loc[:, ('Adj Close', 'AAPL')].shift(1)
    # SMA Ratio
    sma20 = stk_data.loc[:, ('Adj Close', 'AAPL')].rolling(window=20).mean()
    X4_4 = stk_data.loc[:, ('Adj Close', 'AAPL')] / sma20

    X4 = pd.concat([X4_1, X4_2, X4_3, X4_4], axis=1)
    X4.columns = ['AAPL_Mom', 'AAPL_Vol', 'AAPL_Gap', 'AAPL_SMA_Ratio']

    # 6. Combine all 13 features
    X = pd.concat([X1, X2, X3, X4], axis=1)
    
    # 7. Final Data Processing
    dataset = pd.concat([Y, X], axis=1).dropna().iloc[::return_period, :]
    
    # CRITICAL FIX: Return exactly the 13 columns in X. 
    # Do NOT use iloc[:, 1:] as it removes the first feature.
    features = dataset[X.columns].sort_index().reset_index(drop=True)
    
    return features

def get_bitcoin_historical_prices(days=60):
    BASE_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {
        'vs_currency': 'usd',
        'days': days,
        'interval': 'daily'
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['Timestamp', 'Close Price (USD)'])
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms').dt.normalize()
    df = df[['Date', 'Close Price (USD)']].set_index('Date')
    return df


