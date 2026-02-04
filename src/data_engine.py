import yfinance as yf
import pandas as pd
import numpy as np
import requests
import feedparser
import urllib.parse


def get_all_stock_data(ticker):
    try:
        # 1. Download data
        df = yf.download(ticker, period='3y', interval='1d', auto_adjust=True)

        if df is None or df.empty:
            return None

        # 2. CRITICAL FIX: Flatten MultiIndex immediately
        # This ensures columns are exactly ['Close', 'Open', etc.]
        # and NOT [('Close', 'NVDA'), ('Open', 'NVDA')]
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 3. Calculate technical indicators
        # Moving Averages
        df['MA7'] = df['Close'].rolling(window=7).mean()
        df['MA21'] = df['Close'].rolling(window=21).mean()

        # RSI (Wilder's Smoothing)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(com=13, min_periods=14).mean()
        avg_loss = loss.ewm(com=13, min_periods=14).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        df['RSI'] = 100 - (100 / (1 + rs))

        # Returns & Volatility
        df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volatility'] = df['Returns'].rolling(window=21).std()

        # 4. Final Clean
        df = df.dropna()

        # DEBUG: Print columns to terminal to verify 'RSI' exists
        print(f"📊 Processed Columns for {ticker}: {list(df.columns)}")

        return df

    except Exception as e:
        print(f"❌ Data Engine Error: {e}")
        return None


def get_live_news(ticker):
    try:
        # Clean the ticker (remove .NS for better search results)
        query_term = ticker.replace('.NS', '')
        # Construct the Google News RSS URL
        encoded_query = urllib.parse.quote(f"{query_term} stock news")
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"

        # Parse the feed
        feed = feedparser.parse(rss_url)

        # Extract the top 5 unique headlines
        headlines = []
        for entry in feed.entries[:5]:
            # Clean up the headline (removes the source name at the end)
            clean_title = entry.title.split(' - ')[0]
            headlines.append(clean_title)

        return headlines
    except Exception as e:
        print(f"Error fetching RSS news: {e}")
        return ["Could not fetch live news at this moment."]
