import os  
import time  
import numpy as np  
import pandas as pd  
import ta  
import streamlit as st  
import matplotlib.pyplot as plt  
from sklearn.preprocessing import StandardScaler  
from stable_baselines3 import PPO, DDPG  
from stable_baselines3.common.envs import DummyVecEnv  
from stable_baselines3.common.env_util import make_vec_env  
from gym import Env, spaces  
from alpaca.trading.client import TradingClient  
from alpaca.data.historical import StockHistoricalDataClient  
from alpaca.data.requests import StockBarsRequest  
from alpaca.data.timeframe import TimeFrame  
import requests  
import threading  
  
# 🔐 API Keys  
ALPACA_API_KEY = "YOUR_API_KEY"  
ALPACA_SECRET_KEY = "YOUR_SECRET_KEY"  
NEWS_API_KEY = "YOUR_NEWS_API_KEY"  
  
TRADING_CLIENT = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=False)  
DATA_CLIENT = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)  
  
st.set_page_config(page_title="DRL Trading Bot", layout="wide")  
st.title("🤖 DRL Trading Bot")  
st.sidebar.header("🔧 Controls")  
  
symbol = st.sidebar.text_input("Symbol", "AAPL").upper()  
interval = st.sidebar.selectbox("Interval", ["5m", "10m", "30m", "1h", "1d"])  
reinvest = st.sidebar.selectbox("Reinvest %", ["25%", "50%", "75%", "100%"])  
strategy = st.sidebar.selectbox("Strategy", ["PPO", "DDPG", "Both"])  
mode = st.sidebar.radio("Mode", ["Paper", "Live"])  
investment = st.sidebar.number_input("Invest Amount ($)", value=100.0, step=50.0)  
sharpe_threshold = st.sidebar.slider("Sharpe Threshold", 0.0, 2.0, 1.0, step=0.05)  
loop_interval = st.sidebar.number_input("Loop Interval (sec)", 60, 3600, 300, step=60)  
  
if mode == "Live":  
    st.sidebar.warning("⚠️ Live mode enabled — Real trades will execute!")  
  
run_btn, test_btn, cash_btn, sharpe_btn = st.columns(4)  
if run_btn.button("🚀 Run Bot"): st.session_state.run = True  
if test_btn.button("🧪 Test Run"): st.session_state.test = True  
if cash_btn.button("💸 Cash Out"): st.session_state.cash = True  
if sharpe_btn.button("📈 Sharpe Ratio"): st.session_state.sharpe = True  
  
chart_container = st.empty()  
log_container = st.empty()  
sharpe_container = st.empty()  
news_container = st.empty()  
action_container = st.empty()  
  
for key in ["run", "test", "cash", "sharpe", "last_action"]:  
    if key not in st.session_state:  
        st.session_state[key] = False if key != "last_action" else None  
  
class TradingEnv(Env):  
    def __init__(self, data):  
        super(TradingEnv, self).__init__()  
        self.data = data.reset_index(drop=True)  
        self.current_step = 0  
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)  
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)  
  
    def reset(self):  
        self.current_step = 0  
        return self._next_observation()  
  
    def _next_observation(self):  
        row = self.data.iloc[self.current_step]  
        return np.array([row["Close"], row["rsi"], row["sma"], row["settlement"]], dtype=np.float32)  
  
    def step(self, action):  
        self.current_step += 1  
        done = self.current_step >= len(self.data) - 1  
        reward = float(action[0]) * (self.data["Close"].iloc[self.current_step] - self.data["Close"].iloc[self.current_step - 1])  
        obs = self._next_observation() if not done else np.zeros((4,), dtype=np.float32)  
        return obs, reward, done, {}  
  
    def render(self, mode="human"):  
        pass  
  
def fetch_data(symbol, interval, limit=500):
    tf_map = {
        "1m": TimeFrame.Minute,
        "5m": TimeFrame.Minute, 
        "10m": TimeFrame.Minute, 
        "30m": TimeFrame.Minute, 
        "1h": TimeFrame.Hour, 
        "1d": TimeFrame.Day
    }

    multiplier_map = {
        "1m": 1,
        "5m": 5,
        "10m": 10,
        "30m": 30,
        "1h": 1,
        "1d": 1
    }

    tf = tf_map[interval]
    multiplier = multiplier_map[interval]

    end = pd.Timestamp.utcnow()
    start = end - pd.Timedelta(minutes=limit * multiplier)
    
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(unit=tf.unit, value=multiplier),
        start=start,
        end=end
    )
    
    df = DATA_CLIENT.get_stock_bars(req).df
    ...  
    end = pd.Timestamp.utcnow()  
    start = end - pd.Timedelta(minutes=limit * 2)  
    req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=tf, start=start, end=end)  
    df = DATA_CLIENT.get_stock_bars(req).df  
    df = df[df.symbol == symbol].copy().reset_index()  
    df.rename(columns={"close": "Close", "open": "Open", "high": "High", "low": "Low", "volume": "Volume"}, inplace=True)  
    df.drop(columns=["symbol"], inplace=True)  
    df.set_index("timestamp", inplace=True)  
    df["rsi"] = ta.momentum.RSIIndicator(df["Close"], 14).rsi()  
    df["sma"] = df["Close"].rolling(10).mean()  
    df["settlement"] = df["Close"].shift(1).fillna(df["Close"])  
    df.dropna(inplace=True)  
    return df  
  
def calculate_sharpe(returns, rf=0.01):  
    er = returns - rf / 252  
    std = np.std(er)  
    return 0.0 if std == 0 else np.mean(er) / std  
  
@st.cache_resource  
def train_models(df):  
    env = DummyVecEnv([lambda: TradingEnv(df)])  
    ppo_model = PPO("MlpPolicy", env, verbose=0)  
    ddpg_model = DDPG("MlpPolicy", env, verbose=0)  
    ppo_model.learn(total_timesteps=5000)  
    ddpg_model.learn(total_timesteps=5000)  
    return ppo_model, ddpg_model  
  
def analyze_news(symbol):  
    try:  
        url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={NEWS_API_KEY}&pageSize=5&sortBy=publishedAt"  
        data = requests.get(url).json()  
        titles = [article["title"] for article in data.get("articles", [])]  
        if titles:  
            news_container.info("📰 News:\n" + "\n".join(f"• {t}" for t in titles))  
    except Exception as e:  
        news_container.warning(f"News fetch failed: {e}")  
  
def decide_action(action_val, sharpe):  
    if sharpe < sharpe_threshold:  
        return "hold", f"Sharpe {sharpe:.2f} < threshold"  
    if action_val > 0.5:  
        return "buy", f"Action {action_val:.2f} > 0.5"  
    elif action_val < -0.5:  
        return "sell", f"Action {action_val:.2f} < -0.5"  
    else:  
        return "hold", f"Action {action_val:.2f} in [-0.5, 0.5]"  
  
def trade_loop():  
    df = fetch_data(symbol, interval)  
    if df.empty:  
        return df, None, 0, None, 0.0, "No data"  
  
    ppo_model, ddpg_model = train_models(df)  
  
    returns = df["Close"].pct_change().dropna()  
    sharpe = calculate_sharpe(returns)  
    analyze_news(symbol)  
  
    X = df[["Close", "rsi", "sma", "settlement"]].values  
    obs = StandardScaler().fit_transform(X)[-1].reshape(1, -1)  
  
    if strategy == "PPO":  
        action_val = float(ppo_model.predict(obs, deterministic=True)[0][0])  
    elif strategy == "DDPG":  
        action_val = float(ddpg_model.predict(obs, deterministic=True)[0][0])  
    else:  
        ppo_a = float(ppo_model.predict(obs, deterministic=True)[0][0])  
        ddpg_a = float(ddpg_model.predict(obs, deterministic=True)[0][0])  
        action_val = (ppo_a + ddpg_a) / 2  
  
    side, reason = decide_action(action_val, sharpe)  
    price = df["Close"].iloc[-1]  
    qty = round(investment / price, 4)  
  
    if side != "hold" and mode == "Live":  
        try:  
            TRADING_CLIENT.submit_order(symbol=symbol, qty=qty, side=side, type="market", time_in_force="gtc")  
            log_container.success(f"✅ Order executed: {side.upper()} {qty} @ ${price:.2f}")  
        except Exception as e:  
            log_container.error(f"❌ Order failed: {e}")  
  
    return df, action_val, qty, side, sharpe, reason  
  
def run_forever():  
    while st.session_state.run:  
        df, action_val, qty, side, sharpe, reason = trade_loop()  
        if df is not None:  
            chart_container.line_chart(df[["Close", "sma", "rsi"]])  
            action_container.write(f"🧠 Decision: **{side.upper()}** ({reason})")  
            sharpe_container.write(f"📈 Sharpe Ratio: **{sharpe:.4f}**")  
            log_container.write(f"{symbol} @ ${df['Close'].iloc[-1]:.2f} | Action={action_val:.3f} | Qty={qty}")  
        time.sleep(loop_interval)  
  
# Run once on press  
if st.session_state.test:  
    st.info("✅ Test Run OK")  
    st.session_state.test = False  
  
if st.session_state.cash:  
    st.warning("💸 Cash Out triggered (placeholder)")  
    st.session_state.cash = False  
  
if st.session_state.sharpe:  
    df = fetch_data(symbol, interval)  
    sr = calculate_sharpe(df["Close"].pct_change().dropna()) if not df.empty else 0.0  
    sharpe_container.write(f"📉 Sharpe Ratio: **{sr:.4f}**")  
    st.session_state.sharpe = False  
  
if st.session_state.run:  
    if "thread" not in st.session_state or not st.session_state.thread.is_alive():  
        st.session_state.thread = threading.Thread(target=run_forever, daemon=True)  
        st.session_state.thread.start()  
    st.success("🔁 Bot is running continuously.")  
else:  
    st.info("Press 🚀 to start the 24-hour trading bot.")  
