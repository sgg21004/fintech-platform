from fastapi import FastAPI, Query, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from prophet import Prophet
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from database import SessionLocal, engine, User, SearchHistory

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auth
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# DB connection
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_access_token(data: dict, expires_delta: timedelta):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

@app.post("/signup")
def signup(username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.username == username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")
    hashed_pw = get_password_hash(password)
    user = User(username=username, hashed_password=hashed_pw)
    db.add(user)
    db.commit()
    return {"message": "User created successfully"}

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Fintech Platform API"}

@app.get("/stock-info/")
def get_stock_info(ticker: str = Query(...)):
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "ticker": ticker.upper(),
        "longName": info.get("longName"),
        "currentPrice": info.get("currentPrice"),
        "marketCap": info.get("marketCap"),
        "sector": info.get("sector"),
        "website": info.get("website"),
        "summary": info.get("longBusinessSummary"),
    }

@app.get("/predict-price/")
def predict_price(ticker: str = Query(...)):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="60d")
    if hist.empty:
        return {"error": "No historical data found."}
    hist.reset_index(inplace=True)
    hist['DateOrdinal'] = hist['Date'].map(datetime.toordinal)
    X = hist[['DateOrdinal']]
    y = hist['Close']
    model = LinearRegression()
    model.fit(X, y)
    next_day = datetime.now() + timedelta(days=1)
    next_day_ordinal = next_day.toordinal()
    predicted_price = model.predict([[next_day_ordinal]])[0]
    return {
        "ticker": ticker.upper(),
        "predicted_close_price": round(predicted_price, 2),
        "prediction_date": next_day.strftime('%Y-%m-%d')
    }

@app.get("/historical-prices/")
def get_historical_prices(ticker: str = Query(...)):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="60d")
    if hist.empty:
        return {"error": "No historical data found."}
    hist.reset_index(inplace=True)
    prices = []
    for _, row in hist.iterrows():
        prices.append({
            "date": row["Date"].strftime('%Y-%m-%d'),
            "price": round(row["Close"], 2)
        })
    return {"prices": prices}

@app.get("/predict-forecast/")
def predict_forecast(ticker: str = Query(...)):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="6mo")
    if hist.empty:
        return {"error": "No historical data found."}
    hist.reset_index(inplace=True)
    hist['Date'] = hist['Date'].dt.tz_localize(None)
    df = hist[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)
    forecast_results = []
    for row in forecast.tail(7).itertuples():
        forecast_results.append({
            "date": row.ds.strftime('%Y-%m-%d'),
            "predicted_price": round(row.yhat, 2)
        })
    return {
        "ticker": ticker.upper(),
        "forecast": forecast_results
    }

@app.get("/recommend-investments/")
def recommend_investment(ticker: str = Query(...)):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="6mo")
    if hist.empty:
        return {"error": "No historical data found."}
    hist.reset_index(inplace=True)
    hist['Date'] = hist['Date'].dt.tz_localize(None)
    df = hist[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)
    forecast_tail = forecast.tail(7)
    prices = forecast_tail['yhat'].tolist()
    trend = prices[-1] - prices[0]
    if trend > 2:
        recommendation = "Buy"
    elif trend < -2:
        recommendation = "Sell"
    else:
        recommendation = "Hold"
    return {
        "ticker": ticker.upper(),
        "recommendation": recommendation,
        "7_day_change": round(trend, 2),
        "forecast_start": round(prices[0], 2),
        "forecast_end": round(prices[-1], 2)
    }

@app.get("/technical-indicators/")
def technical_indicators(ticker: str = Query(...)):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="3mo")
    if hist.empty:
        return {"error": "No historical data found."}
    hist.reset_index(inplace=True)
    hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')
    hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
    delta = hist['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    hist['RSI'] = 100 - (100 / (1 + rs))
    indicators = []
    for _, row in hist.iterrows():
        indicators.append({
            "date": row["Date"],
            "close": round(row["Close"], 2),
            "volume": int(row["Volume"]),
            "sma_20": round(row["SMA_20"], 2) if not pd.isna(row["SMA_20"]) else None,
            "rsi": round(row["RSI"], 2) if not pd.isna(row["RSI"]) else None
        })
    return {
        "ticker": ticker.upper(),
        "indicators": indicators[-60:]
    }

@app.get("/smart-recommendation/")
def smart_recommendation(ticker: str = Query(...)):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="3mo")
    if hist.empty:
        return {"error": "No historical data found."}

    forecast_hist = hist[['Close']].copy()
    forecast_hist.reset_index(inplace=True)
    forecast_hist['Date'] = forecast_hist['Date'].dt.tz_localize(None)
    df = forecast_hist.rename(columns={"Date": "ds", "Close": "y"})
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)
    prices = forecast.tail(7)['yhat'].tolist()
    forecast_slope = prices[-1] - prices[0]

    delta = hist['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    latest_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

    volatility = hist['Close'].std()

    if forecast_slope > 3 and latest_rsi < 60 and volatility < 5:
        action = "Strong Buy"
    elif forecast_slope > 1 and latest_rsi < 70:
        action = "Buy"
    elif abs(forecast_slope) < 1 or (45 < latest_rsi < 60):
        action = "Hold"
    elif forecast_slope < -1 and latest_rsi > 60:
        action = "Sell"
    else:
        action = "Strong Sell"

    return {
        "ticker": ticker.upper(),
        "smart_action": action,
        "forecast_slope": round(forecast_slope, 2),
        "latest_rsi": round(latest_rsi, 2),
        "volatility": round(volatility, 2)
    }
