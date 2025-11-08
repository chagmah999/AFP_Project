import os

# API keys
FMP_API_KEY = os.getenv("FMP_API_KEY", "YOUR_FMP_API_KEY")

# FMP bases 
FMP_BASE = "https://financialmodelingprep.com/api/v3"
FMP_STABLE_BASE = "https://financialmodelingprep.com/stable"
FMP_V4_BASE = "https://financialmodelingprep.com/api/v4"

# General params
DEFAULT_START_DATE = "2022-01-01"
LOOKBACK_DAYS = 126        # 6 months
FORECAST_HORIZON_DAYS = 21 # ~1 month

# Streamlit defaults
DEFAULT_UNIVERSE_SIZE = 50
MAX_UNIVERSE_SIZE = 509
