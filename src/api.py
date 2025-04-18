from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime, timedelta
import sqlite3
import os
from typing import List, Optional
import json
import logging  
import traceback
import sys                          
import warnings

# Initialize FastAPI app
app = FastAPI(
    title="Offer Recommendation API",
    description="API for generating offer-datafile recommendations based on campaign data.",
    version="1.0.0"
)

# Pydantic model for input validation
class RecommendationRequest(BaseModel):
    projection_volume: int = 800000
    exclude_days: int = 15
    prediction_basis: List[str] = ["Revenue"]
    selected_isps: List[str] = ["All ISPs"]
    selected_sponsors: List[str] = ["All Sponsors"]

# Pydantic model for recommendation output
class Recommendation(BaseModel):
    campaign_name: str
    datafile: str
    cpm: float
    volume: int
    priority: int
    isp_name: str
    sponsor: str
    category: str
    last_send_date: str
    file_type: Optional[str] = None
    file_category: Optional[str] = None

# Mock recent combinations (same as in app.py)
recent_combinations = [
    {"campaign_name": "excenduprot", "datafile": "rr_2024_cls", "isp_name": "RR", "date": "2025-04-10"},
    {"campaign_name": "scheech", "datafile": "rr_2023_cls", "isp_name": "RR", "date": "2025-04-09"},
]

# Default sponsor and ISP lists
default_sponsors = ["PDS", "W4", "GWM", "Madrivo", "EFL", "CFW", "DAG", "DFO", "BSK", "NIW", "LLS", "W4E"]
default_isps = ["RR", "Comcast-ESP", "Comcast", "Charter-ESP", "Juno-Ded", "Charter - Ded", "Atlanticbb", "TWC-Ded", "TWC-ESP", "Cox", "AOL", "Yahoo"]

# Database setup
def init_db():
    conn = sqlite3.connect(r"C:\Users\User.Think-EDS-37\Desktop\Learning\working code REC\data\Rec.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            campaign_name TEXT,
            datafile TEXT,
            cpm REAL,
            volume INTEGER,
            priority INTEGER,
            isp_name TEXT,
            sponsor TEXT,
            category TEXT,
            last_send_date TEXT,
            recommendation_date TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# Reuse functions from app.py (with modifications)
def fetch_data(api_url: str):
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            json_data = response.json()
            if isinstance(json_data, dict) and 'data' in json_data:
                return json_data['data']
            else:
                raise HTTPException(status_code=500, detail=f"Unexpected API response structure: {json_data}")
        else:
            raise HTTPException(status_code=response.status_code, detail=f"API request failed with status code: {response.status_code}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data from {api_url}: {str(e)}")

# Include preprocess_data, get_new_offer_priority, categorize_offer, is_recent_combination, save_to_db, train_and_predict_recommendations
# (Copy these directly from your app.py, but modify error handling to raise HTTPException instead of st.error/st.warning)

def preprocess_data(report_df, campaigns_df, sheet16_df):
    required_report_cols = ['offer_code', 'offer_date', 'cpm', 'sent', 'opens', 'clicks', 'revenue', 'campaign_name', 'isp_name', 'datafile', 'esp_name', 'sub_id', 'mailer_name', 'epc']
    required_campaigns_cols = ['cid', 'campaign_name', 'sponsor', 'category', 'sub_url', 'unsub_urlК', 'last_send_date']
    required_sheet16_cols = ['offer_code', 'offer_date', 'cpm', 'sent', 'opens', 'clicks', 'revenue', 'campaign_name', 'isp_name', 'datafile', 'esp_name', 'sub_id', 'mailer_name', 'epc']