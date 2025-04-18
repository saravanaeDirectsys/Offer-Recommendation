import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime, timedelta
import streamlit as st
import sqlite3
import os

# Streamlit App Configuration
st.set_page_config(
    page_title="Offer Recommendation System",
    layout="wide",
    page_icon="📊"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        cursor: pointer;
    }
    .stTextInput>label, .stNumberInput>label, .stSelectbox>label, .stMultiSelect>label {
        font-size: 16px;
        font-weight: bold;
        color: #333;
    }
    .stSpinner {
        color: #4CAF50;
    }
    .recommendation-box {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    .header {
        font-size: 36px;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 20px;
    }
    .subheader {
        font-size: 24px;
        color: #34495e;
        margin-top: 20px;
    }
    .sidebar .stSidebar {
        background-color: #e8ecef;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar for Inputs
with st.sidebar:
    st.header("📋 Input Parameters")
    
    st.markdown("**Configure your recommendation settings below:**")
    
    projection_volume = st.number_input(
        "Projection Volume per ISP",
        min_value=1,
        value=800000,
        step=1000,
        help="Enter the desired volume per ISP."
    )
    
    exclude_days = st.number_input(
        "Exclude combinations from last N days",
        min_value=0,
        value=15,
        step=1,
        help="Number of days to exclude recent offer-datafile combinations."
    )
    
    prediction_basis = st.multiselect(
        "Predict based on",
        ["Revenue", "CPM"],
        default=["Revenue"],
        help="Choose whether to predict based on high revenue, high CPM, or both."
    )
    
    isp_options = st.session_state.get('isp_options', ["All ISPs"])
    selected_isps = st.multiselect(
        "Select ISP(s)",
        isp_options,
        default=["All ISPs"],
        help="Select one or more ISPs, or 'All ISPs' for no filter."
    )
    
    sponsor_options = st.session_state.get('sponsor_options', ["All Sponsors"])
    selected_sponsors = st.multiselect(
        "Select Sponsor(s)",
        sponsor_options,
        default=["All Sponsors"],
        help="Select one or more sponsors, or 'All Sponsors' for no filter."
    )
    
    start_button = st.button("🚀 Start Prediction", key="start_button")

# Main Content
st.markdown("<div class='header'>Offer Recommendation System</div>", unsafe_allow_html=True)
st.markdown("""
    Welcome to the **Offer Recommendation System**! This tool helps you select optimal offer-datafile pairs based on your ISP projection volume,
    ensuring 7–23 offers, 1–5 data files per offer, and avoiding recent combinations or offers repeated within 3 days (unless CPM ≥ 1.5).
""")

# Placeholder for results
results_container = st.container()

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.merged_df = None
    st.session_state.campaigns_df = None
    st.session_state.sheet16_df = None
    st.session_state.isp_options = ["All ISPs"]
    st.session_state.sponsor_options = ["All Sponsors"]
    st.session_state.recommendations_df = None

# Mock recent combinations log
recent_combinations = [
    {"campaign_name": "excenduprot", "datafile": "rr_2024_cls", "isp_name": "RR", "date": "2025-04-10"},
    {"campaign_name": "scheech", "datafile": "rr_2023_cls", "isp_name": "RR", "date": "2025-04-09"},
]

# Default sponsor and ISP lists
default_sponsors = ["PDS", "W4", "GWM", "Madrivo", "EFL", "CFW", "DAG", "DFO", "BSK", "NIW", "LLS", "W4E"]
default_isps = ["RR", "Comcast-ESP", "Comcast", "Charter-ESP", "Juno-Ded", "Charter - Ded", "Atlanticbb", "TWC-Ded", "TWC-ESP", "Cox", "AOL", "Yahoo"]

# Database Setup
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

# Function Definitions
def fetch_data(api_url):
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            json_data = response.json()
            if isinstance(json_data, dict) and 'data' in json_data:
                return json_data['data']
            else:
                st.error(f"Unexpected API response structure: {json_data}")
                return None
        else:
            st.error(f"API Request Failed with Status Code: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error fetching data from {api_url}: {e}")
        return None

def preprocess_data(report_df, campaigns_df, sheet16_df):
    required_report_cols = ['offer_code', 'offer_date', 'cpm', 'sent', 'opens', 'clicks', 'revenue', 'campaign_name', 'isp_name', 'datafile', 'esp_name', 'sub_id', 'mailer_name', 'epc']
    required_campaigns_cols = ['cid', 'campaign_name', 'sponsor', 'category', 'sub_url', 'unsub_url', 'payout', 'campaign_id', 'payment_type', 'status', 'tags', 'description']
    required_sheet16_cols = ['ISP Name', 'Data File', 'DF Count', 'FILE TYPE']
    
    for df, cols, name in [
        (report_df, required_report_cols, "report_df"),
        (campaigns_df, required_campaigns_cols, "campaigns_df"),
        (sheet16_df, required_sheet16_cols, "sheet16_df")
    ]:
        if not all(col in df.columns for col in cols):
            missing = [col for col in cols if col not in df.columns]
            st.error(f"Missing columns in {name}: {missing}")
            return None, None, None
    
    report_df['offer_date'] = pd.to_datetime(report_df['offer_date'], errors='coerce')
    if report_df['offer_date'].isna().any():
        st.warning("Some offer_date values could not be converted to datetime. Filling with today's date.")
        report_df['offer_date'] = report_df['offer_date'].fillna(datetime.now())
    
    numeric_cols_report = ['cpm', 'sent', 'opens', 'clicks', 'revenue', 'epc']
    for col in numeric_cols_report:
        if col in report_df.columns:
            report_df[col] = pd.to_numeric(report_df[col].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce')
        else:
            report_df[col] = 0
    
    if 'payout' in campaigns_df.columns:
        campaigns_df['payout'] = pd.to_numeric(campaigns_df['payout'].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce')
    else:
        campaigns_df['payout'] = 0
    
    sheet16_df['DF Count'] = pd.to_numeric(sheet16_df['DF Count'], errors='coerce')
    
    report_df['campaign_name'] = report_df['campaign_name'].astype(str).str.strip().str.lower()
    campaigns_df['campaign_name'] = campaigns_df['campaign_name'].astype(str).str.strip().str.lower()
    report_df['isp_name'] = report_df['isp_name'].astype(str).str.strip().str.lower()
    sheet16_df['ISP Name'] = sheet16_df['ISP Name'].astype(str).str.strip().str.lower()
    
    merged_df = pd.merge(report_df, campaigns_df, on='campaign_name', how='inner', suffixes=('_report', '_LIMcampaign'))
    
    if merged_df.empty:
        st.warning("Merged DataFrame is empty. Using report_df only.")
        merged_df = report_df.copy()
        merged_df['tags'] = ''
        merged_df['payout'] = 0
        merged_df['sponsor'] = 'Unknown'
    
    for col in ['cpm', 'sent', 'opens', 'clicks', 'revenue', 'payout', 'epc']:
        if col in merged_df:
            merged_df[col] = merged_df[col].fillna(0)
        else:
            merged_df[col] = 0
    
    merged_df = pd.merge(merged_df, sheet16_df[['ISP Name', 'Data File', 'DF Count', 'FILE TYPE']],
                         left_on=['isp_name', 'datafile'], right_on=['ISP Name', 'Data File'], how='left')
    
    for col in ['DF Count']:
        if col in merged_df:
            merged_df[col] = merged_df[col].fillna(0)
    
    isp_options = ["All ISPs"] + sorted(merged_df['isp_name'].dropna().unique().tolist())
    sponsor_options = ["All Sponsors"] + sorted(merged_df['sponsor'].dropna().unique().tolist())
    
    if not isp_options or len(isp_options) <= 1:
        isp_options = ["All ISPs"] + sorted(default_isps)
    if not sponsor_options or len(sponsor_options) <= 1:
        sponsor_options = ["All Sponsors"] + sorted(default_sponsors)
    
    st.session_state.isp_options = isp_options
    st.session_state.sponsor_options = sponsor_options
    
    return merged_df, campaigns_df, sheet16_df

def get_new_offer_priority(tags):
    tags_str = str(tags).lower()
    if "new" in tags_str and "top" in tags_str and "must try" in tags_str:
        return 1
    elif "new" in tags_str and "must try" in tags_str:
        return 2
    elif "new" in tags_str and "top" in tags_str:
        return 3
    elif "new" in tags_str:
        return 4
    return 99

def categorize_offer(pair, merged_df, cutoff_date, case):
    campaign_name = pair.get('campaign_name', '')
    offer_code = next((row['offer_code'] for _, row in merged_df.iterrows() if row.get('campaign_name', '') == campaign_name), None)
    cpm_value = pair.get('cpm', 0)
    cpm_threshold = 1.75 if case == 3 else 1.5
    
    if offer_code:
        recent_data = merged_df[(merged_df['offer_code'] == offer_code) & (merged_df['offer_date'] >= pd.to_datetime(cutoff_date))]
        if not recent_data.empty and cpm_value >= cpm_threshold:
            return "Recent high-CPM offer"
    
    historical = merged_df.groupby('offer_code').agg({'revenue': 'mean'}).sort_values(by='revenue', ascending=False).index
    if offer_code in historical[:5].tolist():
        return "Historical best offer"
    
    if 'new' in str(pair.get('tags', '')).lower():
        return "New offer"
    
    return "Other"

def is_recent_combination(campaign_name, datafile, isp_name, recent_combinations, exclude_days):
    today = datetime.now()
    cutoff_date = today - timedelta(days=exclude_days)
    for combo in recent_combinations:
        if (combo['campaign_name'] in campaign_name and
            combo['datafile'] == datafile and
            combo['isp_name'].lower() == isp_name.lower()):
            combo_date = datetime.strptime(combo['date'], '%Y-%m-%d')
            if combo_date >= cutoff_date:
                return True
    return False

def save_to_db(recommendations_df):
    try:
        conn = sqlite3.connect(r"C:\Users\User.Think-EDS-37\Desktop\Learning\working code REC\data\Rec.db")
        recommendations_df['recommendation_date'] = datetime.now().strftime('%Y-%m-%d')
        required_columns = ['campaign_name', 'datafile', 'cpm', 'volume', 'priority', 'isp_name', 'sponsor', 'category', 'last_send_date', 'recommendation_date']
        for col in required_columns:
            if col not in recommendations_df.columns:
                recommendations_df[col] = None
        recommendations_df[required_columns].to_sql('recommendations', conn, if_exists='append', index=False)
        conn.commit()
        conn.close()
        st.success("Recommendations saved to Rec.db locally!")
    except Exception as e:
        st.error(f"Error saving to database locally: {e}")

def send_to_api(recommendations_df, api_url="http://127.0.0.1:8000/store-recommendations"):
    try:
        # Convert DataFrame to list of dictionaries
        recommendations_list = recommendations_df.to_dict(orient="records")
        # Ensure all required fields are present and handle missing ones
        for rec in recommendations_list:
            rec['file_type'] = rec.get('file_type', None)
            rec['file_category'] = rec.get('file_category', None)
        # Send POST request to API
        response = requests.post(api_url, json=recommendations_list)
        if response.status_code == 200:
            st.success("Recommendations successfully sent to API and stored!")
            return response.json()
        else:
            st.error(f"Failed to send recommendations to API. Status code: {response.status_code}, Error: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error sending recommendations to API: {str(e)}")
        # Fallback to local save
        st.warning("Saving recommendations locally as API is unavailable.")
        save_to_db(recommendations_df)
        return None

def train_and_predict_recommendations(merged_df, sheet16_df, projection_volume, exclude_days, recent_combinations, prediction_basis, selected_isps, selected_sponsors, target_offer_count=None):
    working_df = merged_df.copy()
    working_sheet16 = sheet16_df.copy()
    
    if "All ISPs" in selected_isps:
        selected_isps = working_df['isp_name'].dropna().unique().tolist()
    if "All Sponsors" in selected_sponsors:
        selected_sponsors = working_df['sponsor'].dropna().unique().tolist()
    
    working_df = working_df[
        (working_df['isp_name'].str.lower().isin([isp.lower() for isp in selected_isps])) &
        (working_df['sponsor'].str.lower().isin([sponsor.lower() for sponsor in selected_sponsors]))
    ]
    working_sheet16 = working_sheet16[
        working_sheet16['ISP Name'].str.lower().isin([isp.lower() for isp in selected_isps])
    ]
    
    if working_df.empty:
        st.error("No data available for the selected ISPs or sponsors.")
        return pd.DataFrame()
    
    working_sheet16['File Category'] = working_sheet16['Data File'].apply(
        lambda x: 'CLS' if 'CLS' in str(x).upper() else 'OPS' if 'OPS' in str(x).upper() else 'Normal'
    )
    
    base_features = ['sent', 'opens', 'clicks', 'payout']
    models = {}
    for basis in prediction_basis:
        target_feature = 'cpm' if basis == "CPM" else 'revenue'
        all_features = base_features + [target_feature]
        
        X = working_df[all_features].fillna(0)
        median_value = working_df[target_feature].median()
        y = (working_df[target_feature] > median_value).astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        results_container.write(f"**Model Accuracy ({basis}):** {accuracy_score(y_test, y_pred):.2f}")
        results_container.write(f"**Classification Report ({basis}):**\n```\n{classification_report(y_test, y_pred, zero_division=0)}\n```")
        
        models[basis] = model
    
    if projection_volume < 750000:
        recent_limit, historical_limit, new_limit = 2, 4, 0
        min_offers, max_offers = 7, 12
    elif projection_volume < 1500000:
        recent_limit, historical_limit, new_limit = 5, 7, 0
        min_offers, max_offers = 10, 15
    elif projection_volume < 2000000:
        recent_limit, historical_limit, new_limit = len(working_df[(pd.to_datetime(working_df['offer_date']) >= pd.to_datetime(datetime.now() - timedelta(days=3))) & (working_df['cpm'] >= 1.75)]), 10, 2
        min_offers, max_offers = 15, 18
    else:
        recent_limit, historical_limit, new_limit = len(working_df[(pd.to_datetime(working_df['offer_date']) >= pd.to_datetime(datetime.now() - timedelta(days=3))) & (working_df['cpm'] >= 1.75)]), 10, 5
        min_offers, max_offers = 15, 23
    
    cutoff_date = datetime.now() - timedelta(days=3)
    recommendations = []
    datafile_usage = {}
    
    for isp in selected_isps:
        isp_df = working_df[working_df['isp_name'].str.lower() == isp.lower()].copy()
        if isp_df.empty:
            continue
        
        isp_projection_volume = projection_volume
        isp_offers = isp_df[(isp_df['revenue'] > 0) | (isp_df['cpm'] > 0)].drop_duplicates(subset=['campaign_name'])
        isp_recommendations = []
        used_offers = set()
        used_datafiles = set()
        current_volume = 0
        offer_count = 0
        
        def can_use_datafile(datafile):
            return datafile_usage.get(datafile, 0) < 2
        
        def update_datafile_usage(datafile):
            datafile_usage[datafile] = datafile_usage.get(datafile, 0) + 1
            if datafile_usage[datafile] >= 2:
                used_datafiles.add(datafile)
        
        isp_offers_with_priority = isp_offers.copy()
        isp_offers_with_priority['priority'] = isp_offers_with_priority['tags'].apply(get_new_offer_priority)
        
        recent_offers = isp_offers_with_priority[
            (pd.to_datetime(isp_offers_with_priority['offer_date']) >= pd.to_datetime(cutoff_date)) &
            (isp_offers_with_priority['cpm'] >= 1.75)
        ].sort_values(by='cpm', ascending=False)
        
        for _, offer in recent_offers.head(recent_limit).iterrows():
            if offer['campaign_name'] in used_offers or current_volume >= isp_projection_volume or offer_count >= max_offers:
                continue
            
            datafiles = working_sheet16[working_sheet16['ISP Name'].str.lower() == isp.lower()].copy()
            datafiles = datafiles[~datafiles['Data File'].isin(used_datafiles)]
            datafiles = pd.merge(datafiles, working_df[['isp_name', 'datafile', 'cpm']],
                                left_on=['ISP Name', 'Data File'], right_on=['isp_name', 'datafile'], how='left').fillna(0)
            valid_datafiles = datafiles.sort_values(by='cpm', ascending=False).head(5)
            
            if valid_datafiles.empty:
                continue
            
            datafile_count = 0
            for _, datafile in valid_datafiles.iterrows():
                if not can_use_datafile(datafile['Data File']) or is_recent_combination(offer['campaign_name'], datafile['Data File'], isp, recent_combinations, exclude_days):
                    continue
                if current_volume + datafile['DF Count'] <= isp_projection_volume:
                    rec = {
                        'campaign_name': offer['campaign_name'],
                        'datafile': datafile['Data File'],
                        'cpm': offer.get('cpm', 0),
                        'volume': datafile['DF Count'],
                        'priority': offer['priority'],
                        'isp_name': isp,
                        'sponsor': offer.get('sponsor', 'Unknown'),
                        'category': 'Recent High CPM Offer',
                        'last_send_date': offer.get('offer_date', '').strftime('%Y-%m-%d') if pd.notnull(offer.get('offer_date')) else 'N/A',
                        'file_type': datafile['FILE TYPE'],
                        'file_category': datafile['File Category']
                    }
                    isp_recommendations.append(rec)
                    current_volume += datafile['DF Count']
                    update_datafile_usage(datafile['Data File'])
                    datafile_count += 1
                if current_volume >= isp_projection_volume or datafile_count >= 5:
                    break
            if datafile_count > 0:
                used_offers.add(offer['campaign_name'])
                offer_count += 1
            if current_volume >= isp_projection_volume or offer_count >= max_offers:
                break
        
        historical_offers = isp_offers_with_priority[
            (pd.to_datetime(isp_offers_with_priority['offer_date']) < pd.to_datetime(cutoff_date)) &
            (isp_offers_with_priority['cpm'] >= 1.5) &
            (~isp_offers_with_priority['campaign_name'].isin([c['campaign_name'] for c in recent_combinations if c['isp_name'].lower() == isp.lower()]))
        ].sort_values(by='cpm', ascending=False)
        
        for _, offer in historical_offers.head(historical_limit).iterrows():
            if offer['campaign_name'] in used_offers or current_volume >= isp_projection_volume or offer_count >= max_offers:
                continue
            
            datafiles = working_sheet16[working_sheet16['ISP Name'].str.lower() == isp.lower()].copy()
            datafiles = datafiles[~datafiles['Data File'].isin(used_datafiles)]
            datafiles = pd.merge(datafiles, working_df[['isp_name', 'datafile', 'cpm']],
                                left_on=['ISP Name', 'Data File'], right_on=['isp_name', 'datafile'], how='left').fillna(0)
            valid_datafiles = datafiles.sort_values(by='cpm', ascending=False).head(5)
            
            if valid_datafiles.empty:
                continue
            
            datafile_count = 0
            for _, datafile in valid_datafiles.iterrows():
                if not can_use_datafile(datafile['Data File']) or is_recent_combination(offer['campaign_name'], datafile['Data File'], isp, recent_combinations, exclude_days):
                    continue
                if current_volume + datafile['DF Count'] <= isp_projection_volume and datafile_count < 5:
                    rec = {
                        'campaign_name': offer['campaign_name'],
                        'datafile': datafile['Data File'],
                        'cpm': offer.get('cpm', 0),
                        'volume': datafile['DF Count'],
                        'priority': offer['priority'],
                        'isp_name': isp,
                        'sponsor': offer.get('sponsor', 'Unknown'),
                        'category': 'Historical Best Offer',
                        'last_send_date': offer.get('offer_date', '').strftime('%Y-%m-%d') if pd.notnull(offer.get('offer_date')) else 'N/A',
                        'file_type': datafile['FILE TYPE'],
                        'file_category': datafile['File Category']
                    }
                    isp_recommendations.append(rec)
                    current_volume += datafile['DF Count']
                    update_datafile_usage(datafile['Data File'])
                    datafile_count += 1
                if current_volume >= isp_projection_volume or datafile_count >= 5:
                    break
            if datafile_count > 0:
                used_offers.add(offer['campaign_name'])
                offer_count += 1
            if current_volume >= isp_projection_volume or offer_count >= max_offers:
                break
        
        if projection_volume >= 1500000 and current_volume < isp_projection_volume and offer_count < max_offers:
            new_offers = isp_offers_with_priority[
                ('new' in isp_offers_with_priority['tags'].str.lower()) &
                (~isp_offers_with_priority['campaign_name'].isin(used_offers)) &
                (~isp_offers_with_priority['campaign_name'].isin([c['campaign_name'] for c in recent_combinations if c['isp_name'].lower() == isp.lower()]))
            ].sort_values(by='priority', ascending=True)
            
            new_offer_total_volume = 0
            new_offer_count = 0
            min_new_offers = 2 if projection_volume >= 1500000 else 0
            for _, offer in new_offers.head(new_limit).iterrows():
                if offer['campaign_name'] in used_offers or new_offer_total_volume >= 500000 or current_volume >= isp_projection_volume or offer_count >= max_offers or new_offer_count >= min_new_offers:
                    continue
                
                datafiles = working_sheet16[working_sheet16['ISP Name'].str.lower() == isp.lower()].copy()
                datafiles = datafiles[~datafiles['Data File'].isin(used_datafiles)]
                datafiles = pd.merge(datafiles, working_df[['isp_name', 'datafile', 'clicks', 'cpm']],
                                    left_on=['ISP Name', 'Data File'], right_on=['isp_name', 'datafile'], how='left').fillna(0)
                
                selected_files = []
                for category in ['CLS', 'OPS', 'Normal']:
                    category_files = datafiles[datafiles['File Category'] == category]
                    if not category_files.empty:
                        if category == 'CLS':
                            top_file = category_files.sort_values(by='clicks', ascending=False).head(1)
                        else:
                            top_file = category_files.sort_values(by='cpm', ascending=False).head(1)
                        if can_use_datafile(top_file['Data File'].iloc[0]) and not is_recent_combination(
                            offer['campaign_name'], top_file['Data File'].iloc[0], isp, recent_combinations, exclude_days
                        ):
                            selected_files.append(top_file)
                
                if not selected_files:
                    continue
                
                valid_datafiles = pd.concat(selected_files).drop_duplicates()
                for _, datafile in valid_datafiles.iterrows():
                    if new_offer_total_volume + datafile['DF Count'] > 500000 or current_volume + datafile['DF Count'] > isp_projection_volume:
                        continue
                    rec = {
                        'campaign_name': offer['campaign_name'],
                        'datafile': datafile['Data File'],
                        'cpm': offer.get('cpm', 0),
                        'volume': datafile['DF Count'],
                        'priority': offer['priority'],
                        'isp_name': isp,
                        'sponsor': offer.get('sponsor', 'Unknown'),
                        'category': 'New Test Offer',
                        'last_send_date': offer.get('offer_date', '').strftime('%Y-%m-%d') if pd.notnull(offer.get('offer_date')) else 'N/A',
                        'file_type': datafile['FILE TYPE'],
                        'file_category': datafile['File Category']
                    }
                    isp_recommendations.append(rec)
                    current_volume += datafile['DF Count']
                    new_offer_total_volume += datafile['DF Count']
                    update_datafile_usage(datafile['Data File'])
                if new_offer_total_volume > 0:
                    used_offers.add(offer['campaign_name'])
                    offer_count += 1
                    new_offer_count += 1
                if new_offer_total_volume >= 500000 or current_volume >= isp_projection_volume or offer_count >= max_offers or (new_offer_count >= new_limit and new_offer_count >= min_new_offers):
                    break
        
        if projection_volume >= 1500000 and current_volume < isp_projection_volume and offer_count < max_offers:
            new_test_count = sum(1 for rec in isp_recommendations if rec['category'] == 'New Test Offer')
            if new_test_count < 2:
                continue
            
            other_isp_offers = working_df[
                (~working_df['isp_name'].str.lower().isin([isp.lower()])) &
                ((working_df['revenue'] > working_df['revenue'].quantile(0.75)) | (working_df['cpm'] > working_df['cpm'].quantile(0.75))) &
                (~working_df['campaign_name'].isin(used_offers)) &
                (~working_df['campaign_name'].isin([c['campaign_name'] for c in recent_combinations if c['isp_name'].lower() != isp.lower()]))
            ].drop_duplicates(subset=['campaign_name'])
            
            for _, offer in other_isp_offers.iterrows():
                if offer['campaign_name'] in used_offers or current_volume >= isp_projection_volume or offer_count >= max_offers:
                    continue
                
                datafiles = working_sheet16[working_sheet16['ISP Name'].str.lower() == isp.lower()].copy()
                datafiles = datafiles[~datafiles['Data File'].isin(used_datafiles)]
                datafiles = pd.merge(datafiles, working_df[['isp_name', 'datafile', 'cpm']],
                                    left_on=['ISP Name', 'Data File'], right_on=['isp_name', 'datafile'], how='left').fillna(0)
                valid_datafiles = datafiles.sort_values(by='cpm', ascending=False).head(4)
                
                isp_volume = 0
                for _, datafile in valid_datafiles.iterrows():
                    if not can_use_datafile(datafile['Data File']) or is_recent_combination(offer['campaign_name'], datafile['Data File'], isp, recent_combinations, exclude_days):
                        continue
                    if projection_volume < 1500000 and isp_volume + datafile['DF Count'] > 10000:
                        continue
                    if current_volume + datafile['DF Count'] <= isp_projection_volume:
                        rec = {
                            'campaign_name': offer['campaign_name'],
                            'datafile': datafile['Data File'],
                            'cpm': offer.get('cpm', 0),
                            'volume': datafile['DF Count'],
                            'priority': 99,
                            'isp_name': isp,
                            'sponsor': offer.get('sponsor', 'Unknown'),
                            'category': 'Other ISP Selected Offer',
                            'last_send_date': offer.get('offer_date', '').strftime('%Y-%m-%d') if pd.notnull(offer.get('offer_date')) else 'N/A',
                            'file_type': datafile['FILE TYPE'],
                            'file_category': datafile['File Category']
                        }
                        isp_recommendations.append(rec)
                        current_volume += datafile['DF Count']
                        isp_volume += datafile['DF Count']
                        update_datafile_usage(datafile['Data File'])
                    if current_volume >= isp_projection_volume or isp_volume >= 10000 or len([r for r in isp_recommendations if r['campaign_name'] == offer['campaign_name']]) >= 4:
                        break
                if isp_volume > 0:
                    used_offers.add(offer['campaign_name'])
                    offer_count += 1
                if current_volume >= isp_projection_volume or offer_count >= max_offers:
                    break
        
        remaining_volume = isp_projection_volume - current_volume
        if remaining_volume > 0 and offer_count < max_offers:
            available_offers = isp_offers_with_priority[~isp_offers_with_priority['campaign_name'].isin(used_offers)]
            for _, offer in available_offers.iterrows():
                if current_volume >= isp_projection_volume or offer_count >= max_offers:
                    break
                
                datafiles = working_sheet16[working_sheet16['ISP Name'].str.lower() == isp.lower()].copy()
                datafiles = datafiles[~datafiles['Data File'].isin(used_datafiles)]
                datafiles = pd.merge(datafiles, working_df[['isp_name', 'datafile', 'cpm']],
                                    left_on=['ISP Name', 'Data File'], right_on=['isp_name', 'datafile'], how='left').fillna(0)
                valid_datafiles = datafiles.sort_values(by='cpm', ascending=False).head(5)
                
                if valid_datafiles.empty:
                    continue
                
                datafile_count = 0
                category = 'Additional Historical Offer' if pd.to_datetime(offer['offer_date']) < cutoff_date else 'Additional Recent Offer'
                for _, datafile in valid_datafiles.iterrows():
                    if not can_use_datafile(datafile['Data File']) or is_recent_combination(offer['campaign_name'], datafile['Data File'], isp, recent_combinations, exclude_days):
                        continue
                    if current_volume + datafile['DF Count'] <= isp_projection_volume and datafile_count < 5:
                        rec = {
                            'campaign_name': offer['campaign_name'],
                            'datafile': datafile['Data File'],
                            'cpm': offer.get('cpm', 0),
                            'volume': datafile['DF Count'],
                            'priority': offer['priority'],
                            'isp_name': isp,
                            'sponsor': offer.get('sponsor', 'Unknown'),
                            'category': category,
                            'last_send_date': offer.get('offer_date', '').strftime('%Y-%m-%d') if pd.notnull(offer.get('offer_date')) else 'N/A',
                            'file_type': datafile['FILE TYPE'],
                            'file_category': datafile['File Category']
                        }
                        isp_recommendations.append(rec)
                        current_volume += datafile['DF Count']
                        update_datafile_usage(datafile['Data File'])
                        datafile_count += 1
                    if current_volume >= isp_projection_volume or datafile_count >= 5:
                        break
                if datafile_count > 0:
                    used_offers.add(offer['campaign_name'])
                    offer_count += 1
        
        recommendations.extend(isp_recommendations)
    
    recommendations_df = pd.DataFrame(recommendations)
    if recommendations_df.empty:
        st.error("No recommendations generated.")
        return recommendations_df
    
    recommendations_df = recommendations_df.drop_duplicates(subset=['campaign_name', 'datafile', 'isp_name'])
    
    for isp in selected_isps:
        isp_recs = recommendations_df[recommendations_df['isp_name'] == isp]
        total_volume = isp_recs['volume'].sum()
        offer_counts = isp_recs['campaign_name'].nunique()
        
        if total_volume > projection_volume:
            isp_recs = isp_recs.sort_values(by=['cpm', 'volume'], ascending=False)
            current_vol = 0
            filtered_recs = []
            for _, rec in isp_recs.iterrows():
                if current_vol + rec['volume'] <= projection_volume:
                    filtered_recs.append(rec)
                    current_vol += rec['volume']
            recommendations_df = recommendations_df[recommendations_df['isp_name'] != isp]
            recommendations_df = pd.concat([recommendations_df, pd.DataFrame(filtered_recs)], ignore_index=True)
        
        if offer_counts < min_offers:
            st.warning(f"For ISP {isp}: Generated {offer_counts} offers, which is less than the minimum {min_offers} required.")
        elif offer_counts > max_offers:
            top_offers = isp_recs.groupby('campaign_name').head(1).sort_values(by='cpm', ascending=False).head(max_offers)
            recommendations_df = recommendations_df[recommendations_df['isp_name'] != isp]
            recommendations_df = pd.concat([recommendations_df, top_offers], ignore_index=True)
    
    st.session_state.recommendations_df = recommendations_df
    return recommendations_df

# Prediction Logic
if start_button:
    with results_container:
        with st.spinner("Fetching data from APIs..."):
            report_api_url = "https://httransfer.com/edsmobiver/index.php?r=apiv1/report&api_key=2025-edsapi-67890-ljwllli"
            campaigns_api_url = "https://httransfer.com/edsmobiver/index.php?r=apiv1/campaigns&api_key=2025-edsapi-67890-ljwllli"
            
            report_data = fetch_data(report_api_url)
            campaigns_data = fetch_data(campaigns_api_url)
            
            if not report_data or not campaigns_data:
                st.error("Failed to fetch required data. Please check the API URLs and try again.")
                st.stop()
            
            try:
                report_df = pd.DataFrame(report_data)
                campaigns_df = pd.DataFrame(campaigns_data)
            except Exception as e:
                st.error(f"Error converting API data to DataFrames: {e}")
                st.stop()
        
        with st.spinner("Loading Sheet16 data..."):
            try:
                sheet16_df = pd.read_excel(r"C:\Users\User.Think-EDS-37\Desktop\Learning\working code REC\src\Backup\Sheet16.xlsx")
                st.session_state.sheet16_df = sheet16_df
            except Exception as e:
                st.error(f"Error loading Sheet16.xlsx: {e}")
                st.stop()
        
        if not selected_isps:
            st.error("Please select at least one ISP or 'All ISPs'.")
            st.stop()
        if not selected_sponsors:
            st.error("Please select at least one Sponsor or 'All Sponsors'.")
            st.stop()
        if not prediction_basis:
            st.error("Please select at least one prediction basis (Revenue or CPM).")
            st.stop()
        
        with st.spinner("Preprocessing data..."):
            merged_df, campaigns_df, sheet16_df = preprocess_data(report_df, campaigns_df, sheet16_df)
            if merged_df is None:
                st.error("Preprocessing failed. Please check the data and try again.")
                st.stop()
            st.session_state.merged_df = merged_df
            st.session_state.campaigns_df = campaigns_df
            st.session_state.sheet16_df = sheet16_df
            st.session_state.data_loaded = True
        
        with st.spinner("Training model and generating recommendations..."):
            recommendations = train_and_predict_recommendations(
                merged_df,
                sheet16_df,
                projection_volume,
                exclude_days,
                recent_combinations,
                prediction_basis,
                selected_isps,
                selected_sponsors
            )
        
        if recommendations.empty:
            st.error("No recommendations generated. Check data, projection volume, or filters.")
            st.stop()
        
        # Send recommendations to API automatically
        with st.spinner("Sending recommendations to API..."):
            send_to_api(recommendations)
        
        basis_text = " and ".join(prediction_basis)
        isp_text = ", ".join(selected_isps)
        sponsor_text = ", ".join(selected_sponsors)
        st.markdown(f"<div class='subheader'>Recommended Offer-Datafile Pairs for Today (April 16, 2025) - Based on {basis_text} for ISPs: {isp_text} and Sponsors: {sponsor_text}</div>", unsafe_allow_html=True)
        recommendations = recommendations.rename(columns={'category': 'Category', 'file_type': 'File Type'})
        
        unique_offers = recommendations.groupby('campaign_name').agg({
            'isp_name': 'first',
            'volume': 'sum',
            'cpm': 'mean',
            'sponsor': 'first',
            'Category': 'first',
            'last_send_date': 'first'
        }).reset_index()
        
        for _, offer in unique_offers.iterrows():
            offer_recs = recommendations[recommendations['campaign_name'] == offer['campaign_name']].sort_values(by='volume', ascending=False)
            if len(offer_recs) > 5:
                st.warning(f"Offer {offer['campaign_name']} exceeds 5 data files. Limiting to 5.")
                offer_recs = offer_recs.head(5)
            
            with st.container():
                st.markdown(f"""
                    <div class='recommendation-box'>
                        <strong>Offer: {offer['campaign_name']}</strong><br>
                        ISP: {offer['isp_name']} | Sponsor: {offer['sponsor']} | Total Volume: {offer['volume']:,.0f} | Category: {offer['Category']} | Last Send Date: {offer['last_send_date']}<br>
                        <strong>Data Files:</strong>
                        <ul>
                """, unsafe_allow_html=True)
                
                for _, row in offer_recs.iterrows():
                    st.markdown(f"""
                        <li>{row['datafile']} (CPM: {row['cpm']:.2f}, Volume: {row['volume']:,.0f}, Priority: {row['priority']}, File Type: {row['File Type']})</li>
                    """, unsafe_allow_html=True)
                
                st.markdown("</ul></div>", unsafe_allow_html=True)
        
        # Optional button to save recommendations locally
        if st.button("Save Recommendations to Local Database"):
            save_to_db(st.session_state.recommendations_df)
        
        st.success("Recommendations generated successfully!")
        st.markdown("---")
        st.markdown("**GFGTS** | Built with ❤️ using Streamlit")

if not start_button and not st.session_state.data_loaded:
    st.info("Configure the parameters in the sidebar and click **Start Prediction** to generate recommendations.")
