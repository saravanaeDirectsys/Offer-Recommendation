import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def send_to_api(recommendations_df, api_url="http://127.0.0.1:8000/store-recommendations"):
    logger.info("Sending recommendations to API")
    try:
        import requests  # Ensure the requests library is imported
        response = requests.post(api_url, json=recommendations_df.to_dict(orient="records"))
        logger.info(f"API response: {response.json()}")
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        # ...