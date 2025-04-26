import time
import requests
import os
from dotenv import load_dotenv
import json
import logging

load_dotenv()
TIARA_API_KEY = os.getenv("TIARA_API_KEY")
TIARA_SENDER_ID = os.getenv("TIARA_SENDER_ID")
TIARA_API_ENDPOINT = os.getenv("TIARA_API_ENDPOINT")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")


# Set up logging
logger = logging.getLogger(__name__)
# Sample tools for function calling
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    print(f"Fetching weather for {location}")
    return f"The weather in {location} is currently sunny with a temperature of 72°F."

def get_time(timezone: str = "UTC") -> str:
    """Get the current time in a specific timezone."""
    current_time = time.strftime("%H:%M:%S", time.gmtime())
    return f"The current time in {timezone} is {current_time}."

def remember_fact(fact: str) -> str:
    """Remember a fact about the user."""
    return f"I'll remember that {fact}"

def send_sms(message: str, to: str, ):
    """Send an SMS using the TiaraConnect API."""
    logger.info(f"Sending SMS to {to}: {message}")
    print(f"Sending SMS to {to}: {message}")
    request_data = {
        "to": to,
        "from": os.getenv("TIARA_SENDER_ID"),
        "message": message
    }
    request_body = json.dumps(request_data)
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer eyJhbGciOiJIUzUxMiJ9.eyJzdWIiOiIzMDciLCJvaWQiOjMwNywidWlkIjoiZGY5NTQ0ZDctYzUwNi00MGE2LTg5NjYtN2E0NmMzYTAxN2YyIiwiYXBpZCI6NDM4LCJpYXQiOjE3MzcyODcyODQsImV4cCI6MjA3NzI4NzI4NH0.k1Q63Min6Q6rjtwffmMeuQxFWzwZuSmr5rgncNRMuabmO2Bz5DB2qO7W4VuYI05u0Yy6npIe5s7Xsl_awt58Ww"  # Added Bearer prefix
    }

    try:
        # Make the API request
        response = requests.post(
            url=TIARA_API_ENDPOINT,
            headers=headers,
            json=request_data,  # Using json parameter instead of data
            timeout=30
        )

        # Check if we got a successful response
        response.raise_for_status()
        
        # Parse response if we have content
        if response.text.strip():
            return response.json()
        else:
            return {"status": "SUCCESS", "message": "Request processed but no content returned"}
            
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {str(e)}")
        if hasattr(e, 'response'):
            if hasattr(e.response, 'status_code'):
                logging.error(f"Status code: {e.response.status_code}")
            if hasattr(e.response, 'text'):
                logging.error(f"Response text: {e.response.text}")
        raise


def search_serper(query):
    print(f"Searching for: {query}")
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "q": query
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise error for bad responses
        return response.json()
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None
    