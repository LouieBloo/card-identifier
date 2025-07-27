# services/scryfall_client.py
import requests
import config

def fetch_card_by_scryfall_id(scryfall_id: str):
    """Fetches card data from Scryfall using its Scryfall ID."""
    url = f"{config.SCRYFALL_API_URL}/cards/{scryfall_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Scryfall ID {scryfall_id}: {e}")
        return {"error": "Failed to fetch card details from Scryfall"}

def fetch_card_by_oracle_id(oracle_id: str):
    """Fetches card data from Scryfall using its Oracle ID."""
    url = f"{config.SCRYFALL_API_URL}/cards/search"
    params = {'q': f'oracleid:{oracle_id}'}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if data.get('total_cards', 0) > 0:
            return data['data'][0]
        return {"error": "No card found with this Oracle ID"}
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Oracle ID {oracle_id}: {e}")
        return {"error": "Failed to fetch card details from Scryfall"}