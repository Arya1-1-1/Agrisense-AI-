import os
from dotenv import load_dotenv
import requests

load_dotenv()
api_key = os.getenv("FIREWORKS_API_KEY")

url = "https://api.fireworks.ai/inference/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

def get_crop_advisory(crop, district, ndvi, ndwi, farmer_query):
    prompt = f"""You are an expert agricultural advisor for Indian farmers.
    
Farmer details:
- Crop: {crop}
- District: {district}, Haryana
- Satellite NDVI value: {ndvi} (0-1 scale, below 0.3 means stressed crop)
- Satellite NDWI value: {ndwi} (negative means dry/water stressed)
- Farmer's question: {farmer_query}

Give specific advice in exactly 3 bullet points:
1. What is the problem
2. What action to take immediately
3. What to watch for in next 7 days"""

    payload = {
        "model": "accounts/fireworks/models/glm-5p2",
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post(url, headers=headers, json=payload)
    result = response.json()
    return result['choices'][0]['message']['content']

# Test with dummy data
advisory = get_crop_advisory(
    crop="Wheat",
    district="Karnal",
    ndvi=0.3,
    ndwi=-0.2,
    farmer_query="My wheat leaves are turning yellow"
)

print("=== AgriSense AI Advisory ===")
print(advisory)