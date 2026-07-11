import os
import re
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import requests

load_dotenv()
api_key = os.getenv("FIREWORKS_API_KEY")
weather_api_key = os.getenv("OPENWEATHER_API_KEY")

app = Flask(__name__)
CORS(app)

url = "https://api.fireworks.ai/inference/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

NDVI_MAP = {
    'Karnal': 0.4, 'Hisar': 0.2, 'Rohtak': 0.35,
    'Panipat': 0.3, 'Ambala': 0.45, 'Gurgaon': 0.25,
    'Jind': 0.28, 'Fatehabad': 0.22
}

NDWI_MAP = {
    'Karnal': -0.1, 'Hisar': -0.3, 'Rohtak': -0.2,
    'Panipat': -0.2, 'Ambala': -0.05, 'Gurgaon': -0.35,
    'Jind': -0.25, 'Fatehabad': -0.32
}

def get_moisture_status(ndwi):
    if ndwi < -0.1:
        return "SEVERELY DRY - irrigate immediately"
    elif ndwi < 0.1:
        return "DRY - irrigation needed soon"
    elif ndwi < 0.3:
        return "MODERATE - monitor closely"
    else:
        return "ADEQUATE - no immediate action needed"

def get_weather(district):
    try:
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?q={district},Haryana,IN&appid={weather_api_key}&units=metric"
        response = requests.get(weather_url, timeout=10)
        data = response.json()
        if response.status_code == 200:
            temp = data['main']['temp']
            humidity = data['main']['humidity']
            rain = "rain expected" if 'rain' in data else "no rain expected"
            return f"Temperature: {temp}°C, Humidity: {humidity}%, {rain}"
        else:
            return "Weather data unavailable"
    except Exception:
        return "Weather data unavailable"

def extract_json(raw_text):
    # Strip markdown code fences if present
    text = re.sub(r'```json|```', '', raw_text).strip()
    # Find the first {...} block in case there's extra text around it
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        text = match.group(0)
    try:
        return json.loads(text)
    except Exception:
        return None

@app.route('/advisory', methods=['POST'])
def get_advisory():
    data = request.json
    crop = data.get('crop')
    district = data.get('district')
    question = data.get('question')

    ndvi = NDVI_MAP.get(district, 0.3)
    ndwi = NDWI_MAP.get(district, -0.2)

    prompt = f"""You are an expert agricultural advisor for Indian farmers.

Farmer details:
- Crop: {crop}
- District: {district}, Haryana
- Satellite NDVI value: {ndvi} (0-1 scale, below 0.3 means stressed crop)
- Satellite NDWI value: {ndwi}
- Moisture status: {get_moisture_status(ndwi)}
- Current weather: {get_weather(district)}
- Farmer's question: {question}

Respond with ONLY a valid JSON object, nothing else before or after. No markdown, no code fences, no explanation. Use exactly this structure:

{{"problem": "one sentence describing the problem", "action": "one sentence describing immediate action to take", "watch": "one sentence describing what to watch for in the next 7 days"}}"""

    payload = {
        "model": "accounts/fireworks/models/glm-5p2",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1200
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        result = response.json()
        raw_advisory = result['choices'][0]['message']['content']
    except Exception as e:
        return jsonify({
            "advisory": "Advisory service is temporarily unavailable. Please try again in a moment.",
            "error": str(e)
        }), 200

    parsed = extract_json(raw_advisory)

    if parsed and 'problem' in parsed and 'action' in parsed and 'watch' in parsed:
        advisory = f"1. What is the problem: {parsed['problem']}\n2. What action to take immediately: {parsed['action']}\n3. What to watch for in next 7 days: {parsed['watch']}"
    else:
        # fallback - just clean up whatever text came back
        advisory = raw_advisory.replace('**', '').replace('*', '').replace('•', '').strip()

    return jsonify({"advisory": advisory})

if __name__ == '__main__':
    app.run(debug=True, port=5050)