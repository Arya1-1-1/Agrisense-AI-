import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import requests

load_dotenv()
api_key = os.getenv("FIREWORKS_API_KEY")

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
- Satellite NDWI value: {ndwi} (negative means dry/water stressed)
- Farmer's question: {question}

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
    advisory = result['choices'][0]['message']['content']
    return jsonify({"advisory": advisory})

if __name__ == '__main__':
    app.run(debug=True, port=5000)