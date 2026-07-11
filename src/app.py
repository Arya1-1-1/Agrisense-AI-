import os
import re
import json
import sqlite3
import secrets
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
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

# --- Auth: SQLite-backed accounts + token sessions ---
DB_PATH = os.path.join(os.path.dirname(__file__), "users.db")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


init_db()


def get_username_from_token(req):
    """Reads 'Authorization: Bearer <token>' header, returns username or None."""
    auth_header = req.headers.get('Authorization', '')
    if not auth_header.startswith('Bearer '):
        return None
    token = auth_header.split(' ', 1)[1].strip()
    if not token:
        return None
    conn = get_db()
    row = conn.execute("SELECT username FROM sessions WHERE token = ?", (token,)).fetchone()
    conn.close()
    return row['username'] if row else None


@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Invalid request body."}), 400

    username = (data.get('username') or '').strip()
    password = data.get('password') or ''

    if len(username) < 3:
        return jsonify({"error": "Username must be at least 3 characters."}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters."}), 400

    password_hash = generate_password_hash(password)
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, password_hash, datetime.utcnow().isoformat())
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({"error": "Username already taken."}), 409
    conn.close()

    return jsonify({"success": True, "message": "Account created. Please log in."}), 201


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Invalid request body."}), 400

    username = (data.get('username') or '').strip()
    password = data.get('password') or ''

    conn = get_db()
    row = conn.execute("SELECT password_hash FROM users WHERE username = ?", (username,)).fetchone()

    if not row or not check_password_hash(row['password_hash'], password):
        conn.close()
        return jsonify({"error": "Invalid username or password."}), 401

    token = secrets.token_hex(24)
    conn.execute(
        "INSERT INTO sessions (token, username, created_at) VALUES (?, ?, ?)",
        (token, username, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()

    return jsonify({"success": True, "token": token, "username": username}), 200


@app.route('/logout', methods=['POST'])
def logout():
    auth_header = request.headers.get('Authorization', '')
    if auth_header.startswith('Bearer '):
        token = auth_header.split(' ', 1)[1].strip()
        conn = get_db()
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()
        conn.close()
    return jsonify({"success": True}), 200


@app.route('/verify', methods=['GET'])
def verify():
    username = get_username_from_token(request)
    if username is None:
        return jsonify({"valid": False}), 401
    return jsonify({"valid": True, "username": username}), 200

# --- ICRISAT real yield data (2017-2019 avg, from District Level Database) ---
# File should sit alongside app.py: data/haryana_crop_yield.json
YIELD_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "haryana_crop_yield.json")
try:
    with open(YIELD_DATA_PATH, encoding="utf-8") as f:
        CROP_YIELD_DATA = json.load(f)
except FileNotFoundError:
    CROP_YIELD_DATA = {}
    print(f"WARNING: {YIELD_DATA_PATH} not found - yield context will be skipped.")

# Dashboard dropdown names -> ICRISAT dataset names (only where they differ)
DISTRICT_ALIASES = {
    "Hisar": "Hissar",
    "Sonipat": "Sonepat",
    "Nuh": "Mewat",
}

CROP_ALIASES = {
    "Mustard": "Rapeseed And Mustard",
    "Bajra": "Pearl Millet",
    "Gram": "Chickpea",
    "Jowar": "Sorghum",
}


def get_yield_context(district, crop):
    """Return a real ICRISAT yield/trend string for this district+crop, or None if unavailable."""
    d = DISTRICT_ALIASES.get(district, district)
    c = CROP_ALIASES.get(crop, crop)
    entry = CROP_YIELD_DATA.get(d, {}).get(c)
    if not entry:
        return None
    return (
        f"Historical yield (ICRISAT, 2017-2019 avg): {entry['avg_yield_t_per_ha']} tonnes/hectare, "
        f"grown on ~{entry['avg_area_000ha']}k hectares, yield trend: {entry['trend']}"
    )

NDVI_MAP = {
    'Karnal': 0.40, 'Hisar': 0.20, 'Rohtak': 0.35,
    'Panipat': 0.30, 'Ambala': 0.45, 'Gurgaon': 0.25,
    'Jind': 0.28, 'Fatehabad': 0.22,
    'Ambala': 0.45, 'Bhiwani': 0.24, 'Charkhi Dadri': 0.26,
    'Faridabad': 0.27, 'Kaithal': 0.38, 'Kurukshetra': 0.42,
    'Mahendragarh': 0.23, 'Nuh': 0.21, 'Palwal': 0.29,
    'Panchkula': 0.44, 'Rewari': 0.25, 'Sirsa': 0.20,
    'Sonipat': 0.33, 'Yamunanagar': 0.41, 'Jhajjar': 0.31
}

NDWI_MAP = {
    'Karnal': -0.10, 'Hisar': -0.30, 'Rohtak': -0.20,
    'Panipat': -0.20, 'Ambala': -0.05, 'Gurgaon': -0.35,
    'Jind': -0.25, 'Fatehabad': -0.32,
    'Bhiwani': -0.34, 'Charkhi Dadri': -0.31,
    'Faridabad': -0.28, 'Kaithal': -0.12, 'Kurukshetra': -0.08,
    'Mahendragarh': -0.36, 'Nuh': -0.38, 'Palwal': -0.22,
    'Panchkula': -0.06, 'Rewari': -0.33, 'Sirsa': -0.29,
    'Sonipat': -0.18, 'Yamunanagar': -0.09, 'Jhajjar': -0.19
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


@app.route('/districts', methods=['GET'])
def get_districts():
    """Returns the list of Haryana districts with real ICRISAT crop data."""
    return jsonify({"districts": sorted(CROP_YIELD_DATA.keys())})


@app.route('/crops', methods=['GET'])
def get_crops():
    """Returns crops grown in a given district (query param ?district=...), or all crops if omitted."""
    district = request.args.get('district')
    if district:
        d = DISTRICT_ALIASES.get(district, district)
        crops = sorted(CROP_YIELD_DATA.get(d, {}).keys())
    else:
        crops = sorted({c for crops in CROP_YIELD_DATA.values() for c in crops})
    return jsonify({"crops": crops})


@app.route('/advisory', methods=['POST'])
def get_advisory():
    username = get_username_from_token(request)
    if username is None:
        return jsonify({"error": "Please log in to get an advisory."}), 401

    # Safely parse JSON body - never crash even if Content-Type header is missing/wrong
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({
            "advisory": "Could not read your request. Please try again.",
            "error": "Request body was not valid JSON."
        }), 200

    crop = data.get('crop', '')
    district = data.get('district', '')
    question = data.get('question', '')

    if not question or not question.strip():
        return jsonify({
            "advisory": "1. What is the problem: Please enter a question about your crop.\n2. What action to take immediately: Type your question in the box above.\n3. What to watch for in next 7 days: N/A"
        }), 200

    ndvi = NDVI_MAP.get(district, 0.3)
    ndwi = NDWI_MAP.get(district, -0.2)
    yield_context = get_yield_context(district, crop)
    yield_line = f"- {yield_context}\n" if yield_context else ""

    prompt = f"""You are an expert agricultural advisor for Indian farmers.

Farmer details:
- Crop: {crop}
- District: {district}, Haryana
- Satellite NDVI value: {ndvi} (0-1 scale, below 0.3 means stressed crop)
- Satellite NDWI value: {ndwi}
- Moisture status: {get_moisture_status(ndwi)}
- Current weather: {get_weather(district)}
{yield_line}- Farmer's question: {question}

Respond with ONLY a valid JSON object, no extra text, no markdown code fences:

{{"problem": "one sentence describing the problem", "action": "one sentence describing immediate action to take", "watch": "one sentence describing what to watch for in the next 7 days"}}"""

    payload = {
        "model": "accounts/fireworks/models/glm-5p2",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1200,
        "reasoning_effort": "none"
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