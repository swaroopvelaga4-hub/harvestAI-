from flask import Flask, render_template, request, jsonify
from datetime import datetime
import os
import requests

app = Flask(__name__)

app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY", "harvestai-development-key"
)

# ---------------- HOME PAGE ----------------

@app.route("/")
def home():
    return render_template("index.html")


# ---------------- WEATHER ----------------

@app.route("/weather")
def weather():
    return render_template("weather.html")


@app.route("/api/weather")
def api_weather():
    city = request.args.get("city", "Vijayawada").strip()

    if not city:
        return jsonify({"error": "Please enter a location"}), 400

    try:
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"

        geo_response = requests.get(
            geo_url,
            params={"name": city, "count": 1, "language": "en"},
            timeout=10
        )
        geo_response.raise_for_status()

        locations = geo_response.json().get("results", [])

        if not locations:
            return jsonify({"error": "Location not found"}), 404

        place = locations[0]

        weather_url = "https://api.open-meteo.com/v1/forecast"

        weather_response = requests.get(
            weather_url,
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": (
                    "temperature_2m,relative_humidity_2m,"
                    "precipitation,wind_speed_10m,weather_code"
                ),
                "daily": "precipitation_probability_max",
                "timezone": "auto",
                "forecast_days": 1
            },
            timeout=10
        )
        weather_response.raise_for_status()

        data = weather_response.json()
        current = data.get("current", {})
        daily = data.get("daily", {})

        return jsonify({
            "location": place.get("name", city),
            "state": place.get("admin1", ""),
            "country": place.get("country", ""),
            "temperature": current.get("temperature_2m"),
            "humidity": current.get("relative_humidity_2m"),
            "rainfall": current.get("precipitation"),
            "wind_speed": current.get("wind_speed_10m"),
            "rain_chance": (
                daily.get("precipitation_probability_max", [None])[0]
            ),
            "weather_code": current.get("weather_code"),
            "source": "Open-Meteo"
        })

    except requests.RequestException:
        return jsonify({
            "error": "Weather service is temporarily unavailable"
        }), 503


# ---------------- CROP RECOMMENDATION ----------------

@app.route("/crop-recommendation", methods=["GET", "POST"])
def crop_recommendation():
    recommendations = None

    if request.method == "POST":
        soil = request.form.get("soil", "").strip().lower()
        season = request.form.get("season", "").strip().lower()

        crop_data = {
            "black": ["Cotton", "Sorghum", "Pulses"],
            "red": ["Groundnut", "Millets", "Pulses"],
            "sandy": ["Groundnut", "Millets"],
            "loamy": ["Maize", "Pulses"],
            "alluvial": ["Rice", "Maize", "Pulses"],
            "clay": ["Rice (where water is suitable)", "Pulses"]
        }

        recommendations = crop_data.get(
            soil,
            ["Please consult a local agricultural officer"]
        )

        if season == "winter":
            recommendations = [
                crop for crop in recommendations
                if crop not in ["Rice (where water is suitable)"]
            ]

    return render_template(
        "crop_recommendation.html",
        recommendations=recommendations
    )


# ---------------- FARMING CHATBOT ----------------

@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")


# ---------------- CROP CALENDAR ----------------

@app.route("/crop-calendar")
def crop_calendar():
    return render_template("calendar.html")


# ---------------- MARKET PRICES ----------------

@app.route("/market-prices")
def market_prices():
    return render_template("market_prices.html")


# ---------------- LEAF DOCTOR ----------------

@app.route("/crop-disease")
def crop_disease():
    return render_template("crop_disease.html")


# ---------------- PEST DOCTOR ----------------

@app.route("/pest-doctor")
def pest_doctor():
    return render_template("pest_doctor.html")


# ---------------- SOIL DOCTOR ----------------

@app.route("/soil-doctor")
def soil_doctor():
    return render_template("soil_doctor.html")


# ---------------- PROFIT CALCULATOR ----------------

@app.route("/profit-calculator")
def profit_calculator():
    return render_template("profit_calculator.html")


# ---------------- TELUGU VOICE ASSISTANT ----------------

@app.route("/voice-assistant")
def voice_assistant():
    return render_template("voice_assistant.html")


# ---------------- GOVERNMENT SCHEMES ----------------

@app.route("/government-schemes")
def government_schemes():
    return render_template("government_schemes.html")


# ---------------- FARMING VIDEOS ----------------

@app.route("/farming-videos")
def farming_videos():
    return render_template("farming_videos.html")


# ---------------- HEALTH CHECK ----------------

@app.route("/health")
def health():
    return jsonify({
        "app": "HarvestAI",
        "status": "running",
        "time": datetime.now().isoformat()
    })


# ---------------- RUN APPLICATION ----------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
      )
