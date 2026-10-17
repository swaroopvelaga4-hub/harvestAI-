from flask import Flask, render_template, request, jsonify
from datetime import datetime
from werkzeug.utils import secure_filename
import os
import requests

app = Flask(__name__)

app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY", "harvestai-development-key"
)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

UPLOAD_FOLDER = os.path.join(app.static_folder, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN")

# Hugging Face image classification model
DISEASE_MODEL = (
    "https://router.huggingface.co/hf-inference/models/"
    "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def classify_disease(image_path):
    if not HF_TOKEN:
        return None, (
            "AI service is not configured. "
            "Please add HF_TOKEN in Render Environment."
        )

    try:
        with open(image_path, "rb") as image_file:
            response = requests.post(
                DISEASE_MODEL,
                headers={
                    "Authorization": f"Bearer {HF_TOKEN}",
                    "Content-Type": "application/octet-stream"
                },
                data=image_file,
                timeout=60
            )

        if response.status_code != 200:
            return None, (
                "AI service is temporarily unavailable. "
                "Please try again later."
            )

        results = response.json()

        if not isinstance(results, list) or not results:
            return None, "AI could not identify the image."

        results = sorted(
            results,
            key=lambda item: item.get("score", 0),
            reverse=True
        )

        return results[0], None

    except requests.RequestException:
        return None, "Could not connect to the AI service."

    except (ValueError, OSError):
        return None, "Could not process the uploaded image."


def disease_advice(label):
    text = label.lower()

    if "healthy" in text:
        return {
            "treatment": (
                "The model predicts a healthy leaf. "
                "Continue regular crop monitoring."
            ),
            "fertilizer": (
                "Use fertilizer based on soil testing and "
                "the crop's growth stage."
            ),
            "quantity": (
                "No quantity can be recommended from this image alone."
            ),
            "prevention": (
                "Use healthy planting material, maintain field hygiene, "
                "and inspect leaves regularly."
            )
        }

    return {
        "treatment": (
            "This is an AI prediction, not a confirmed diagnosis. "
            "Check the symptoms with a local agricultural officer "
            "before selecting a treatment."
        ),
        "fertilizer": (
            "Choose fertilizer according to the crop, soil test, "
            "and confirmed diagnosis."
        ),
        "quantity": (
            "A safe quantity cannot be determined from the image alone."
        ),
        "prevention": (
            "Monitor the crop, remove severely affected plant material "
            "where appropriate, avoid unnecessary leaf wetness, "
            "and seek local agricultural advice."
        )
    }


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
        geo_response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={
                "name": city,
                "count": 1,
                "language": "en"
            },
            timeout=10
        )
        geo_response.raise_for_status()

        locations = geo_response.json().get("results", [])

        if not locations:
            return jsonify({"error": "Location not found"}), 404

        place = locations[0]

        weather_response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
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
            "rain_chance": daily.get(
                "precipitation_probability_max", [None]
            )[0],
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
                if crop != "Rice (where water is suitable)"
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

@app.route("/market-prices", methods=["GET", "POST"])
def market_prices():
    return render_template("market_prices.html")


# ---------------- CROP DISEASE DOCTOR ----------------

@app.route("/crop-disease")
def crop_disease():
    return render_template("crop_disease.html")


@app.route("/predict", methods=["POST"])
def predict():
    uploaded_file = request.files.get("image")
    crop = request.form.get("crop", "Unknown").strip()

    if not uploaded_file or not uploaded_file.filename:
        return render_template(
            "crop_disease.html",
            error="Please upload a leaf photo."
        )

    if not allowed_file(uploaded_file.filename):
        return render_template(
            "crop_disease.html",
            error="Upload a JPG, JPEG, PNG, or WEBP image."
        )

    filename = secure_filename(uploaded_file.filename)

    if not filename:
        return render_template(
            "crop_disease.html",
            error="Invalid filename. Please select another image."
        )

    image_path = os.path.join(UPLOAD_FOLDER, filename)

    try:
        uploaded_file.save(image_path)

        result, error = classify_disease(image_path)

        if error:
            return render_template(
                "crop_disease.html",
                error=error,
                crop=crop
            )

        label = result.get("label", "Unknown")
        confidence = round(
            float(result.get("score", 0)) * 100, 2
        )

        advice = disease_advice(label)

        return render_template(
            "crop_disease.html",
            crop=crop,
            crop_name=crop,
            disease=label,
            confidence=f"{confidence}%",
            treatment=advice["treatment"],
            fertilizer=advice["fertilizer"],
            recommended_quantity=advice["quantity"],
            prevention=advice["prevention"]
        )

    except OSError:
        return render_template(
            "crop_disease.html",
            error="Could not save the image. Please try again."
        )


# ---------------- PEST DOCTOR ----------------

@app.route("/pest-doctor")
def pest_doctor():
    return render_template("pest_doctor.html")


@app.route("/pest-identify", methods=["POST"])
def pest_identify():
    uploaded_file = request.files.get("pest_image")

    if not uploaded_file or not uploaded_file.filename:
        return render_template(
            "pest_doctor.html",
            error="Please upload a pest photo."
        )

    if not allowed_file(uploaded_file.filename):
        return render_template(
            "pest_doctor.html",
            error="Please upload a JPG, JPEG, PNG, or WEBP image."
        )

    # Pest AI is not connected in this version.
    return render_template(
        "pest_doctor.html",
        error=(
            "Photo uploaded, but pest AI detection is not connected yet. "
            "Please consult an agricultural officer for identification."
        )
    )


# ---------------- SOIL DOCTOR ----------------

@app.route("/soil-doctor")
def soil_doctor():
    return render_template("soil_doctor.html")


@app.route("/soil-detect", methods=["POST"])
def soil_detect():
    return render_template(
        "soil_doctor.html",
        error="Soil image analysis is not connected in this version."
    )


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


# ---------------- FILE SIZE ERROR ----------------

@app.errorhandler(413)
def file_too_large(error):
    return (
        "Photo is too large. Please upload an image under 10 MB.",
        413
    )


# ---------------- RUN APPLICATION ----------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
