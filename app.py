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

DISEASE_MODEL = (
    "https://router.huggingface.co/hf-inference/models/"
    "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


# ---------------- COMMON HELPERS ----------------

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
                "Use fertilizer according to soil testing "
                "and the crop's growth stage."
            ),
            "quantity": (
                "A fertilizer quantity cannot be recommended "
                "from this image alone."
            ),
            "prevention": (
                "Use healthy planting material, maintain field hygiene, "
                "and inspect leaves regularly."
            )
        }

    return {
        "treatment": (
            "This is an AI prediction, not a confirmed diagnosis. "
            "Check symptoms with a local agricultural officer "
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
            "Monitor the crop, maintain field hygiene, and seek "
            "local agricultural advice."
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


# ---------------- CHATBOT ----------------

@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")


# ---------------- CROP CALENDAR ----------------

@app.route("/crop-calendar", methods=["GET", "POST"])
def crop_calendar():
    calendar = None
    message = None

    if request.method == "POST":
        crop = request.form.get("crop", "").strip()
        season = request.form.get("season", "").strip()
        district = request.form.get("district", "").strip()

        if not crop or not season or not district:
            message = "Please enter the crop, season and district."
        else:
            calendar = [
                {
                    "stage": "Land preparation",
                    "details": (
                        "Prepare the field and check soil condition "
                        "before planting."
                    )
                },
                {
                    "stage": "Sowing or planting",
                    "details": (
                        "Use suitable seeds and follow local "
                        "agricultural recommendations."
                    )
                },
                {
                    "stage": "Crop growth",
                    "details": (
                        "Monitor water, weeds, pests and crop health."
                    )
                },
                {
                    "stage": "Harvest",
                    "details": (
                        "Harvest when the crop reaches the appropriate "
                        "maturity stage."
                    )
                }
            ]

            message = (
                f"General calendar for {crop} in {district}, "
                f"season: {season}. This is not a date-specific forecast."
            )

    return render_template(
        "calendar.html",
        calendar=calendar,
        message=message
    )


# ---------------- MARKET PRICES ----------------

@app.route("/market-prices", methods=["GET", "POST"])
def market_prices():
    error = (
        "Live market prices are not connected yet. "
        "Please check an official agricultural market source."
    )

    return render_template(
        "market_prices.html",
        error=error,
        prices=None
    )


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

    # Use a unique filename to avoid collisions between uploads.
    filename = f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{filename}"
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
        confidence = round(float(result.get("score", 0)) * 100, 2)
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

    return render_template(
        "pest_doctor.html",
        error=(
            "The photo was uploaded, but AI pest detection is not "
            "connected yet. Please consult an agricultural officer."
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
        error=(
            "Soil image analysis is not connected yet. "
            "Please use a soil test for reliable results."
        )
    )


# ---------------- PROFIT CALCULATOR ----------------

@app.route("/profit-calculator", methods=["GET", "POST"])
def profit_calculator():
    if request.method == "POST":
        try:
            crop = request.form.get("crop", "").strip()
            area = float(request.form.get("area", 0))
            investment = float(request.form.get("investment", 0))
            yield_amount = float(request.form.get("yield_amount", 0))
            sale_price = float(request.form.get("sale_price", 0))

            if not crop:
                raise ValueError("Please select a crop.")

            if (
                area <= 0
                or investment < 0
                or yield_amount < 0
                or sale_price < 0
            ):
                raise ValueError("Please enter valid non-negative values.")

            revenue = yield_amount * sale_price
            total_cost = investment
            profit = revenue - total_cost

            return render_template(
                "profit_calculator.html",
                crop=crop,
                area=area,
                yield_amount=yield_amount,
                revenue=revenue,
                total_cost=total_cost,
                profit=profit
            )

        except (ValueError, TypeError):
            return render_template(
                "profit_calculator.html",
                error=(
                    "Enter a crop, a positive cultivated area, "
                    "and valid non-negative expense, yield and price values."
                )
            )

    return render_template("profit_calculator.html")


# ---------------- TELUGU VOICE ASSISTANT ----------------

@app.route("/voice-assistant")
def voice_assistant():
    try:
        return render_template("voice_assistant.html")
    except Exception:
        return (
            "Voice Assistant page is not available yet. "
            "Create templates/voice_assistant.html first.",
            404
        )


# ---------------- GOVERNMENT SCHEMES ----------------

@app.route("/government-schemes", methods=["GET", "POST"])
def government_schemes():
    state = request.values.get("state", "Andhra Pradesh").strip()
    category = request.values.get("category", "all").strip().lower()

    all_schemes = [
        {
            "name": "PM-KISAN",
            "category": "financial",
            "description": (
                "A central government income-support scheme for "
                "eligible landholding farmer families, subject to rules."
            ),
            "eligibility": (
                "Eligibility and exclusions are determined by official "
                "scheme guidelines."
            ),
            "benefit": (
                "Financial assistance under the current scheme rules."
            ),
            "url": "https://pmkisan.gov.in/"
        },
        {
            "name": "Pradhan Mantri Fasal Bima Yojana",
            "category": "insurance",
            "description": (
                "Crop insurance support for eligible notified crops "
                "and areas, subject to enrolment and scheme rules."
            ),
            "eligibility": (
                "Check current season, crop, area and enrolment rules."
            ),
            "benefit": (
                "Insurance protection against covered crop losses."
            ),
            "url": "https://pmfby.gov.in/"
        },
        {
            "name": "Soil Health Card",
            "category": "soil",
            "description": (
                "Helps farmers understand soil nutrient status and "
                "receive soil-management recommendations."
            ),
            "eligibility": (
                "Contact the local agriculture department for available "
                "soil testing services."
            ),
            "benefit": (
                "Soil test information to support fertilizer decisions."
            ),
            "url": "https://soilhealth.dac.gov.in/"
        }
    ]

    if category == "all":
        schemes = all_schemes
    else:
        schemes = [
            scheme for scheme in all_schemes
            if scheme["category"] == category
        ]

    return render_template(
        "government_schemes.html",
        state=state,
        category=category,
        schemes=schemes
    )


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


# ---------------- GENERAL ERROR HANDLER ----------------

@app.errorhandler(500)
def internal_server_error(error):
    app.logger.exception("Internal server error")
    return (
        "Something went wrong on the server. Please try again later.",
        500
    )


# ---------------- RUN APPLICATION ----------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
                    )
