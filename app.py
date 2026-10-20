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

DATA_GOV_IN_API_KEY = os.environ.get("DATA_GOV_IN_API_KEY")

MARKET_API_URL = (
    "https://api.data.gov.in/resource/"
    "9ef84268-d588-465a-a308-a864a43d0070"
)

DISEASE_MODEL = (
    "https://router.huggingface.co/hf-inference/models/"
    "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


# ---------------- COMMON HELPERS ----------------

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


def classify_disease(image_path):
    if not HF_TOKEN:
        app.logger.error("HF_TOKEN missing in Render Environment.")
        return None, "AI service is not configured. Add HF_TOKEN in Render."

    try:
        with open(image_path, "rb") as image_file:
            response = requests.post(
                DISEASE_MODEL,
                headers={
                    "Authorization": f"Bearer {HF_TOKEN}",
                    "Content-Type": "application/octet-stream"
                },
                data=image_file,
                timeout=90
            )

        if response.status_code != 200:
            app.logger.error(
                "Disease API HTTP %s: %s",
                response.status_code,
                response.text[:1000]
            )

            if response.status_code in (401, 403):
                return None, (
                    "AI authentication failed. Check your HF_TOKEN."
                )

            if response.status_code == 429:
                return None, (
                    "AI request limit reached. Try again later."
                )

            if response.status_code in (500, 502, 503, 504):
                return None, (
                    "AI model is temporarily unavailable. "
                    "Please try again later."
                )

            return None, (
                f"AI service error: HTTP {response.status_code}. "
                "Check Render Logs."
            )

        results = response.json()

        if not isinstance(results, list) or not results:
            app.logger.error(
                "Unexpected disease model response: %s",
                str(results)[:1000]
            )
            return None, "AI could not identify this image."

        valid_results = [
            item for item in results
            if isinstance(item, dict)
            and item.get("label")
            and isinstance(item.get("score"), (int, float))
        ]

        if not valid_results:
            return None, "AI returned no valid predictions."

        results = sorted(
            valid_results,
            key=lambda item: item["score"],
            reverse=True
        )

        return results[0], None

    except requests.Timeout:
        app.logger.exception("Disease API timed out.")
        return None, "AI service took too long. Please try again."

    except requests.RequestException:
        app.logger.exception("Disease API connection failed.")
        return None, "Could not connect to the AI service."

    except (ValueError, OSError):
        app.logger.exception("Disease image processing failed.")
        return None, "Could not process the uploaded image."


def disease_advice(label):
    text = label.lower()

    if "healthy" in text:
        return {
            "treatment": (
                "The AI predicts a healthy leaf. "
                "Continue monitoring the crop."
            ),
            "fertilizer": (
                "Choose fertilizer according to soil testing "
                "and crop requirements."
            ),
            "quantity": (
                "A fertilizer quantity cannot be determined "
                "from the image alone."
            ),
            "prevention": (
                "Use healthy planting material and inspect "
                "the crop regularly."
            )
        }

    return {
        "treatment": (
            "This is an AI prediction, not a confirmed diagnosis. "
            "Verify the symptoms with a local agricultural officer "
            "before applying pesticides or other treatments."
        ),
        "fertilizer": (
            "Select fertilizer according to the crop, soil test "
            "and confirmed diagnosis."
        ),
        "quantity": (
            "A safe quantity cannot be determined from the image alone."
        ),
        "prevention": (
            "Maintain field hygiene, monitor the crop regularly "
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
            timeout=15
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
            timeout=15
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
        app.logger.exception("Weather API failed.")
        return jsonify({
            "error": "Weather service is temporarily unavailable."
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
            "clay": ["Rice", "Pulses"]
        }

        recommendations = crop_data.get(
            soil,
            ["Consult a local agricultural officer."]
        )

        if season == "winter":
            recommendations = [
                crop for crop in recommendations
                if crop != "Rice"
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
            message = "Please enter crop, season and district."
        else:
            calendar = [
                {
                    "stage": "Land preparation",
                    "details": (
                        "Prepare the field and check soil condition."
                    )
                },
                {
                    "stage": "Sowing or planting",
                    "details": (
                        "Use suitable seeds and follow local advice."
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
                        "Harvest at the appropriate maturity stage."
                    )
                }
            ]

            message = (
                f"General calendar for {crop} in {district}, "
                f"season: {season}. Confirm dates locally."
            )

    return render_template(
        "calendar.html",
        calendar=calendar,
        message=message
    )


# ---------------- MARKET PRICES ----------------

@app.route("/market-prices", methods=["GET", "POST"])
def market_prices():
    prices = None
    error = None

    crop = request.values.get("crop", "").strip()
    state = request.values.get("state", "").strip()
    district = request.values.get("district", "").strip()

    if request.method == "POST":
        if not DATA_GOV_IN_API_KEY:
            error = (
                "Market API key is not configured. "
                "Add DATA_GOV_IN_API_KEY in Render Environment."
            )
        else:
            try:
                params = {
                    "api-key": DATA_GOV_IN_API_KEY,
                    "format": "json",
                    "limit": 100
                }

                if crop:
                    params["filters[commodity]"] = crop

                if state:
                    params["filters[state]"] = state

                if district:
                    params["filters[district]"] = district

                response = requests.get(
                    MARKET_API_URL,
                    params=params,
                    timeout=25
                )

                if response.status_code != 200:
                    app.logger.error(
                        "Market API HTTP %s: %s",
                        response.status_code,
                        response.text[:1000]
                    )
                    error = (
                        "Government market data is unavailable. "
                        "Check the API key, dataset and Render Logs."
                    )
                else:
                    data = response.json()
                    records = data.get("records", [])

                    if records:
                        prices = records
                    else:
                        error = (
                            "No market prices found for these filters. "
                            "Try another crop, state or district."
                        )

            except (requests.RequestException, ValueError):
                app.logger.exception("Market API request failed.")
                error = (
                    "Could not connect to the market data service. "
                    "Please try again later."
                )

    return render_template(
        "market_prices.html",
        prices=prices,
        error=error,
        crop=crop,
        state=state,
        district=district
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
            error="Please upload a leaf photo.",
            crop=crop
        )

    if not allowed_file(uploaded_file.filename):
        return render_template(
            "crop_disease.html",
            error="Upload JPG, JPEG, PNG or WEBP.",
            crop=crop
        )

    filename = secure_filename(uploaded_file.filename)

    if not filename:
        return render_template(
            "crop_disease.html",
            error="Invalid filename. Select another image."
        )

    filename = (
        f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{filename}"
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
        app.logger.exception("Could not save uploaded leaf image.")
        return render_template(
            "crop_disease.html",
            error="Could not save the image. Please try again.",
            crop=crop
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
            error="Upload JPG, JPEG, PNG or WEBP."
        )

    return render_template(
        "pest_doctor.html",
        error=(
            "Photo uploaded, but AI pest detection is not connected yet."
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
                raise ValueError

            if (
                area <= 0
                or investment < 0
                or yield_amount < 0
                or sale_price < 0
            ):
                raise ValueError

            revenue = yield_amount * sale_price
            profit = revenue - investment

            return render_template(
                "profit_calculator.html",
                crop=crop,
                area=area,
                yield_amount=yield_amount,
                revenue=revenue,
                total_cost=investment,
                profit=profit
            )

        except (ValueError, TypeError):
            return render_template(
                "profit_calculator.html",
                error="Please enter valid crop and numeric values."
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
    state = request.values.get(
        "state", "Andhra Pradesh"
    ).strip()

    category = request.values.get(
        "category", "all"
    ).strip().lower()

    all_schemes = [
        {
            "name": "PM-KISAN",
            "category": "financial",
            "description": (
                "Income support for eligible farmer families "
                "under the scheme rules."
            ),
            "eligibility": "Check the current official eligibility rules.",
            "benefit": "Financial assistance under scheme guidelines.",
            "url": "https://pmkisan.gov.in/"
        },
        {
            "name": "Pradhan Mantri Fasal Bima Yojana",
            "category": "insurance",
            "description": "Crop insurance for eligible notified crops.",
            "eligibility": "Check current season and enrolment rules.",
            "benefit": "Insurance protection against covered crop losses.",
            "url": "https://pmfby.gov.in/"
        },
        {
            "name": "Soil Health Card",
            "category": "soil",
            "description": "Soil nutrient testing and management guidance.",
            "eligibility": "Contact your local agriculture department.",
            "benefit": "Soil information for fertilizer decisions.",
            "url": "https://soilhealth.dac.gov.in/"
        }
    ]

    if category == "all":
        schemes = all_schemes
    else:
        schemes = [
            item for item in all_schemes
            if item["category"] == category
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
                "status": "success",
        "message": "HarvestAI is running successfully"
    })
   
