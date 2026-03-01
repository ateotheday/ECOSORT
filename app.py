from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import mysql.connector
from mysql.connector import errors
from werkzeug.security import generate_password_hash, check_password_hash
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import requests

def get_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="ecosort"
    )

app = Flask(__name__)
app.secret_key = "ecosort_secret_key"

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

DISPLAY_CLASS_NAMES = ["Organic/Non-Recyclable", "Recyclable"]
DB_CLASS_NAMES = ["Organic", "Recyclable"]

MODEL_PATH = os.path.join(os.path.dirname(__file__), "waste_model_final.keras")

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

ESP32_IP = "http://192.168.1.50"

def create_user(username, password):
    db = get_db()
    cur = db.cursor()
    pw_hash = generate_password_hash(password)
    cur.execute("INSERT INTO users (email, password_hash) VALUES (%s, %s)", (username, pw_hash))
    db.commit()
    cur.close()
    db.close()

def get_user(username, password):
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT id, email, password_hash, role FROM users WHERE email=%s", (username,))
    row = cur.fetchone()
    cur.close()
    db.close()
    if not row:
        return None
    user_id, email, pw_hash, role = row
    if not check_password_hash(pw_hash, password):
        return None
    return (user_id, email, role)

def add_history(user_id, filename, predicted_label_db, confidence_prob):
    db = get_db()
    cur = db.cursor()
    cur.execute(
        "INSERT INTO predictions (user_id, image_path, predicted_label, confidence) VALUES (%s, %s, %s, %s)",
        (user_id, filename, predicted_label_db, float(confidence_prob))
    )
    db.commit()
    cur.close()
    db.close()

def get_history(user_id, limit=20):
    db = get_db()
    cur = db.cursor()
    cur.execute(
        """
        SELECT image_path, predicted_label, confidence, created_at
        FROM predictions
        WHERE user_id=%s
        ORDER BY created_at DESC
        LIMIT %s
        """,
        (user_id, limit)
    )
    rows = cur.fetchall()
    cur.close()
    db.close()
    return rows

def prepare_image(img):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")

    username = request.form.get("username")
    password = request.form.get("password")

    user = get_user(username, password)
    if not user:
        return render_template("login.html", error="Invalid username or password")

    session["user_id"] = user[0]
    session["username"] = user[1]
    session["role"] = user[2]
    session["hardware_mode"] = False

    return redirect(url_for("dashboard"))

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")

    username = request.form.get("username")
    password = request.form.get("password")

    try:
        create_user(username, password)
        return redirect(url_for("login"))
    except errors.IntegrityError:
        return render_template("register.html", error="Username already exists")
    except Exception:
        return render_template("register.html", error="Registration failed")

@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))

    history_rows = get_history(session["user_id"], limit=20)

    display_rows = []
    for image_path, predicted_label, confidence, created_at in history_rows:
        label_display = "Organic/Non-Recyclable" if predicted_label == "Organic" else "Recyclable"
        conf_percent = round(float(confidence) * 100, 2)
        display_rows.append((image_path, label_display, conf_percent, created_at))

    return render_template(
        "dashboard.html",
        username=session.get("username"),
        hardware_mode=session.get("hardware_mode", False),
        history=display_rows
    )

@app.route("/toggle_hardware", methods=["POST"])
def toggle_hardware():
    if "user_id" not in session:
        return jsonify({"status": "not_logged_in"}), 401

    state = request.json.get("state", "off")
    session["hardware_mode"] = (state == "on")
    return jsonify({"status": "success", "hardware_mode": session["hardware_mode"]})

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html")

    if model is None:
        return "Model failed to load.", 500

    file = request.files.get("file")
    if not file:
        return "No file uploaded.", 400

    filename = file.filename
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    img = Image.open(filepath)
    img_array = prepare_image(img)

    preds = model.predict(img_array)
    class_index = int(np.argmax(preds[0]))
    confidence_prob = float(np.max(preds[0]))

    prediction_display = DISPLAY_CLASS_NAMES[class_index]
    prediction_db = DB_CLASS_NAMES[class_index]
    confidence_percent = round(confidence_prob * 100, 2)

    used_hardware = ("user_id" in session) and session.get("hardware_mode", False)

    if "user_id" in session:
        add_history(session["user_id"], filename, prediction_db, confidence_prob)

    if used_hardware:
        try:
            requests.get(f"{ESP32_IP}/control?class={class_index}", timeout=2)
        except:
            pass

    return render_template(
        "predict.html",
        prediction=prediction_display,
        confidence=confidence_percent,
        filename=filename
    )

if __name__ == "__main__":
    app.run(debug=True)