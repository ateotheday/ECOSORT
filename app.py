from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import mysql.connector
from mysql.connector import errors
from werkzeug.security import generate_password_hash, check_password_hash
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import requests
import json
import threading 


#initial db definitions
def get_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="ecosort"
    )

app = Flask(__name__)
app.secret_key = "ecosort_secret_key"


# for each upload we save it to a folder and then pass the path to the model for prediction
UPLOAD_FOLDER = "static/uploads" 
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# We have 2 classes: Organic/Non-Recyclable and Recyclable.
DISPLAY_CLASS_NAMES = ["Organic/Non-Recyclable", "Recyclable"]
DB_CLASS_NAMES = ["Organic", "Recyclable"]


# The model is "waste_model_final.keras"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "waste_model_final.keras")


# We load the model once at startup to avoid reloading it for every prediction, which would be very slow.
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model Loaded Successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None



#IP of the ESP32. Changes everytime 
ESP32_IP = "http://172.31.241.204"

#  NEW FUNCTION to make movements faster by sending the command in a separate thread without waiting for the response. 
def send_to_esp(class_index):
    try:
        requests.get(f"{ESP32_IP}/control?class={class_index}", timeout=5)
    except Exception as e:
        print(f"ESP32 Error: {e}")

# ---------------------- Database & User Functions ----------------------
def create_user(username, password):
    db = get_db()
    cur = db.cursor()

    pw_hash = generate_password_hash(password)
    display_name = username.split("@")[0] if "@" in username else username

    cur.execute(
        "INSERT INTO users (name, email, password_hash, points) VALUES (%s, %s, %s, %s)",
        (display_name, username, pw_hash, 0)
    )

    db.commit()
    cur.close()
    db.close()

def get_user(username, password):
    db = get_db()
    cur = db.cursor()

    cur.execute(
        "SELECT id, name, email, password_hash, role, points FROM users WHERE email=%s",
        (username,)
    )
    row = cur.fetchone()

    cur.close()
    db.close()

    if not row:
        return None

    user_id, name, email, pw_hash, role, points = row

    if not check_password_hash(pw_hash, password):
        return None

    display_name = name if name else (email.split("@")[0] if "@" in email else email)

    return (user_id, display_name, email, role, points)

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

def add_points(user_id, points):
    db = get_db()
    cur = db.cursor()

    cur.execute(
        "UPDATE users SET points = points + %s WHERE id = %s",
        (points, user_id)
    )

    db.commit()
    cur.close()
    db.close()

def get_user_points(user_id):
    db = get_db()
    cur = db.cursor()

    cur.execute(
        "SELECT points FROM users WHERE id = %s",
        (user_id,)
    )

    row = cur.fetchone()
    cur.close()
    db.close()
    return row[0] if row else 0

def get_leaderboard(limit=10):
    db = get_db()
    cur = db.cursor()

    cur.execute(
        """
        SELECT
            COALESCE(NULLIF(u.name, ''), SUBSTRING_INDEX(u.email, '@', 1)) AS display_name,
            u.points,
            COUNT(CASE WHEN p.predicted_label = 'Recyclable' THEN 1 END) AS recyclable_count
        FROM users u
        LEFT JOIN predictions p ON u.id = p.user_id
        GROUP BY u.id, u.name, u.email, u.points
        ORDER BY u.points DESC, u.id ASC
        LIMIT %s
        """,
        (limit,)
    )

    rows = cur.fetchall()
    cur.close()
    db.close()
    return rows

def get_user_rank(user_id):
    db = get_db()
    cur = db.cursor()

    cur.execute(
        """
        SELECT rank_num
        FROM (
            SELECT id, DENSE_RANK() OVER (ORDER BY points DESC, id ASC) AS rank_num
            FROM users
        ) ranked
        WHERE id = %s
        """,
        (user_id,)
    )

    row = cur.fetchone()
    cur.close()
    db.close()
    return row[0] if row else None

# ---------------------- Image Prediction ----------------------
def prepare_image(img):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ---------------------- Routes ----------------------
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
    session["email"] = user[2]
    session["role"] = user[3]
    session["points"] = user[4]

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

    user_id = session["user_id"]
    history_rows = get_history(user_id, limit=20)
    current_points = get_user_points(user_id)
    current_rank = get_user_rank(user_id)

    display_rows = []
    for image_path, predicted_label, confidence, created_at in history_rows:
        label_display = "Organic/Non-Recyclable" if predicted_label == "Organic" else "Recyclable"
        conf_percent = round(float(confidence) * 100, 2)
        display_rows.append((image_path, label_display, conf_percent, created_at))

    return render_template(
        "dashboard.html",
        username=session.get("username"),
        history=display_rows,
        points=current_points,
        rank=current_rank
    )

@app.route("/leaderboard")
def leaderboard():
    leaderboard_rows = get_leaderboard(10)
    return render_template("leaderboard.html", leaderboard=leaderboard_rows)

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

    if "user_id" in session:
        add_history(session["user_id"], filename, prediction_db, confidence_prob)
        add_points(session["user_id"], 10)
        session["points"] = get_user_points(session["user_id"])

    # FAST NON-BLOCKING CALL
    threading.Thread(target=send_to_esp, args=(class_index,)).start()

    return render_template(
        "predict.html",
        prediction=prediction_display,
        confidence=confidence_percent,
        filename=filename
    )

@app.route("/test_hardware/<int:class_index>")
def test_hardware(class_index):
    try:
        requests.get(f"{ESP32_IP}/control?class={class_index}", timeout=3)
        return jsonify({"status": "success", "sent_class": class_index})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ---------------------- Ollama Chat ----------------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"reply": "Please type something!"})

    try:
        payload = {
            "model": "llama3:latest",
            "prompt": user_message,
            "temperature": 0.7,
            "max_tokens": 200
        }
        response = requests.post(
            "http://127.0.0.1:11434/completions",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=10
        )
        response_json = response.json()
        bot_reply = response_json.get("completion", "Sorry, I couldn't understand that.")
    except Exception as e:
        bot_reply = f"Error contacting Ollama: {e}"

    return jsonify({"reply": bot_reply})

# ---------------------- Run ----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)