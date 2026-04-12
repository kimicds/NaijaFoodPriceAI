import re
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
model = joblib.load("food_price_new_model.pkl")

# ── Categorical options ─────────────────────────────────────────────────────
VENDOR_LIST = ["Government", "Supermarket"]
STORE_LIST = [
    "NBS", "Chowdeck", "Mano",
    "Glovo-Shoprite", "Glovo-Spar", "Glovo-Justrite",
    "Glovo-Glovo-Bargains", "Glovo-Food-Nation", "Glovo-Madina",
]
FOOD_CATEGORY_LIST = ["Condiment", "Protein", "Bread", "Staple", "Vegetable"]
FOOD_ITEM_LIST = [
    "beef", "beans", "bread", "chicken", "eggs", "fish",
    "garri", "milk", "oil", "pepper", "potato", "rice",
    "salt", "spaghetti", "sugar", "tomato", "yam",
]

# category string → (mass_g, volume_ml, count)  — shown as friendly label
CATEGORY_OPTIONS = [
    # label shown to user          mass_g   volume_ml  count
    ("500 g",                       500,      0,        0),
    ("1 kg",                       1000,      0,        0),
    ("2 kg",                       2000,      0,        0),
    ("5 kg",                       5000,      0,        0),
    ("10 kg",                     10000,      0,        0),
    ("25 kg",                     25000,      0,        0),
    ("50 kg",                     50000,      0,        0),
    ("250 ml",                        0,    250,        0),
    ("500 ml",                        0,    500,        0),
    ("750 ml",                        0,    750,        0),
    ("1 litre",                       0,   1000,        0),
    ("2 litres",                      0,   2000,        0),
    ("3 litres",                      0,   3000,        0),
    ("5 litres",                      0,   5000,        0),
    ("1 unit / piece",                0,      0,        1),
    ("6 eggs",                        0,      0,        6),
    ("12 eggs",                       0,      0,       12),
    ("30 eggs",                       0,      0,       30),
    ("1 loaf",                        0,      0,        1),
    ("1 plate",                       0,      0,        1),
]

# Ordinal label-encode for categorical columns
vendor_map       = {v: i for i, v in enumerate(VENDOR_LIST)}
store_map        = {v: i for i, v in enumerate(STORE_LIST)}
food_category_map = {v: i for i, v in enumerate(FOOD_CATEGORY_LIST)}
food_item_map    = {v: i for i, v in enumerate(FOOD_ITEM_LIST)}


# ── Feature builder ──────────────────────────────────────────────────────────
def build_features(form):
    """Return a 2-D list of features matching model column order:
    food_item, vendor_type, store_visited, latitude, longitude,
    food_category, mass_g, volume_ml, count,
    year, month, day, quarter
    """
    date     = pd.to_datetime(form["date"])
    year     = date.year
    month    = date.month
    day      = date.day
    quarter  = date.quarter

    vendor    = vendor_map.get(form["vendor_type"], 0)
    store     = store_map.get(form["store_visited"], 0)
    food_cat  = food_category_map.get(form["food_category"], 0)
    food_item = food_item_map.get(form["food_item"], 0)

    lat = float(form.get("latitude", 6.5244))
    lon = float(form.get("longitude", 3.3792))

    cat_idx   = int(form.get("category_idx", 0))
    _, mass_g, volume_ml, count = CATEGORY_OPTIONS[cat_idx]

    features = [[
        food_item, vendor, store, lat, lon,
        food_cat, mass_g, volume_ml, count,
        year, month, day, quarter,
    ]]
    return features


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict")
def predict():
    return render_template(
        "prediction.html",
        vendor_list=VENDOR_LIST,
        store_list=STORE_LIST,
        food_category_list=FOOD_CATEGORY_LIST,
        food_item_list=FOOD_ITEM_LIST,
        category_options=[(i, lbl) for i, (lbl, *_) in enumerate(CATEGORY_OPTIONS)],
    )


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/result", methods=["POST"])
def result():
    try:
        features = build_features(request.form)
        raw_pred = model.predict(features)[0]
        price    =  raw_pred #round(float(np.expm1(raw_pred)), 2)

        cat_idx = int(request.form.get("category_idx", 0))
        cat_label = CATEGORY_OPTIONS[cat_idx][0]

        return render_template(
            "result.html",
            price=price,
            vendor=request.form["vendor_type"],
            store=request.form["store_visited"],
            food_cat=request.form["food_category"],
            food_item=request.form["food_item"],
            category=cat_label,
            date=request.form["date"],
        )
    except Exception as e:
        return render_template("error.html", error=str(e)), 500


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON endpoint for programmatic access."""
    try:
        features = build_features(request.json)
        raw_pred = model.predict(features)[0]
        price    = round(float(np.expm1(raw_pred)), 2)
        return jsonify({"predicted_price_ngn": price, "status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e), "status": "fail"}), 400


if __name__ == "__main__":
    app.run(debug=False)
