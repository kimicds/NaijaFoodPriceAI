from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load your trained model
model = joblib.load("food_price_model.pkl")

# --- Auto-generated mappings from your lists ---
vendor_list = ["Government", "Supermarket"]
store_list = [
    "NBS","Chowdeck","Mano",
    "Glovo-Shoprite","Glovo-Spar","Glovo-Justrite",
    "Glovo-Glovo-Bargains","Glovo-Food-Nation","Glovo-Madina"
]
food_category_list = ["Condiment", "Protein", "Bread", "Staple", "Vegetable"]
food_item_list = [
    "oil","beef","fish","chicken","milk","eggs","bread",
    "potato","rice","beans","yam","garri","tomato","sugar",
    "pepper","spaghetti","salt"
    # Add all other items from your dataset
]
item_type_list = [
    "vegetable","bone in","fish","feet","wings","frozen","evaporated tin",
    "catfish smoked","agric","sliced","unsliced","groundnut","mudfish","palm",
    "boneless","irish","sweet","ofada","local","imported","medium grained",
    "white black eye","tuber","white","yellow","tomato","brown","dangote",
    "avocado","olive","sesame","red","kings","sunflower","soya","power",
    "canola","linseed","almond","turbinado","golden penny","black","gluten free",
    "slim","spaghettini","yam","granulated","icing","cube","12","eno fruit",
    "mr chef","refined&iodized","sea","himalayan pink","wheat","30","6","24",
    "15","corn beef","minced","refined","dano milk","dano milk (full cream powder)",
    "whole","breast","stir fry","smoked","titus","croaker","nuggets","pelati",
    "plum","paste","hot","red chili","green chili","suya","fresh","ijebu",
    "basmati","whole grain","parboiled","long grain","rodo","full","ground black",
    "ewa oloyin","baked","green","multigrain","ground regular","shredded",
    "(saki) honeycomb","cubes boneless","red bell","coconut","thigh","laps",
    "crayfish (ground)","dano milk (powder)","cherry","loaf","black eye","locust",
    "ground","big bull","red kidney","cocoyam","mixed","red habanero","shombo",
    "tatashe red","thin","gizzard","pops","wholemeal soft","sweet red",
    "catfish smofked","ground lean","cane","luncheon meat","cuts","smoked fish",
    "sachet","ground dry chilli","dehydrated vegetable","semolina",
    "steak smoked dried catfish","roast agrd","shin soft","lemon","green bell",
    "yellow bell","rock"
]
category_list = [
    "1000.0ml","1000.0g","1.0unit","1.0eggs","1.0loaf","1.0piece","3000.0ml",
    "3500.0ml","50.0ml","250.0g","500.0ml","2000.0ml","750.0ml","900.0ml","5000.0ml",
    "25000.0ml","69.0g","1600.0ml","1500.0ml","946.0ml","710.0ml","500.0g","3",
    "185.0ml","4.0litres","2.0litre","120.0g","1","700.0ml","150.0ml","4730.0ml",
    "4500.0ml","2750.0ml","4000.0g","250.0ml","200.0ml","680.0g","1400.0ml",
    "10000.0ml","50.0g","283.5g","4535.0g","4500.0g","2267.0g","250000.0g",
    "485.0g","14","100.0g","600.0ml","275.0g","12.0eggs","88.7ml","750.0g",
    "125.0g","1130.0g","200.0g","800.0g","195.0g","75.0og","369.0g","30.0eggs",
    "6.0eggs","24eggs","12eggs","30eggs","15.0eggs","340.0g","390.0g","236.0ml",
    "3780.0ml","24.0eggs","1800.0g","354.0g","170.0g","1.0plate","700.0g",
    "450.0g","2200.0g","50000.0g","400.0g","210.0g","93.0g","90.0g","650.0g",
    "552.0g","1250.0g","470.0g","397.0g","300.0ml","300.0g","150.0g","1500.0g",
    "70.0g","10000.0g","110.0g","567.0g","3000.0g","5000.0g","2000.0g","9070.0g",
    "454.0g","11340.0g","900.0g","45.0ml","600.0g","4000.0g","2500.0g","57.0g",
    "410.0g","350.0g","907.0g","0.3","42.0g","270.0g","6.0un","240.0g","4.0un",
    "1.0un","475.0g","80.0g","5.0un","1300.0g","550.0g","908.0g","60.0g","65.0g",
    "1100.0g","238.0g","4600.0ml","2600.0ml","2500.0ml","2700.0ml","1900.0ml",
    "2.75","51000.0ml","2","90.0cubes","950.0g","474.0g","850.0g","178.0g",
    "65.7g","420.0g","415.0g","4.0s","12","1025.0g","1.0s","375.0g","507.0g",
    "25000.0g","2250.0g","20000.0g","45.0cubes","1600.0g","220.0g","440.0g",
    "6eggs","100.0ml","320.0g","55.0g","1.5g","2.267","250.0gr","10.0pcs",
    "1.0gg","20.0pcs","15eggs","1900.0g"
]

# Generate safe mapping dictionaries
vendor_map = {v:i for i,v in enumerate(vendor_list)}
store_map = {v:i for i,v in enumerate(store_list)}
food_category_map = {v:i for i,v in enumerate(food_category_list)}
food_item_map = {v:i for i,v in enumerate(food_item_list)}
item_type_freq = {v:1.0 for v in item_type_list}   # frequency encoding placeholder
category_freq = {v:1.0 for v in category_list}    # frequency encoding placeholder

# --- Routes ---
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict")
def predict():
    return render_template("prediction.html",
                           vendor_list=vendor_list,
                           store_list=store_list,
                           food_category_list=food_category_list,
                           food_item_list=food_item_list,
                           item_type_list=item_type_list,
                           category_list=category_list)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/result", methods=["POST"])
def result():
    try:
        # Date and engineered features
        date = pd.to_datetime(request.form["date"])
        year = date.year
        month = date.month
        day = date.day
        dayofweek = date.dayofweek
        weekofyear = int(date.isocalendar().week)
        quarter = date.quarter
        weekend = 1 if dayofweek in [5,6] else 0

        # Safe categorical mappings with default 0
        vendor = vendor_map.get(request.form["vendor_type"], 0)
        store = store_map.get(request.form["store_visited"], 0)
        food_cat = food_category_map.get(request.form["food_category"], 0)
        food_item = food_item_map.get(request.form["food_item"], 0)
        category_encoded = category_freq.get(request.form["category"], 0)
        item_encoded = item_type_freq.get(request.form["item_type"], 0)

        # Location
        lat = float(request.form.get("latitude", 0))
        lon = float(request.form.get("longitude", 0))

        # Features for model
        features = [[
            vendor, store, food_cat, food_item,
            category_encoded, item_encoded,
            year, month, day, dayofweek,
            weekofyear, quarter, weekend,
            lat, lon
        ]]

        # Prediction and reverse log transform
        prediction = model.predict(features)[0]
        price = np.expm1(prediction)

        return render_template("result.html", price=round(price,2),
                               vendor=request.form["vendor_type"],
                               store=request.form["store_visited"],
                               food_cat=request.form["food_category"],
                               food_item=request.form["food_item"],
                               item_type=request.form["item_type"],
                               category=request.form["category"],
                               date=request.form["date"]
                               )

    except Exception as e:
        return f"<h3>Prediction failed: {e}</h3>"

if __name__=="__main__":
    app.run(debug=True)