# FoodPriceNG 🇳🇬

ML-powered food price prediction app for Nigerian markets.

## Project Structure

```
food_price_app/
├── app.py                  # Flask app (main entry point)
├── food_price_new_model.pkl  # ← your trained model (add this!)
├── requirements.txt
├── Procfile                # for Render / Railway / Heroku
└── templates/
    ├── base.html
    ├── home.html
    ├── prediction.html
    ├── result.html
    ├── about.html
    └── error.html
```

## Model Features (exact order)

The model expects these columns in this order:

| # | Feature | Type |
|---|---------|------|
| 1 | food_item | label-encoded int |
| 2 | vendor_type | label-encoded int |
| 3 | store_visited | label-encoded int |
| 4 | latitude | float |
| 5 | longitude | float |
| 6 | food_category | label-encoded int |
| 7 | mass_g | float (0 if not applicable) |
| 8 | volume_ml | float (0 if not applicable) |
| 9 | count | float (0 if not applicable) |
| 10 | year | int |
| 11 | month | int |
| 12 | day | int |
| 13 | quarter | int |

## Run Locally

```bash
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000

## Deploy to Render (free tier)

1. Push this folder to a GitHub repo.
2. Go to https://render.com → New Web Service.
3. Connect your repo, set:
   - **Build command**: `pip install -r requirements.txt`
   - **Start command**: `gunicorn app:app`
4. Upload `food_price_new_model.pkl` to your repo root.
5. Deploy!

## API Endpoint

POST `/api/predict` with JSON body matching the form fields:

```json
{
  "vendor_type": "Supermarket",
  "store_visited": "Glovo-Shoprite",
  "food_category": "Staple",
  "food_item": "rice",
  "category_idx": "4",
  "date": "2025-06-01",
  "latitude": "6.5244",
  "longitude": "3.3792"
}
```

Returns:
```json
{ "predicted_price_ngn": 8450.25, "status": "ok" }
```
