# FoodPriceNG 🇳🇬

> ML-powered food price prediction for Nigerian markets — built under the NitHub AI Ecosystem.

![Python](https://img.shields.io/badge/Python-3.11-green?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange?style=flat-square)
![NitHub](https://img.shields.io/badge/NitHub-AI%20Ecosystem-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)

---

## About

FoodPriceNG is an open-source machine-learning web application that estimates the retail price of everyday food items across Nigerian markets. It aggregates price signals from government surveys, supermarket chains, and on-demand delivery platforms — and serves predictions in real time through a clean, mobile-friendly interface.

**The problem:** Nigeria's food inflation surpassed 35% in 2024, yet ordinary households have no reliable tool to know what a fair price looks like before they shop. FoodPriceNG closes that gap — free, for anyone with a browser.

---

## NitHub AI Ecosystem

This project is proudly developed under the **[NitHub AI Ecosystem](https://nithub.unilag.edu.ng)** — a community driving open-source innovation and AI-assisted development across Africa. Contributions and collaborations from the NitHub community are warmly welcome.

---

## Project Structure

```
food_price_app/
├── app.py                  # Flask app 
├── food_price_new_model.pkl  # ← your trained model
├── requirements.txt
├── Procfile                # for Render 
└── templates/
    ├── base.html
    ├── home.html
    ├── prediction.html
    ├── result.html
    ├── about.html
    └── error.html
```

## License

MIT License — see `LICENSE` for details.

---

<p align="center">
  Built with ❤️ under the <strong>NitHub AI Ecosystem</strong> · Nigeria 🇳🇬
</p
