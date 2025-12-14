from flask import Flask, render_template, request
import pandas as pd
from model_utils import (
    predict_scenario1,
    predict_scenario2,
    predict_scenario3
)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/scenario/<int:scenario>")
def scenario_page(scenario):
    return render_template("scenario.html", scenario=scenario)

# =========================
# MANUAL INPUT ONLY
# =========================
@app.route("/manual/<int:scenario>", methods=["GET", "POST"])
def manual_input(scenario):
    if request.method == "GET":
        return render_template("manual.html", scenario=scenario)

    data = request.form.to_dict()
    df = pd.DataFrame([data]).astype(float)

    if scenario == 1:
        result = predict_scenario1(df)[0]
    elif scenario == 2:
        result = round(predict_scenario2(df)[0], 4)
    elif scenario == 3:
        result = predict_scenario3(df)[0]
    else:
        result = "Invalid scenario"

    return render_template("result.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
