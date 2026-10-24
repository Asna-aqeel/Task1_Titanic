"""Titanic Survival Prediction - Flask app for PythonAnywhere."""
import os
import joblib
import pandas as pd
from flask import Flask, request, render_template_string

BASE = os.path.dirname(os.path.abspath(__file__))
art = joblib.load(os.path.join(BASE, "titanic_model.pkl"))
model, enc, FEATURES = art["model"], art["encoders"], art["features"]

app = Flask(__name__)

PAGE = """
<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Titanic Survival Predictor</title>
<style>
 body{font-family:Arial,sans-serif;background:#0b2545;margin:0;padding:20px;color:#222}
 .card{max-width:480px;margin:auto;background:#fff;border-radius:12px;padding:24px}
 h2{margin-top:0;color:#0b2545} label{display:block;margin-top:12px;font-weight:bold}
 input,select{width:100%;padding:8px;margin-top:4px;box-sizing:border-box}
 button{margin-top:18px;width:100%;padding:12px;background:#13315c;color:#fff;border:0;border-radius:6px;font-size:16px;cursor:pointer}
 .res{margin-top:18px;padding:14px;border-radius:8px;font-size:18px;text-align:center}
 .ok{background:#d4edda;color:#155724}.no{background:#f8d7da;color:#721c24}
</style></head><body><div class="card">
<h2>🚢 Titanic Survival Predictor</h2>
<form method="post">
 <label>Passenger class</label>
 <select name="pclass">{% for v in [1,2,3] %}<option value="{{v}}" {{'selected' if f.get('pclass')==v|string}}>{{v}}</option>{% endfor %}</select>
 <label>Sex</label>
 <select name="sex">{% for v in sexes %}<option {{'selected' if f.get('sex')==v}}>{{v}}</option>{% endfor %}</select>
 <label>Age</label><input type="number" step="0.5" min="0" max="100" name="age" value="{{f.get('age','30')}}" required>
 <label>Siblings / spouses aboard</label><input type="number" min="0" max="10" name="sibsp" value="{{f.get('sibsp','0')}}" required>
 <label>Parents / children aboard</label><input type="number" min="0" max="10" name="parch" value="{{f.get('parch','0')}}" required>
 <label>Fare (£)</label><input type="number" step="0.01" min="0" name="fare" value="{{f.get('fare','32')}}" required>
 <label>Port of embarkation (C=Cherbourg, Q=Queenstown, S=Southampton)</label>
 <select name="embarked">{% for v in ports %}<option {{'selected' if f.get('embarked')==v}}>{{v}}</option>{% endfor %}</select>
 <button type="submit">Predict</button>
</form>
{% if result %}<div class="res {{cls}}">{{result}}<br><small>Survival probability: {{prob}}%</small></div>{% endif %}
</div></body></html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    ctx = dict(f=request.form, sexes=list(enc["Sex"].classes_), ports=list(enc["Embarked"].classes_),
               result=None, cls="", prob=None)
    if request.method == "POST":
        row = pd.DataFrame([{
            "Pclass": int(request.form["pclass"]),
            "Sex": enc["Sex"].transform([request.form["sex"]])[0],
            "Age": float(request.form["age"]),
            "SibSp": int(request.form["sibsp"]),
            "Parch": int(request.form["parch"]),
            "Fare": float(request.form["fare"]),
            "Embarked": enc["Embarked"].transform([request.form["embarked"]])[0],
        }])[FEATURES]
        p = float(model.predict_proba(row)[0, 1])
        survived = p >= 0.5
        ctx.update(result="✅ Likely SURVIVED" if survived else "❌ Likely DID NOT SURVIVE",
                   cls="ok" if survived else "no", prob=round(p * 100, 1))
    return render_template_string(PAGE, **ctx)

if __name__ == "__main__":
    app.run(debug=True)
