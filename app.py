import os, joblib, pandas as pd, streamlit as st
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")
art = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "titanic_model.pkl"))
model, enc, FEATURES = art["model"], art["encoders"], art["features"]

st.title("🚢 Titanic Survival Predictor")
st.caption(f"Model: {art['model_name']}")
c1, c2 = st.columns(2)
pclass = c1.selectbox("Passenger class", [1, 2, 3], index=2)
sex = c2.selectbox("Sex", list(enc["Sex"].classes_))
age = c1.number_input("Age", 0.0, 100.0, 30.0, 0.5)
fare = c2.number_input("Fare (£)", 0.0, 600.0, 32.0)
sibsp = c1.number_input("Siblings / spouses aboard", 0, 10, 0)
parch = c2.number_input("Parents / children aboard", 0, 10, 0)
emb = st.selectbox("Port of embarkation", list(enc["Embarked"].classes_), index=2)

if st.button("Predict", type="primary"):
    row = pd.DataFrame([{"Pclass": pclass, "Sex": enc["Sex"].transform([sex])[0], "Age": age,
                         "SibSp": sibsp, "Parch": parch, "Fare": fare,
                         "Embarked": enc["Embarked"].transform([emb])[0]}])[FEATURES]
    p = model.predict_proba(row)[0, 1]
    (st.success if p >= 0.5 else st.error)(
        f"{'✅ Likely SURVIVED' if p >= 0.5 else '❌ Likely DID NOT SURVIVE'} — probability {p:.1%}")
