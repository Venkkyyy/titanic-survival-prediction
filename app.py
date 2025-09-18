import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load your model (if saved) or train quickly inside the script
@st.cache_resource
def load_model():
    data = pd.read_csv("train.csv")

    # Simple preprocessing
    data["Age"].fillna(data["Age"].mean(), inplace=True)
    data["Sex"] = data["Sex"].map({"male": 0, "female": 1})
    X = data[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]
    y = data["Survived"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = load_model()

st.title("🚢 Titanic Survival Prediction")

# Collect user input
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Number of Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare Paid", 0, 500, 50)

# Convert sex to number
sex = 0 if sex == "male" else 1

# Make prediction
if st.button("Predict Survival"):
    input_data = pd.DataFrame([[pclass, sex, age, sibsp, parch, fare]],
                              columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"])
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success(" This passenger would have **Survived!**")
    else:
        st.error(" This passenger would have **Not Survived!**")
