import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset directly from GitHub
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# Keep useful columns
df = df[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","Survived"]]

# Handle missing values
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna("S", inplace=True)

# Convert categorical to numeric
df["Sex"] = df["Sex"].map({"male":0, "female":1})
df["Embarked"] = df["Embarked"].map({"S":0, "C":1, "Q":2})

# Features & target
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model Accuracy
acc = model.score(X_test, y_test)
st.sidebar.write(f"Model Accuracy: {acc:.2f}")

# ---------------- Streamlit App ----------------
st.title("Titanic Survival Prediction")

st.sidebar.header("Enter Passenger Details")

pclass = st.sidebar.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1,2,3])
sex = st.sidebar.selectbox("Sex", ["male","female"])
age = st.sidebar.slider("Age", 1, 80, 25)
sibsp = st.sidebar.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.sidebar.number_input("Parents/Children Aboard", 0, 10, 0)
fare = st.sidebar.slider("Ticket Fare", 0.0, 500.0, 32.0)
embarked = st.sidebar.selectbox("Embarked (Port)", ["S","C","Q"])

# Convert inputs
sex = 0 if sex=="male" else 1
embarked = {"S":0,"C":1,"Q":2}[embarked]

# Prediction
import numpy as np
input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
prediction = model.predict(input_data)

if st.button("Predict Survival"):
    if prediction[0] == 1:
        st.success("This passenger would have SURVIVED!")
    else:
        st.error(" This passenger would NOT have survived.")

import warnings
warnings.filterwarnings("ignore")