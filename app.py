import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

st.title("Salary Prediction System")

st.write("Enter training data to build the model.")

# User inputs training data
experience = st.number_input("Years of Experience", 0, 20)
education = st.number_input("Education Level (1=Bachelor, 2=Master, 3=PhD)", 1, 3)
skill_score = st.number_input("Skill Score (0-100)", 0, 100)
company_size = st.number_input("Company Size", 10, 1000)
salary = st.number_input("Actual Salary")

# Store data
if "data" not in st.session_state:
    st.session_state.data = []

# Add data button
if st.button("Add Training Data"):
    st.session_state.data.append(
        [experience, education, skill_score, company_size, salary]
    )
    st.success("Data added!")

# Convert to dataframe
if len(st.session_state.data) > 0:
    df = pd.DataFrame(
        st.session_state.data,
        columns=["experience", "education", "skill_score", "company_size", "salary"],
    )

    st.subheader("Training Data")
    st.dataframe(df)

    X = df[["experience", "education", "skill_score", "company_size"]]
    y = df["salary"]

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    # Evaluate model
    predictions = model.predict(X)

    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)

    st.subheader("Model Evaluation")
    st.write("MAE:", mae)
    st.write("R² Score:", r2)

    st.subheader("Predict Salary")

    exp = st.number_input("Experience for prediction", 0, 20, key="p1")
    edu = st.number_input("Education Level", 1, 3, key="p2")
    skill = st.number_input("Skill Score", 0, 100, key="p3")
    comp = st.number_input("Company Size", 10, 1000, key="p4")

    if st.button("Predict Salary"):
        input_data = np.array([[exp, edu, skill, comp]])
        pred_salary = model.predict(input_data)

        st.success(f"Predicted Salary: {pred_salary[0]:,.2f}")