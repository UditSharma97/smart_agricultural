import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
import pickle

# ---------------------------
# Load Dataset
# ---------------------------
df = pd.read_csv("rainfall_by_districts_2019.csv")

st.title("🌧️ Rainfall Trend & Drought Risk Prediction")
st.write("Analyze rainfall data and predict drought risk levels for districts.")

# ---------------------------
# Data Preprocessing
# ---------------------------
df.columns = df.columns.str.strip()

df['Rainfall Deviation'] = df['Total Actual Rainfall (June\'17 to May\'18) in mm'] - df['Total Normal Rainfall (June\'17 to May\'18) in mm']

# Create drought risk labels
def drought_level(row):
    if row['Rainfall Deviation'] < -100:
        return "High Risk"
    elif -100 <= row['Rainfall Deviation'] <= 100:
        return "Medium Risk"
    else:
        return "Low Risk"

df['Drought Risk'] = df.apply(drought_level, axis=1)

# ---------------------------
# Train simple model
# ---------------------------
X = df[['Total Normal Rainfall (June\'17 to May\'18) in mm']]
y = df['Drought Risk']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# ---------------------------
# Streamlit Interface
# ---------------------------
st.sidebar.header("Select District for Analysis")
district = st.sidebar.selectbox("District", df['District'].unique())

district_data = df[df['District'] == district].iloc[0]

st.subheader(f"📍 District: {district}")
st.write(f"**Actual Rainfall:** {district_data['Total Actual Rainfall (June\'17 to May\'18) in mm']} mm")
st.write(f"**Normal Rainfall:** {district_data['Total Normal Rainfall (June\'17 to May\'18) in mm']} mm")

# Predict drought risk
predicted_risk = model.predict([[district_data['Total Normal Rainfall (June\'17 to May\'18) in mm']]])[0]
st.success(f"🌾 Predicted Drought Risk: **{predicted_risk}**")

# ---------------------------
# Visualization
# ---------------------------
st.subheader("📊 Rainfall Deviation by District")
fig = px.bar(df, x='District', y='Rainfall Deviation', color='Drought Risk',
             title="District-wise Rainfall Deviation",
             color_discrete_map={'High Risk':'red', 'Medium Risk':'orange', 'Low Risk':'green'})
st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Summary Insights
# ---------------------------
st.subheader("🔍 Insights Summary")
avg_rainfall = df['Total Actual Rainfall (June\'17 to May\'18) in mm'].mean()
st.write(f"🌧️ **Average Actual Rainfall:** {avg_rainfall:.2f} mm")
st.write(f"🚨 **High Drought Risk Districts:** {', '.join(df[df['Drought Risk']=='High Risk']['District'].values)}")
st.write(f"✅ **Low Drought Risk Districts:** {', '.join(df[df['Drought Risk']=='Low Risk']['District'].values)}")
