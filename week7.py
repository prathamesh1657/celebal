import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Flower Classifier", layout="centered")

# App title and description
st.title("🌼 Iris Flower Classifier App")
st.markdown("This tool helps identify the Iris flower species using its physical characteristics via a trained Random Forest model.")

# Load and train model
@st.cache_data
def train_model():
    dataset = load_iris()
    features = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    labels = dataset.target
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(features, labels)
    return clf, dataset

clf_model, iris_data = train_model()
cols = iris_data.feature_names
species_names = iris_data.target_names

# Sidebar inputs
st.sidebar.header("Provide Flower Attributes")
sl = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
sw = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
pl = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.2)
pw = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.3)

# User input for prediction
user_input = pd.DataFrame([[sl, sw, pl, pw]], columns=cols)

# Prediction button
if st.button("Predict Iris Species"):
    result = clf_model.predict(user_input)[0]
    confidence = clf_model.predict_proba(user_input)[0]
    species_result = species_names[result]

    st.success(f"🌸 Identified as: **{species_result}**")
    st.markdown("### Prediction Confidence")
    st.bar_chart(pd.DataFrame(confidence, index=species_names, columns=["Confidence"]))

# Plot section
st.subheader("🌿 Sepal Distribution by Species")
df_viz = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
df_viz["Species"] = pd.Categorical.from_codes(iris_data.target, iris_data.target_names)

fig, ax = plt.subplots()
sns.scatterplot(data=df_viz, x="sepal_length (cm)", y="sepal_width (cm)", hue="Species", ax=ax)
st.pyplot(fig)
