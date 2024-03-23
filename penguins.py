"""Custom penguins classifier interface."""
import pickle

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import streamlit as st

st.title("Penguin Classifier")
st.write(
    "This app uses 6 inputs to predict the species of "
    "penguin using a model built on the Palmer Penguins "
    "dataset. Use the form below to get started!"
)

password_guess = st.text_input("What is the Password?")
if password_guess != st.secrets["password"]:
    st.stop()

penguin_file = st.file_uploader("Upload your own penguin data")

# Load default model if user file not found
if penguin_file is None:
    with open("random_forest_penguin.pickle", "rb") as model_file:
        rfc = pickle.load(model_file)

    with open("output_penguin.pickle", "rb") as label_file:
        unique_penguin_mapping = pickle.load(label_file)

    penguin_df = pd.read_csv("penguins.csv")

else:
    penguin_df = pd.read_csv(penguin_file)
    penguin_df.dropna(inplace=True)

    output = penguin_df["species"]
    features = penguin_df[
        [
            "island",
            "bill_length_mm",
            "bill_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
            "sex",
        ]
    ]

    features = pd.get_dummies(features)
    output, unique_penguin_mapping = pd.factorize(output)


    X_train, X_test, y_train, y_test = train_test_split(
        features, output, test_size=0.8
    )
    rfc = RandomForestClassifier(random_state=15)
    rfc.fit(X_train.values, y_train)

    y_pred = rfc.predict(X_test.values)
    score = accuracy_score(y_pred, y_test) * 100
    st.write(
        "We trained a Random Forest model on these data, "
        f"it has a score of {round(score, 1)}%! "
        "Use the inputs below to try out the model"
    )

with st.form("user_inputs"):
    island_options = ["Biscoe", "Dream", "Torgerson"]
    island = st.selectbox(
        "Penguin Island", options=island_options
    )
    sex_options = ["Female", "Male"]
    sex = st.selectbox(
        "Sex", options=sex_options
    )

    bill_length = st.number_input(
        "Bill Length (mm)", min_value=0
    )
    bill_depth = st.number_input(
        "Bill Depth (mm)", min_value=0
    )
    flipper_length = st.number_input(
        "Flipper Length (mm)", min_value=0
    )
    body_mass = st.number_input(
        "Body Mass (g)", min_value=0
    )

    # Vectorization
    island_index = {
        option: 0
        for option in island_options
    }
    island_index[island] = 1

    sex_index = {
        option: 0
        for option in sex_options
    }
    sex_index[sex] = 1

    st.form_submit_button()

new_prediction = rfc.predict(
    [
        [
            bill_length, bill_depth, flipper_length, body_mass
        ] + list(
            island_index.values()
        ) + list(
            sex_index.values()
        )
    ]
)
prediction_species = unique_penguin_mapping[new_prediction][0]

st.subheader("Predicting Your Penguin's Species:")
st.write(f"We predict your penguin is of {prediction_species} species")

st.write(
    "We used Random Forest to predict the species, the features "
    "userd in this prediction are ranked by relative importance below."
)
st.image("feature_importance.png")

st.write(
    "Below are the histograms for each continuous variable "
    "separated by penguin species. The vertical line represents "
    "your input value."
)

fig, ax = plt.subplots()
ax = sns.displot(
    x=penguin_df["bill_length_mm"],
    hue=penguin_df["species"],
)
plt.axvline(bill_length)
st.pyplot(ax)

fig, ax = plt.subplots()
ax = sns.displot(
    x=penguin_df["bill_depth_mm"],
    hue=penguin_df["species"],
)
plt.axvline(bill_depth)
plt.title("Bill Depth by Species")
st.pyplot(ax)

fig, ax = plt.subplots()
ax = sns.displot(
    x=penguin_df["flipper_length_mm"],
    hue=penguin_df["species"],
)
plt.axvline(flipper_length)
plt.title("Flipper Length by Species")
st.pyplot(ax)
