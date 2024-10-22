#%%
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

#%%
# Set page configuration as the first command
st.set_page_config(page_title="Salary Predictor & Explorer", page_icon="💰", layout="wide")

# Load data and preprocess
def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map

def clean_experience(x):
    if x == 'More than 50 years':
        return 50
    if x == 'Less than 1 year':
        return 0.5
    return float(x)

def clean_education(x):
    if 'Bachelor’s degree' in x:
        return 'Bachelor’s degree'
    if 'Master’s degree' in x:
        return 'Master’s degree'
    if 'Professional degree' in x or 'Other doctoral' in x:
        return 'Post grad'
    return 'Less than a Bachelors'

#%%
@st.cache_data
def load_data():
    df = pd.read_csv("survey_results_public.csv")
    df = df[["Country", "EdLevel", "YearsCodePro", "Employment", "ConvertedCompYearly"]]
    df = df[df["ConvertedCompYearly"].notnull()]
    df = df.dropna()
    df = df[df["Employment"] == "Employed, full-time"]
    df = df.drop("Employment", axis=1)

    country_map = shorten_categories(df.Country.value_counts(), 400)
    df["Country"] = df["Country"].map(country_map)
    df = df[df["ConvertedCompYearly"] <= 250000]
    df = df[df["ConvertedCompYearly"] >= 10000]
    df = df[df["Country"] != "Other"]

    df["YearsCodePro"] = df["YearsCodePro"].apply(clean_experience)
    df["EdLevel"] = df["EdLevel"].apply(clean_education)
    df = df.rename({"ConvertedCompYearly": "Salary"}, axis=1)
    return df

df = load_data()

#%%
# Prepare features and target variable
le_education = LabelEncoder()
df['EdLevel'] = le_education.fit_transform(df['EdLevel'])
le_country = LabelEncoder()
df['Country'] = le_country.fit_transform(df['Country'])

X = df.drop("Salary", axis=1)
y = df["Salary"]

#%%
# Train model
regressor = DecisionTreeRegressor(random_state=0)
parameters = {"max_depth": [None, 2, 4, 6, 8, 10, 12]}
gs = GridSearchCV(regressor, parameters, scoring='neg_mean_squared_error')
gs.fit(X, y.values)
regressor = gs.best_estimator_
regressor.fit(X, y.values)

#%%
# Streamlit app
st.title("💰 Salary Predictor & Explorer")
mode = st.sidebar.radio("Select Mode", ["Predict Salary", "Explore Data"])

if mode == "Predict Salary":
    st.subheader("Salary Prediction")
    st.markdown("""
        Fill in the details below to get an estimated salary based on your inputs.
    """)

    # User input for prediction
    country = st.selectbox("Select Country", le_country.inverse_transform(range(len(le_country.classes_))))
    years_experience = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, step=0.5)
    education_level = st.selectbox("Select Education Level", le_education.inverse_transform(range(len(le_education.classes_))))

    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'Country': [country],
        'EdLevel': [education_level],
        'YearsCodePro': [years_experience]
    })

    # Encode input data
    input_data['EdLevel'] = le_education.transform(input_data['EdLevel'])
    input_data['Country'] = le_country.transform(input_data['Country'])

    # Predict salary
    predicted_salary = regressor.predict(input_data)

    # Display result
    st.subheader("🔍 Estimated Salary")
    st.write(f"Based on your inputs, your estimated salary is: **${predicted_salary[0]:,.2f}**")

else:
    st.title("Explore Software Engineer Salaries")

    st.write(
        """
    ### Stack Overflow Developer Survey 2020
    """
    )

    data = df["Country"].value_counts()

    # Pie chart for country distribution
    fig1, ax1 = plt.subplots()
    ax1.pie(data, labels=data.index, autopct="%1.1f%%", shadow=True, startangle=90)
    ax1.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.write("""#### Distribution of Data from Different Countries""")
    st.pyplot(fig1)

    # Bar chart for mean salary by country
    st.write("#### Mean Salary Based On Country")
    data = df.groupby(["Country"])["Salary"].mean().sort_values(ascending=True)
    st.bar_chart(data)

    # Line chart for mean salary by years of experience
    st.write("#### Mean Salary Based On Experience")
    data = df.groupby(["YearsCodePro"])["Salary"].mean().sort_values(ascending=True)
    st.line_chart(data)

#%%
# Optional: Add a footer with additional information
st.markdown("""
    ---
    ### About this App
    This application uses data from a global survey of developers to predict salaries and explore salary trends. 
    Please enter your details for prediction or explore the data visualizations.
""")

# Optional: Add an image or logo
# st.image("path/to/logo.png", use_column_width=True)  # Uncomment and provide path if you have a logo

# %%
