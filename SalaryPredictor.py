#%%
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

#%%
st.set_page_config(page_title="Salary Predictor & Explorer", page_icon="💰", layout="wide")

# Data cleaning functions
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

# Encode for model training
le_education = LabelEncoder()
df['EdLevel'] = le_education.fit_transform(df['EdLevel'])
le_country = LabelEncoder()
df['Country'] = le_country.fit_transform(df['Country'])

X = df.drop("Salary", axis=1)
y = df["Salary"]

# Train model
regressor = DecisionTreeRegressor(random_state=0)
parameters = {"max_depth": [None, 2, 4, 6, 8, 10, 12]}
gs = GridSearchCV(regressor, parameters, scoring='neg_mean_squared_error')
gs.fit(X, y.values)
regressor = gs.best_estimator_
regressor.fit(X, y.values)

#%%
st.title("💰 Salary Predictor & Explorer")
mode = st.sidebar.radio("Select Mode", ["Predict Salary", "Explore Data"])

if mode == "Predict Salary":
    st.subheader("Salary Prediction")
    st.markdown("Fill in the details below to get an estimated salary based on your inputs.")

    country = st.selectbox("Select Country", le_country.inverse_transform(range(len(le_country.classes_))))
    years_experience = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, step=0.5)
    education_level = st.selectbox("Select Education Level", le_education.inverse_transform(range(len(le_education.classes_))))

    input_data = pd.DataFrame({
        'Country': [country],
        'EdLevel': [education_level],
        'YearsCodePro': [years_experience]
    })

    input_data['EdLevel'] = le_education.transform(input_data['EdLevel'])
    input_data['Country'] = le_country.transform(input_data['Country'])

    predicted_salary = regressor.predict(input_data)

    st.subheader("🔍 Estimated Salary")
    st.write(f"Based on your inputs, your estimated salary is: **${predicted_salary[0]:,.2f}**")

else:
    st.title("Explore Data Professional Salaries")
    st.markdown("### Stack Overflow Developer Survey 2020")

    df_vis = load_data()
    country_filter = st.multiselect("Filter by Country", df_vis["Country"].unique(), default=df_vis["Country"].unique())
    filtered_df = df_vis[df_vis["Country"].isin(country_filter)]

    # Key metrics
    st.write("### 💼 Key Salary Insights")
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Salary", f"${filtered_df['Salary'].mean():,.0f}")
    col2.metric("Median Salary", f"${filtered_df['Salary'].median():,.0f}")
    col3.metric("Max Salary", f"${filtered_df['Salary'].max():,.0f}")

    # Pie chart
    st.write("#### Distribution of Data from Different Countries")
    country_counts = filtered_df["Country"].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(country_counts, labels=country_counts.index, autopct="%1.1f%%", shadow=True, startangle=90)
    ax1.axis("equal")
    st.pyplot(fig1)

    # Bar chart: salary by country
    st.write("#### Mean Salary Based On Country")
    country_salary = filtered_df.groupby("Country")["Salary"].mean().sort_values(ascending=True)
    st.bar_chart(country_salary)

    # Line chart: salary by experience
    st.write("#### Mean Salary Based On Experience")
    exp_salary = filtered_df.groupby("YearsCodePro")["Salary"].mean().sort_values(ascending=True)
    st.line_chart(exp_salary)

    # Bar chart: salary by education
    st.write("#### Mean Salary Based on Education Level")
    edu_salary = filtered_df.groupby("EdLevel")["Salary"].mean().sort_values()
    st.bar_chart(edu_salary)

    # Box plot: salary by education
    st.write("#### Salary Distribution by Education Level")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.boxplot(x='EdLevel', y='Salary', data=filtered_df, ax=ax2)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    ax2.set_title("Salary Distribution by Education Level")
    ax2.set_ylabel("Salary")
    ax2.set_xlabel("Education Level")
    st.pyplot(fig2)

    # Scatter plot with legend
    st.write("#### Experience vs Salary Colored by Country")
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    filtered_df['Country_Code'] = filtered_df['Country'].astype('category').cat.codes
    scatter = ax3.scatter(filtered_df["YearsCodePro"], filtered_df["Salary"],
                          c=filtered_df["Country_Code"], cmap="tab20", alpha=0.6)
    ax3.set_xlabel("Years of Experience")
    ax3.set_ylabel("Salary")
    ax3.set_title("Experience vs Salary by Country")

    # Legend for scatter
    handles = []
    labels = []
    unique_codes = filtered_df[['Country', 'Country_Code']].drop_duplicates().sort_values('Country_Code')
    for _, row in unique_codes.iterrows():
        handles.append(plt.Line2D([], [], marker="o", linestyle="", 
                                  color=plt.cm.tab20(row['Country_Code'] / 20), alpha=0.6))
        labels.append(row['Country'])
    ax3.legend(handles, labels, title="Country", bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig3)

    # Histogram
    st.write("#### Salary Distribution Histogram")
    fig4, ax4 = plt.subplots()
    ax4.hist(filtered_df["Salary"], bins=30, color='skyblue', edgecolor='black', label='Salary')
    ax4.set_xlabel("Salary")
    ax4.set_ylabel("Frequency")
    ax4.set_title("Histogram of Salaries")
    ax4.legend()
    st.pyplot(fig4)

#%%
st.markdown("""
---
### About this App
This application uses data from a global survey of developers to predict salaries and explore salary trends. 
Please enter your details for prediction or explore the data visualizations.

Built with ❤️ using Streamlit and Scikit-learn.
""")
