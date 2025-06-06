# 💰 Salary Predictor & Explorer

This interactive web application helps users **predict software engineering salaries** based on education, experience, and country using data from the **Stack Overflow Developer Survey 2020**. Users can also explore global salary trends through a variety of visualizations.

Built with **Streamlit**, **scikit-learn**, **Pandas**, **Matplotlib**, and **Seaborn**.

---

## 🚀 Features

### 🔮 Salary Prediction
- Input your **education level**, **country**, and **years of professional coding experience**
- Get a **predicted salary** using a trained **Decision Tree Regressor**

### 📊 Data Exploration
- Visualize global salary trends and distributions
- Explore how **education**, **experience**, and **country** affect salary
- Interactive charts:
  - Pie chart: Data distribution by country
  - Bar chart: Average salary by country
  - Line chart: Average salary by experience
  - Box plot: Salary distribution by education level
  - Scatter plot: Experience vs Salary by country (with legend)
  - Histogram: Overall salary distribution

### 📦 Technologies Used
- **Python**
- **Streamlit** for the web interface
- **Pandas** and **NumPy** for data processing
- **scikit-learn** for model training
- **Matplotlib** and **Seaborn** for plotting
- **GridSearchCV** for model tuning
- **LabelEncoder** for handling categorical data

---

## 📁 Dataset

The app uses the [Stack Overflow Developer Survey 2020](https://insights.stackoverflow.com/survey/2020) dataset. Only relevant columns such as:
- `Country`
- `EdLevel` (Education level)
- `YearsCodePro` (Years of professional experience)
- `ConvertedCompYearly` (Annual salary)

Preprocessing steps include:
- Cleaning extreme outliers in salary
- Standardizing education levels
- Simplifying country categories
- Handling missing values

---

## 💻 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/salary-predictor-app.git
   cd salary-predictor-app
