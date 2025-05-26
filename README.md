# 📊 Customer Churn Analysis & Prediction
## Project Overview
This application is designed to help businesses predict customer churn, which refers to the likelihood of customers discontinuing their services or subscriptions. By identifying customers at risk of churn, businesses can implement targeted retention strategies to improve customer satisfaction and reduce revenue loss. Using Gradient Boosting for prediction and an interactive Streamlit dashboard, this solution offers actionable insights for businesses to enhance customer retention and profitability in the telecom sector.

## 📊 Data Source
The dataset used for this project is the Telcom Customer Churn dataset, which includes customer information such as demographics, account details, and usage patterns.
You can find it in my Dataset folder


## 🛠️ Technologies Used
**Power BI:** For in-depth data analysis and visualization of churn patterns.

**Python:** For data processing and machine learning model development.

**Streamlit:** For building and deploying an interactive web application.

**Pandas:** For data manipulation and cleaning.

**Plotly:** For interactive visualizations.

**Scikit-learn:** For machine learning model training.

**SMOTEENN:** For balancing the dataset.

**Joblib:** For saving and loading the trained model.

## 🧩 Key Features
**Customer Insights:** Leverage Power BI for analyzing customer behavior, demographics, and usage patterns.

**Churn Comparison:** Visualize and compare actual churn rates with predicted churn probabilities.

**Prediction Process:** Utilize Gradient Boosting for seamless churn prediction via the Streamlit app.

**Interpretation Assistance:** Get clear explanations for churn predictions and influential features.

**Retention Strategies:** Generate actionable retention strategies based on predictions.

**Data Security:** Ensures customer data confidentiality and compliance with data protection regulations.

**Interactive Dashboards:** Use Power BI and Streamlit for interactive churn trends and insights.

## 📊 Data Visualizations
**Customer Segmentation & Churn Analysis**
Use Power BI to segment customers by demographics, usage patterns, and contract types. Visualizations track churn trends, allowing businesses to predict spikes and act proactively.

## Customer Churn Dashboard Overview

**Churn Prediction Results**

You can explore the full Power BI Churn Dashboard.

## ⚙️ How the Model Works
Churn Prediction with Streamlit
Our Streamlit app allows for real-time churn prediction using a Gradient Boosting Classifier. You can predict churn for individual customers (online mode) or in bulk (batch mode), providing businesses with crucial insights.
You can try out the model here: https://telecom-customer-churn-analysis-prediction.streamlit.app/

### Online Prediction Mode

Predict churn for individual customers in real-time.

Enter customer data such as tenure, services used, and contract details.

### Batch Prediction Mode

Upload a CSV file with multiple customer records to predict churn for a bulk dataset.

View the results in an interactive format.

## Key Benefits:

Improved Retention: Proactively identify and retain high-risk customers.

Enhanced Decision Making: Leverage churn insights to improve resource allocation.

Cost Savings: Reduce customer acquisition costs by retaining existing customers.

Competitive Advantage: Stand out by providing exceptional customer service with data-driven retention strategies.

## 🧑‍💻 How to Run the App Locally
1. Clone the repository
bash

git clone https://github.com/your-username/Telecom-Customer-Churn-Analysis-Prediction.git
cd Telecom-Customer-Churn-Analysis-Prediction
2. Install dependencies
Make sure you have Python 3.10+ installed. Run the following to install required libraries:

bash
pip install -r requirements.txt

3. Run the Streamlit app
bash
streamlit run app.py

5. Access the app
Visit http://localhost:8501 in your web browser to access the Customer Churn Prediction App.

## 📁 Project Structure

```plaintext
telecom-churn-analysis/
│
├── data/
│   ├── Telco-Customer-Churn.csv         ← Raw dataset
│   └── tel_churn_clean.csv             ← Cleaned and preprocessed dataset
│
├── notebooks/
│   └── churn_analysis_model_training.ipynb  ← Jupyter notebook with analysis and model training
│
├── app/
│   └── app.py                         ← Core Streamlit application
│
├── model/
│   └── Bi_gradient_boosting_model.joblib  ← Trained model
│
├── dashboard/
│   └── Customer Churn Dashboard.pbix  ← PowerBI file
│
├── screenshots/
│   ├── dashboard_overview.png
│   └── batch_prediction_result.png     ← Dashboard screenshots for README
│
├── retrain.py                         ← Model retraining script
├── requirements.txt                   ← Python dependencies
├── README.md                          ← Project overview and setup instructions
└── .gitignore                         ← Git ignore file
```

## 📦 Requirements

streamlit
pandas
numpy
scikit-learn==1.2.2
imbalanced-learn
xgboost
pycaret
matplotlib
seaborn
plotly
joblib

## 🔧 Challenges & Limitations
Model Bias: The churn prediction model may exhibit some biases, especially with imbalanced data.

Interpretability: Machine learning models, including Gradient Boosting, can sometimes be difficult to interpret.

Evolving Data: Continuous model updates may be required as customer behaviors change over time.

Data Quality: The performance of the model depends on the accuracy and completeness of the input data.
