# 🎓 Student Performance Prediction System

> **Machine Learning Capstone Project**\
> An interactive academic analytics system that predicts student
> examination performance, compares students with the available dataset,
> identifies subject-level gaps, and generates personalized
> recommendations.

------------------------------------------------------------------------

## 🚀 Project Overview

The **Student Performance Prediction System** uses machine learning to
analyze academic, behavioral, and learning-environment factors and
estimate a student's examination score.

Instead of providing only a numerical prediction, the application
combines:

-   📊 Exploratory data analysis
-   🤖 Multiple machine learning models
-   🎯 Individual score prediction
-   📈 Student-to-dataset comparison
-   📚 Subject-wise analysis
-   📉 Performance trend visualization
-   💡 Personalized recommendations
-   🖥️ Interactive Streamlit dashboard

------------------------------------------------------------------------

## 🎯 Problem Statement

Student performance is influenced by multiple factors such as
attendance, study hours, previous academic performance, tutoring,
motivation, learning resources, and participation.

Manual analysis of these factors can be time-consuming and may not
provide a consistent, personalized view of a student's academic
situation.

This project addresses the problem by building an interactive machine
learning application that predicts examination performance and converts
the prediction into understandable academic insights and
recommendations.

------------------------------------------------------------------------

## ✨ Key Features

### 🔮 Student Prediction

Enter student information and obtain:

-   Predicted examination score
-   Performance category
-   Dataset average comparison
-   Similar-student comparison
-   Dataset percentile

### 📚 Subject Analysis

The application analyzes:

-   Mathematics
-   Science
-   English
-   Internal Marks
-   Participation

It identifies the strongest and weakest subject for the entered student
profile.

### 📈 Performance Trend

The dashboard visualizes:

**Previous Score → Internal Marks → Predicted Exam Score**

### 💡 Personalized Recommendations

Recommendations can focus on:

-   Attendance
-   Study hours
-   Weak subjects
-   Internal marks
-   Participation
-   Tutoring
-   Sleep and learning consistency

### 📊 Analytics Dashboard

The analytics section includes:

-   Attendance vs Exam Score
-   Hours Studied vs Exam Score
-   Previous Scores vs Exam Score
-   Internal Marks vs Exam Score
-   Motivation vs Exam Score
-   Participation vs Exam Score
-   Tutoring Sessions vs Exam Score
-   Subject averages
-   Score distribution
-   Feature correlations

### 🤖 Model Comparison

Three regression algorithms are trained and evaluated:

1.  Linear Regression
2.  Random Forest Regression
3.  Decision Tree Regression

------------------------------------------------------------------------

## 📁 Dataset

The base dataset is:

``` text
data/StudentPerformanceFactors.csv
```

The original dataset contains **6,607 student records** and **20
attributes**.

Important features include:

-   Hours Studied
-   Attendance
-   Previous Scores
-   Tutoring Sessions
-   Motivation Level
-   Access to Resources
-   Parental Involvement
-   Teacher Quality
-   Peer Influence
-   Sleep Hours
-   Physical Activity
-   Family Income
-   School Type
-   Learning Disabilities
-   Parental Education Level
-   Exam Score

### Extended academic attributes

To support the project's required subject-level and internal-performance
analysis, the working dataset was extended with:

-   Mathematics Score
-   Science Score
-   English Score
-   Internal Marks
-   Participation

These five attributes are **self-created prototype fields**, generated
reproducibly from non-target student factors using a fixed random seed.
They were not generated from `Exam_Score`.

> For real institutional deployment, these prototype fields should be
> replaced with verified subject, internal-assessment, and participation
> records.

------------------------------------------------------------------------

## 🧹 Data Preprocessing

The preprocessing pipeline performs:

1.  Missing-value detection
2.  Duplicate checking
3.  Numerical median imputation
4.  Categorical most-frequent imputation
5.  One-Hot Encoding
6.  StandardScaler normalization
7.  Target-score validation/clipping to 0--100
8.  80:20 train-test split

The preprocessing is implemented as a Scikit-learn pipeline so the same
transformations are applied during training and prediction.

------------------------------------------------------------------------

## 🧠 Machine Learning

The problem is formulated as a **regression task** where the target is:

``` text
Exam_Score
```

### Models

  Model                           MAE        RMSE          R²
  ----------------------- ----------- ----------- -----------
  **Linear Regression**     **0.457**   **1.804**   **0.770**
  Random Forest                 1.016       2.096       0.689
  Decision Tree                 1.422       2.469       0.569

The deployed model is **Linear Regression**, selected using the lowest
test RMSE.

### Performance Categories

  Predicted Score   Category
  ----------------- ------------
  80--100           🟢 High
  60--79.99         🟡 Average
  Below 60          🔴 Low

------------------------------------------------------------------------

## 🏗️ System Workflow

``` text
Student Data
     ↓
Data Validation
     ↓
Preprocessing
     ↓
Feature Encoding & Scaling
     ↓
Machine Learning Model
     ↓
Predicted Exam Score
     ↓
Performance Classification
     ↓
Student Comparison
     ↓
Subject & Trend Analysis
     ↓
Personalized Recommendations
     ↓
Interactive Streamlit Dashboard
```

------------------------------------------------------------------------

## 🛠️ Tech Stack

### Programming

-   Python

### Data Processing

-   Pandas
-   NumPy

### Machine Learning

-   Scikit-learn

### Visualization

-   Matplotlib
-   Seaborn

### Web Application

-   Streamlit

### Model Persistence

-   Joblib

------------------------------------------------------------------------

## 📂 Project Structure

``` text
Student_Performance_Prediction/
│
├── app.py
├── upgrade_project.py
├── requirements.txt
├── README.md
│
├── data/
│   └── StudentPerformanceFactors.csv
│
├── models/
│   ├── best_model.joblib
│   ├── linear_regression.joblib
│   ├── random_forest.joblib
│   ├── decision_tree.joblib
│   └── model_results.csv
│
└── .venv/
```

------------------------------------------------------------------------

## ▶️ How to Run Locally

### 1. Clone the repository

``` bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Student_Performance_Prediction
```

### 2. Create a virtual environment

``` bash
python -m venv .venv
```

### 3. Activate the environment

**Windows PowerShell:**

``` powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

``` bash
pip install -r requirements.txt
```

### 5. Retrain the models

``` bash
python upgrade_project.py
```

### 6. Start the Streamlit application

``` bash
streamlit run app.py
```

The application will open in the browser.

------------------------------------------------------------------------

## 🖥️ Application Pages

### 🔮 Predict Student

Individual student prediction with academic profile, subject scores,
comparison, trend and recommendations.

### 🏠 Dashboard

High-level statistics and visual summaries of the student dataset.

### 📊 Analytics

Detailed relationships between academic factors and examination scores.

### 🤖 Model Performance

Comparison of MAE, RMSE and R² for all trained regression models.

------------------------------------------------------------------------

## 📌 Example Prediction

A demonstration input produced:

``` text
Predicted Exam Score: 67.5
Performance Level: Average
Dataset Average: 67.2
Dataset Percentile: 54%
Strongest Subject: Mathematics — 92
Weakest Subject: Science — 61
```

The recommendation engine then suggests targeted improvements based on
the student's profile.

------------------------------------------------------------------------

## 🔐 Data & Responsible Use

This project is an academic prototype.

Predictions should be treated as **decision-support information**, not
as guaranteed academic outcomes.

For real institutional use, the system should include:

-   Verified institutional data
-   Authentication
-   Role-based access
-   Privacy controls
-   Secure storage
-   Model monitoring
-   Bias and fairness evaluation
-   Periodic model retraining

------------------------------------------------------------------------

## 🔮 Future Scope

Possible extensions include:

-   Real institutional subject and internal-assessment data
-   MySQL/cloud database integration
-   Teacher and administrator dashboards
-   Semester-by-semester performance tracking
-   Explainable AI using SHAP or feature importance
-   Academic-risk alerts
-   Automated intervention tracking
-   Model monitoring and retraining
-   Cloud deployment

------------------------------------------------------------------------

## 📊 Project Outcome

The completed prototype demonstrates an end-to-end machine learning
workflow:

**Data → Preprocessing → EDA → Model Training → Evaluation → Prediction
→ Analytics → Recommendations → Dashboard**

The system successfully combines machine learning prediction with
interactive academic analytics to make student-performance data easier
to understand and act upon.

------------------------------------------------------------------------

## 👩‍💻 Project Type

**Machine Learning Capstone Project**

**Domain:** Education / Academic Analytics

**Primary Model:** Linear Regression

**Interface:** Streamlit
