# Diabetes Prediction Model

This project aims to develop a machine learning model that predicts whether individuals have diabetes based on specified features. The dataset used for this project comes from a study conducted by the National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK), focusing on Pima Indian women aged 21 and older.

## Project Overview

The primary goal is to apply feature engineering and data analysis techniques to develop an accurate prediction model. The outcome variable indicates whether the patient has diabetes (**1** for positive and **0** for negative results).

## Dataset

The dataset contains 768 observations and 9 features:

- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration after 2 hours in an oral glucose tolerance test (mg/dL)
- **Blood Pressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skinfold thickness (mm)
- **Insulin**: 2-hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **DiabetesPedigreeFunction**: Diabetes pedigree function score
- **Age**: Age in years
- **Outcome**: Class variable (0 or 1), where **1** indicates diabetes, and **0** indicates no diabetes

## Project Tasks

### 1. Exploratory Data Analysis (EDA)
- Analyze the overall structure of the dataset.
- Examine numerical and categorical features.
- Analyze the target variable.
- Identify and handle outliers and missing data.
- Perform correlation analysis.

### 2. Feature Engineering
- Handle missing and incorrect values (e.g., converting 0 values to NaN where applicable).
- Create new features based on existing ones.
- Perform encoding for categorical variables.
- Standardize numerical variables.

### 3. Model Building
- Train and evaluate various machine learning models to predict diabetes.
- Compare model performance and tune hyperparameters.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/diabetes-prediction.git
    ```

2. Navigate to the project directory:
    ```bash
    cd diabetes-prediction
    ```

3. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Once the dependencies are installed, you can start exploring the dataset, performing feature engineering, and training the models. Use the Jupyter notebooks provided to walk through each step of the project.

## License

This project is licensed under the MIT License.
