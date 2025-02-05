# AT&T Internet Speeds and Prices Analysis

## Project Overview
This project analyzes the AT&T dataset on internet speeds, prices, locations, and socioeconomic factors
to explore how regional differences impact broadband availability, affordability, and quality in the
United States. It highlights how disparities and market dynamics shape internet services across different
areas.

The project is divided into four phases. Phase I focuses on feature engineering and exploratory data
analysis (EDA) to refine prediction accuracy. Phase II uses regression analysis on upload speeds to
identify key predictors, like download speeds and broadband usage, and their impact on performance.
Phase III compares seven classification models, including Logistic Regression, Neural Networks, and
Decision Trees, to predict internet package types. Logistic Regression stood out for its simplicity,
efficiency, and high performance. Phase IV uses clustering models, like K-Means and DBSCAN, and
association rule mining to uncover patterns and relationships between features.

## Data Description
The dataset contains information on AT&T's broadband services, including various internet package types, speeds, prices, and service features across different regions.

The dataset originally consists of 432,303 observations and 26 features, with the following breakdown:
- 11 Categorical Features
- 15 Numerical Features

**Numerical Features:**
- These include Download Speeds, Upload Speeds, Latitude, Longitude, Number of Providers, and other related variables.

**Categorical Features:**
- These include State, Technology, Major City, Package Type, and other categorical variables.

**Target Variables:**
- Phase I & II (Regression): ‘speed_up’ – Upload Speeds.
- Phase III (Classification): ‘package’ – Type of internet package, with 4 classes.


## Objectives
- **Phase I:** Perform feature engineering and exploratory data analysis (EDA) to understand the dataset,
implement different models for feature selection, and choose the best feature selection model for
further implementation.
- **Phase II:** Implement regression on a continuous feature (‘speed_up’) to predict and analyze the
upload speed using other predictors in the dataset.
- **Phase III:** Implement seven different classifiers on the categorical target variable (‘package’),
analyze the results, and pick the best classifier based on their performances and other metrics.
- **Phase IV:** Implement clustering using two clustering models, perform association rule mining using
Apriori algorithm model, and analyze the results.

## Analysis
- **Feature Selection & Regression:** Applied various feature selection/dimensionality reduction methods such as VIF, PCA, Random Forest to select key predictors and used regression analysis to model upload speeds, identifying critical features influencing the target.
- **Classification Analysis:** Evaluated seven models (Logistic Regression, SVM, KNN, Decision Trees, etc.), selecting Logistic Regression for its balance of accuracy, efficiency, and interpretability in predicting internet package types.
- **Clustering & Pattern Discovery:** Employed K-means and DBSCAN to identify broadband distribution patterns and used association rule mining (Apriori) to uncover relationships between package types, technologies, and speeds.
