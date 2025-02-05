#%% Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:,.3f}'.format

# loading dataset
att = pd.read_csv('speed_price_att.csv')

# dropping unnecessary columns
att.drop(columns=['collection_datetime', 'fn', 'address_full', 'incorporated_place', 'major_city', 'provider', 'speed_unit'], inplace=True)

# dropping the nan values
col_obs_todrop = ['price', 'technology', 'package']
att.dropna(subset=col_obs_todrop, how='any', inplace=True)

# dropping more unnecessary columns like categorical features as performing regression
att.drop(columns=['state', 'technology', 'package', 'redlining_grade', 'income_lmi', 'income_dollars_below_median'], inplace=True)

# handling the nan values
att['n_providers'].fillna(att['n_providers'].mode(), inplace=True)
att['ppl_per_sq_mile'].fillna(att['ppl_per_sq_mile'].median(), inplace=True)
att['internet_perc_broadband'].fillna(att['internet_perc_broadband'].median(), inplace=True)

# aggregating the dataset by grouping block_group using mean for all and sum for n_providers
aggregated_att = att.groupby('block_group').agg({
    'price': 'mean',
    'speed_down': 'mean',
    'speed_up': 'mean',
    'n_providers': 'sum',
    'ppl_per_sq_mile': 'mean',
    'internet_perc_broadband': 'mean',
    'lat': 'mean',
    'lon': 'mean',
    'race_perc_non_white': 'mean',
    'median_household_income': 'mean'
}).reset_index()

# using kmeans to remove the outliers
kmeans = KMeans(n_clusters=3, random_state=5805)
kmeans.fit(aggregated_att)

# Calculate the distances of each data point from the nearest centroid
distances_to_centroid = cdist(aggregated_att, kmeans.cluster_centers_, 'euclidean').min(axis=1)

# Set a threshold for anomaly detection (e.g., 95th percentile of the distance)
threshold = np.percentile(distances_to_centroid, 95)

# Flag points as outliers (1 for outlier, 0 for non-outlier)
aggregated_att['anomaly'] = (distances_to_centroid > threshold).astype(int)

# Filter out the outliers (keep only non-outliers)
aggregated_att = aggregated_att[aggregated_att['anomaly'] == 0].copy()

# Drop the 'anomaly' column if no longer needed
aggregated_att.drop(columns=['anomaly'], inplace=True)

X = aggregated_att.drop(columns=['speed_up'])

# target as speed_up for regression
y = aggregated_att['speed_up']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5805)

#%% Linear Regression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_train_pred = regressor.predict(X_train)
y_test_pred = regressor.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(y_train, y_train_pred, color='blue', label='Train Data')
plt.scatter(y_test, y_test_pred, color='red', label='Test Data')
plt.plot([min(y), max(y)], [min(y), max(y)], color='green', linestyle='--', label='Perfect Fit')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Multi-linear Regression Analysis')
plt.legend()
plt.grid(True)
plt.show()

X_train_with_const = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train_with_const).fit()

r2 = r2_score(y_test, y_test_pred)
mse = mean_squared_error(y_test, y_test_pred)

# dictionary for the metrics to create a table
metrics = {
    'Metric': ['R-squared', 'Adjusted R-squared', 'AIC', 'BIC', 'MSE'],
    'Value': [
        r2,
        model.rsquared_adj,
        model.aic,
        model.bic,
        mse
    ]
}

metrics_table = pd.DataFrame(metrics)
metrics_table.set_index('Metric', inplace=True)

print("\n************************ PHASE II: REGRESSION ANALYSIS ************************")

print(f"\nMultiple Linear Regression Metrics table:\n{metrics_table}")

print("\n*************************************************************************************")

#%% T-test analysis

t_test_results = model.tvalues
print(f"\nT-test results:\n{t_test_results}")

print("\n*************************************************************************************")

#%% F-test analysis

f_test_result = model.fvalue
print(f"\nF-test result:\n{f_test_result}")

print("\n*************************************************************************************")

#%% Confidence Interval analysis

confidence_intervals = model.conf_int(alpha=0.05)
print(f"\n95% Confidence Intervals for Coefficients:\n{confidence_intervals}")

print("\n*************************************************************************************")

#%% Stepwise Regression

def stepwise_regression(X_train, y_train, significance_level=0.01):
    X_train_with_const = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train_with_const).fit()
    eliminated_features = []

    while True:
        p_values = model.pvalues[1:]
        max_p_value = p_values.max()

        if max_p_value > significance_level:
            eliminated_feature = p_values.idxmax()
            print(f"Eliminating {eliminated_feature} with a p-value of {max_p_value:.4f}")
            eliminated_features.append(eliminated_feature)
            X_train_with_const = X_train_with_const.drop(columns=[eliminated_feature])
            model = sm.OLS(y_train, X_train_with_const).fit()
        else:
            break

    print(f"\nFinal Model:\n{model.summary()}")
    print()

    if eliminated_features:
        print("\nEliminated Features:")
        for feature in eliminated_features:
            print(feature)
    else:
        print("\nNo features were eliminated.")

    return model

stepwise_model = stepwise_regression(X_train, y_train)

print("\n************************ END OF PHASE II ************************")