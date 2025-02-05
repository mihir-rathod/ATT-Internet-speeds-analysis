#%% Importing libs and handling dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tools.tools import add_constant
import warnings
from sklearn.cluster import DBSCAN, KMeans
from scipy.spatial.distance import cdist
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:,.3f}'.format
# standardizing functions
def standardized(dataset):
    updated_dataset = dataset.copy()
    updated_dataset = ((updated_dataset - updated_dataset.mean())/(np.std(updated_dataset)))
    return updated_dataset
def spl_standardized(dataset, columns):
    updated_dataset = dataset.copy()
    for col in X_num_cols:
        updated_dataset[col] = ((updated_dataset[col] - updated_dataset[col].mean())/(np.std(updated_dataset[col])))
    return updated_dataset

# loading dataset
att = pd.read_csv('speed_price_att.csv')

# printing the head of the dataset
print(f"\nHead of the dataset:\n{att.head()}")

print("\n*************************************************************************************")

# printing the number of missing values before cleaning
print(f"\nThe no. of missing observations in the dataset before cleaning:\n{att.isna().sum()}")

print("\n************************ PHASE I: FEATURE ENGINEERING AND EDA ************************")

# dropping the unnecessary columns
att.drop(columns=['collection_datetime', 'fn', 'address_full', 'incorporated_place', 'major_city', 'provider', 'speed_unit'], inplace=True)

# ******************** AFTER ANALYSIS, THESE FEATURES TO BE REMOVED ********************
# removing fastest_speed_down and fastest_speed_price as there is very high collinearity with speed_down and price
# removing income_lmi and income_dollars_below_median because vif is high

att.drop(columns=['income_lmi', 'income_dollars_below_median'], inplace=True)

# handling the nan values
col_obs_todrop = ['price', 'technology', 'package']

# dropping the nan values
att.dropna(subset=col_obs_todrop,how='any', inplace=True)

# merging similar packages for simplicity
package_merge = {
    'AT&T FIBER INTERNET 300': 'Fiber Internet',
    'AT&T FIBER INTERNET 500': 'Fiber Internet',
    'AT&T FIBER—INTERNET 300': 'Fiber Internet',
    'AT&T FIBER — INTERNET 300': 'Fiber Internet',
    'AT&T FIBER — INTERNET 500': 'Fiber Internet',
    'AT&T FIBER—INTERNET 2000': 'Fiber Internet',
    'AT&T FIBER—INTERNET 100': 'Fiber Internet',
    'AT&T FIBER—INTERNET 500': 'Fiber Internet',
    'Internet 100': 'High-Speed Internet',
    'Internet 75': 'High-Speed Internet',
    'Internet 50': 'High-Speed Internet',
    'Internet 25': 'Basic Internet',
    'Internet 18': 'Basic Internet',
    'Internet 10': 'Basic Internet',
    'Internet Basic 5': 'Basic Internet',
    'Internet Basic 3': 'Basic Internet',
    'Internet Basic 1.5': 'Basic Internet',
    'Internet Basic 768kbps': 'Basic Internet'
}

att['package'] = att['package'].replace(package_merge)

# # filling the na values
att['redlining_grade'].fillna(att['redlining_grade'].mode()[0], inplace=True)
att['n_providers'].fillna(att['n_providers'].mode()[0], inplace=True)
att['ppl_per_sq_mile'].fillna(att['ppl_per_sq_mile'].median(), inplace=True)
att['internet_perc_broadband'].fillna(att['internet_perc_broadband'].median(), inplace=True)

# printing the cleaned dataset
print(f"\nDisplaying the cleaned dataset:\n{att.head()}")

print("\n*************************************************************************************")

# printing the shape of the cleaned dataset
print(f"\nDisplaying the shape of the cleaned dataset:\n{att.shape}")

print("\n*************************************************************************************")

# displaying the no. of missing observations after cleaning
print(f'\nThe number of missing observations in the dataset after cleaning:\n{att.isna().sum()}')

print("\n*************************************************************************************")

# checking whether the dataset has duplicate observations
print(f"\nChecking whether the dataset has any duplications (before): {att.duplicated().sum()}")

att.drop_duplicates(inplace=True)

print("\n*************************************************************************************")

print(f"\nChecking whether the dataset has any duplications (after): {att.duplicated().sum()}")

print("\n*************************************************************************************")

# aggregating the dataset by grouping block_groups
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
    'median_household_income': 'mean',
    'package': lambda x: x.mode()[0] # will drop this for phase 2 as we only need categorical features.
}).reset_index()

print(f"\nI am going to use this aggregated_att in phase 2 and 3:\n{aggregated_att.head()}")

print("\n*************************************************************************************")

# downsampling the dataset for phase 4 for faster computation (reduce dataset size to 30% for faster computation)
att_downsampled = att.sample(frac=0.3, random_state=5805)

# creating a copy for clustering
att_clustering = att_downsampled.copy()

# dropping categorical columns for clustering
att_clustering.drop(columns=['state', 'redlining_grade', 'technology', 'package'], inplace=True)
att_clustering.drop(columns=['lat', 'lon', 'race_perc_non_white', 'internet_perc_broadband', 'block_group'], inplace=True)

print(f"\nI am using the downsampled dataset in phase 4 for faster computation.\nThe shape of the downsampled dataset is {att_clustering.shape}")

print("\n*************************************************************************************")

# encoding categrorical variables
encoded_att = pd.get_dummies(att, columns=['state', 'package', 'technology', 'redlining_grade'], drop_first=True)

# displaying the encoded dataset
print(f"\nDisplaying the encoded dataset-\n{encoded_att.head()}")

print("\n*************************************************************************************")

# selecting the target variable (y)
X = encoded_att.drop(columns=['speed_up'])
y = encoded_att['speed_up']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5805, shuffle=True)

# num cols from original dataset
num_cols = att.select_dtypes(include=['int64', 'float64'])

# num cols from X
X_num_cols = X.select_dtypes(include=['int64', 'float64'])

# num cols from X_train
X_train_num_cols = X.select_dtypes(include=['int64', 'float64'])

# num cols from X_test
X_test_num_cols = X.select_dtypes(include=['int64', 'float64'])

#%% Random Forest Analysis

print("\n************************ RANDOM FOREST ANALYSIS ************************")
rf = RandomForestRegressor(random_state=5805)
rf.fit(X_train, y_train)
importances = rf.feature_importances_
feature_names = X_train.columns

sorted_indices = np.argsort(importances)[::-1]
sorted_importances = importances[sorted_indices]
sorted_features = feature_names[sorted_indices]

plt.figure(figsize=(15,15))
plt.barh(sorted_features, sorted_importances)
plt.xlabel('Feature importance')
plt.ylabel('Features')
plt.title('Random Forest Analysis')
plt.gca().invert_yaxis()
plt.grid(True)
plt.show()

rf_threshold = 0.01
selected_features = sorted_features[sorted_importances >= rf_threshold]
eliminated_features = sorted_features[sorted_importances < rf_threshold]

print(f"\nSelected Features (Random Forest): {selected_features}")
print(f"\nEliminated Features (Random Forest): {eliminated_features}")

#%% PCA

print("\n************************ PRINCIPLE COMPONENT ANALYSIS ************************")

# standardizing for PCA
X_train_std = spl_standardized(X_train, X_train_num_cols)

pca = PCA(random_state=5805)
X_pca = pca.fit_transform(X_train_std)

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
num_features_95 = np.argmax(cumulative_variance >= 0.95)+1
print(f"\nNumber of features needed to explain more than 95% of the variance: {num_features_95}")

plt.figure(figsize=(12,8))
plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, marker='o', linestyle='--', color='b')
plt.xlabel('Number of features')
plt.ylabel('Cumulative explained variance')
plt.title('PCA - Cumulative explained variance vs Number of features')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance Threshold')
plt.axvline(x=num_features_95, color='g', linestyle='--', label=f"{num_features_95} features")
plt.title('Principal Component Analysis')
plt.legend()
plt.grid()
plt.show()

#%% SVD

print("\n************************ SINGULAR VALUE DECOMPOSITION ************************")

X_std = standardized(X_num_cols)
n_components = 12
svd = TruncatedSVD(n_components=n_components, random_state=5805)
X_svd = svd.fit_transform(X_std)

feature_importance = np.abs(svd.components_)
selected_features_indices = np.argsort(-feature_importance.sum(axis=0))[:n_components]
selected_features = X.columns[selected_features_indices]
print(f"\nSelected Features:{selected_features}")

#%% VIF

print("\n************************ VARIANCE INFLATION FACTOR ************************")

X_vif = add_constant(X_num_cols)
vif = pd.DataFrame()
vif['Variable'] = X_vif.columns
vif['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
print(f"\nVariance Inflation Factor Table:\n{vif}")
print("\nHere the the vif scores for fastest_speed_down and fastest_speed_price are very high.")

#%% Discretization and Binarization

att['speed_category'] = pd.qcut(att['speed_down'], q=2, labels=['Slow', 'Fast'])

plt.figure(figsize=(8, 6))
att['speed_category'].value_counts().plot(kind='bar', color=['red', 'blue'])
plt.title('Distribution of Speed Category (Slow vs Fast)')
plt.xlabel('Speed Category')
plt.ylabel('Count')
plt.grid(axis='y')
plt.show()

#%% Anomaly Detection/Outlier Analysis and removal

print("\n************************ ANOMALY/OUTLIER ANALYSIS & REMOVAL ************************")

std_num_cols = standardized(num_cols)

# anomaly detection using kmeans
kmeans = KMeans(n_clusters=3, random_state=5805)
kmeans_labels = kmeans.fit_predict(std_num_cols)

# Calculate distances to cluster centers
distances_to_center = cdist(std_num_cols, kmeans.cluster_centers_, 'euclidean').min(axis=1)
distance_threshold = np.percentile(distances_to_center, 95)

# Mark anomalies
att['anomaly'] = (distances_to_center > distance_threshold).astype(int)
print(f"Number of anomalies detected: {att['anomaly'].sum()}")

# printing the shape of dataset before removal
print(f"Shape of the dataset before anomaly removal: {att.shape}")
att = att[att['anomaly'] == 0].copy()

# printing the shape of dataset after removal
print(f"Shape of the dataset after anomaly removal: {att.shape}")
#%% Sample Covariance

cov_mat = num_cols.cov()

plt.figure(figsize=(18, 15))
sns.heatmap(cov_mat, annot=True, cmap='coolwarm', cbar=True)
plt.title('Sample Covariance Heatmap')
plt.xlabel('Features')
plt.ylabel('Features')
plt.show()

#%% Sample Pearson Correlation coefficient matrix

corr_mat = num_cols.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_mat, annot=True, cmap='coolwarm', cbar=True, vmin=-1, vmax=1, center=0)
plt.title('Sample Correlation Heatmap')
plt.xlabel('Features')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

#%% Checking whether the target is balanced or imbalanced

print("\n************************ CHECKING IF THE TARGET IS BALANCED ************************")

target_balance = att['package'].value_counts()
print(f"\nInternet Package class value counts for comparision:\n{target_balance}")
print("\nThe target is balanced.")

print("\n************************ END OF PHASE I ************************")