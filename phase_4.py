#%% Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# loading the dataset
att = pd.read_csv('speed_price_att.csv')

# dropping the unnecessary columns
att.drop(columns=['collection_datetime', 'fn', 'address_full', 'incorporated_place', 'major_city', 'provider', 'speed_unit'], inplace=True)

# handling missing values
att['price'].fillna(att['price'].median(), inplace=True)
att['ppl_per_sq_mile'].fillna(att['ppl_per_sq_mile'].median(), inplace=True)
att['internet_perc_broadband'].fillna(att['internet_perc_broadband'].median(), inplace=True)
att.fillna(att.mean(numeric_only=True), inplace=True)

# creating a new feature with speed_down
att['speed_category'] = pd.qcut(att['speed_down'], q=2, labels=['Slow', 'Fast'])

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

# downsampling the dataset (reduce dataset size to 30% for faster computation)
att_downsampled = att.sample(frac=0.3, random_state=5805)

# creating a copy for clustering
att_clustering = att_downsampled.copy()

# dropping the categorical columns for clustering
att_clustering.drop(columns=['state', 'redlining_grade', 'technology', 'package', 'speed_category', 'income_dollars_below_median'], inplace=True)

# standardizing the dataset for clustering
scaler = StandardScaler()
scaled_data = scaler.fit_transform(att_clustering)

#%% K-Means Clustering

print("\n************************ K-MEANS CLUSTERING ************************")

wcss = []
silhouette_scores_kmeans = []
k_values = range(2, 11)

# Calculate WCSS and silhouette scores for each k
for k in k_values:
    print(f"\nChecking for k={k}")
    kmeans = KMeans(n_clusters=k, random_state=5805, init='k-means++')
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)
    silhouette_scores_kmeans.append(silhouette_score(scaled_data, kmeans.labels_))

# Elbow Method Plot for WCSS
plt.figure(figsize=(10, 6))
plt.plot(k_values, wcss, marker='o')
plt.title('Elbow Method for Optimal K (WCSS)')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# Silhouette Analysis Plot
plt.figure(figsize=(10, 6))
plt.plot(k_values, silhouette_scores_kmeans, marker='o', color='orange')
plt.title('Silhouette Analysis for KMeans')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()

# finding optimal K using WCSS
wcss_diff = np.diff(wcss)
wcss_diff2 = np.diff(wcss_diff)

optimal_k_wcss = np.argmin(wcss_diff2) + 2
print(f"\nOptimal K (using WCSS - Elbow Method): {optimal_k_wcss}")

# finding optimal K using silhouette scores
optimal_k_silhouette = k_values[np.argmax(silhouette_scores_kmeans)]
print(f"\nOptimal K (using Silhouette Score): {optimal_k_silhouette}")

#%% DBSCAN Clustering

print("\n************************ DBSCAN CLUSTERING ************************")

nearest_neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = nearest_neighbors.fit(scaled_data)
distances, indices = neighbors_fit.kneighbors(scaled_data)
sorted_distances = np.sort(distances[:, 4])

plt.figure(figsize=(10, 6))
plt.plot(sorted_distances)
plt.title('Nearest Neighbor Distance Analysis (for eps selection)')
plt.xlabel('Data Points')
plt.ylabel('5th Nearest Neighbor Distance')
plt.grid(True)
plt.show()

dbscan = DBSCAN(eps=1.5, min_samples=8)
dbscan_labels = dbscan.fit_predict(scaled_data)

n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)
print(f"DBSCAN Number of clusters: {n_clusters}")
print(f"DBSCAN Number of noise points: {n_noise}")

#%% Apriori Algorithm for Association Rule Mining

print("\n************************ APRIORI ALGORITHM ************************")

basket = pd.get_dummies(att_downsampled[['package', 'technology', 'speed_category']])

frequent_itemsets = apriori(basket, min_support=0.3, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7, num_itemsets=3)

print(f"\nFrequent Itemsets:\n{frequent_itemsets}")
print(f"\nAssociation Rules:\n{rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]}")

print("\n************************ END OF PHASE IV ************************")
