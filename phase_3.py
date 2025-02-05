#%% Importing necessary libraries and handling dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:,.3f}'.format

# loading the dataset
att = pd.read_csv('speed_price_att.csv')

# dropping the unnecessary columns for classification
att.drop(columns=['collection_datetime', 'fn', 'address_full', 'incorporated_place', 'major_city',
                  'provider', 'speed_unit', 'fastest_speed_down', 'fastest_speed_price',
                  'income_lmi', 'income_dollars_below_median'], inplace=True)

# dropping the na values
col_obs_todrop = ['price', 'technology', 'package']
att.dropna(subset=col_obs_todrop, how='any', inplace=True)

# handling the nan values
att['redlining_grade'].fillna(att['redlining_grade'].bfill(), inplace=True)
att['n_providers'].fillna(att['n_providers'].bfill(), inplace=True)
att['ppl_per_sq_mile'].fillna(att['ppl_per_sq_mile'].median(), inplace=True)
att['internet_perc_broadband'].fillna(att['internet_perc_broadband'].median(), inplace=True)

# merging similar packages
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

# aggregating data by block_group
aggregated_att = att.groupby('block_group').agg({
    'price': 'mean',
    'speed_down': 'mean',
    'speed_up': 'mean',
    'n_providers': 'sum',
    'ppl_per_sq_mile': 'mean',
    'internet_perc_broadband': 'mean',
    'lat': 'mean',
    'lon': 'mean',
    'package': lambda x: x.mode()[0]
}).reset_index()

X = aggregated_att.drop(columns=['package'])

# the target is package for classification
y = aggregated_att['package']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5805, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

def gridsearch_and_metrics(model_name, model, param_grid, X_train, y_train, X_test, y_test, class_labels=None):

    print(f"\nStarting Grid Search for {model_name}...")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=StratifiedKFold(n_splits=3), scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # best model using the in-built function best_estimator
    best_model = grid_search.best_estimator_

    # best parameters
    print(f"\nBest Parameters: {grid_search.best_params_}\n")

    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # train and test accuracies
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)

    # calculating specificity
    specificity = {}
    weighted_specificity = 0
    total_instances = len(y_test)
    for i, class_label in enumerate(class_labels):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        specificity[class_label] = tn/(tn+fp)
        weighted_specificity += specificity[class_label]*np.sum(y_test == class_label)/total_instances

    if hasattr(best_model, "predict_proba"):
        y_test_proba = best_model.predict_proba(X_test)
        y_test_multi = label_binarize(y_test, classes=np.unique(y_test))

        plt.figure(figsize=(10, 8))

        for i, class_label in enumerate(np.unique(y_test)):
            fpr, tpr, _ = roc_curve(y_test_multi[:, i], y_test_proba[:, i])
            plt.plot(fpr, tpr, label=f"Class {class_label} (AUC = {roc_auc_score(y_test_multi[:, i], y_test_proba[:, i]):.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f'Multiclass ROC Curve for {model_name}')
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.legend()
        plt.grid()
        plt.show()

        # auc score
        auc = roc_auc_score(y_test_multi, y_test_proba, multi_class='ovr')
    else:
        auc = 0

    prec = precision_score(y_test, y_test_pred, average='weighted')
    recall = recall_score(y_test, y_test_pred, average='weighted')
    f1 = f1_score(y_test, y_test_pred, average='weighted')

    print(f"Confusion Matrix:\n{cm}")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Specificity: {weighted_specificity:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC: {auc}")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix for {model_name}")
    plt.show()

    # making a dictionary so that I can append it in a table
    metrics = {
        'Model': model_name,
        'Train Accuracy': train_accuracy,
        'Test Accuracy': test_accuracy,
        'Precision': prec,
        'Recall': recall,
        'Specificity (Weighted)': weighted_specificity,
        'F1-score': f1,
        'AUC': auc
    }
    return metrics

metrics_list = []

#%% Pre-pruned Decision Tree

print("\n************************ PRE-PRUNED DECISION TREE CLASSIFIER ************************")

pre_pruned_dt = "Pre-pruned Decision Tree"
pre_pruned_clf = DecisionTreeClassifier(random_state=5805)

param_grid_pre_pruned = {
    'max_depth': [1, 2, 3, 4, 5],
    'min_samples_split': [20,30,40],
    'min_samples_leaf': [10,20,30],
    'criterion':['gini','entropy','log_loss'],
    'splitter':['best','random'],
    'max_features':['sqrt','log2']
}

metrics = gridsearch_and_metrics(pre_pruned_dt, pre_pruned_clf, param_grid_pre_pruned, X_train, y_train, X_test, y_test, class_labels=y.unique())

metrics_list.append(metrics)

#%% Post-pruned Decision Tree

print("\n************************ POST-PRUNED DECISION TREE CLASSIFIER ************************")

post_pruned_dt = "Post-pruned Decision Tree"

path = pre_pruned_clf.cost_complexity_pruning_path(X_train, y_train)
alphas = path['ccp_alphas']

accuracy_train, accuracy_test = [],[]
for a in alphas:
    post_pruned_clf = DecisionTreeClassifier(random_state=5805, ccp_alpha=a)
    post_pruned_clf.fit(X_train, y_train)
    y_train_pred_post = post_pruned_clf.predict(X_train)
    y_test_pred_post = post_pruned_clf.predict(X_test)
    accuracy_train.append(accuracy_score(y_train, y_train_pred_post))
    accuracy_test.append(accuracy_score(y_test, y_test_pred_post))

m = np.argmax(accuracy_test)
optimum_alpha = alphas[m]

best_post_pruned_clf = DecisionTreeClassifier(random_state=5805, ccp_alpha=optimum_alpha)
best_post_pruned_clf.fit(X_train, y_train)

metrics = gridsearch_and_metrics(post_pruned_dt, best_post_pruned_clf, param_grid_pre_pruned, X_train, y_train, X_test, y_test, class_labels=y.unique())

metrics_list.append(metrics)

#%% Logistic Regression

print("\n************************ LOGISTIC REGRESSION CLASSIFIER ************************")

lr = "Logistic Regression"

lr_clf = LogisticRegression(random_state=5805, max_iter=1000)
param_grid_log_reg = {
    'penalty': ['l2'],
    'C': [0.1, 1, 10]
}
metrics = gridsearch_and_metrics(lr, lr_clf, param_grid_log_reg, X_train, y_train, X_test, y_test, class_labels=y.unique())

metrics_list.append(metrics)

#%% KNN

print("\n************************ K-NEAREST NEIGHBORS CLASSIFIER ************************")

knn = "K-Nearest Neighbors"

knn_clf = KNeighborsClassifier()

param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree']
}

metrics = gridsearch_and_metrics(knn, knn_clf, param_grid_knn, X_train, y_train, X_test, y_test, class_labels=y.unique())

metrics_list.append(metrics)

print("\n************************ OPTIMUM K USING ELBOW METHOD ************************")


k_values = range(2, 11)
train_accuracies = []
test_accuracies = []

for k in k_values:
    print(f"\nChecking for k = {k}")

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    train_accuracy = knn.score(X_train, y_train)
    test_accuracy = knn.score(X_test, y_test)

    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

plt.figure(figsize=(10, 6))
plt.plot(k_values, train_accuracies, label='Train Accuracy', marker='o', color='blue')
plt.plot(k_values, test_accuracies, label='Test Accuracy', marker='o', color='orange')
plt.title('Elbow Method for Optimal k (KNN) Using Accuracy')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

optimal_k = k_values[np.argmax(test_accuracies)]
print(f"Optimal k for KNN (using Elbow Method with Accuracy): {optimal_k}")

#%% Support Vector Machine (SVM)

print("\n************************ SUPPORT VECTOR MACHINE CLASSIFIER ************************")

svm = "Support Vector Machine"

svm_clf = SVC(random_state=5805, probability=True)

param_grid_svm = {
    'kernel': ['linear', 'rbf', 'poly']
}
metrics = gridsearch_and_metrics(svm, svm_clf, param_grid_svm, X_train, y_train, X_test, y_test, class_labels=y.unique())

metrics_list.append(metrics)

#%% Naive Bayes

print("\n************************ NAIVES BAYES CLASSIFIER ************************")

nb = "Naive Bayes"

nb_clf = GaussianNB()

param_grid_nb = {
    'var_smoothing': [1e-9, 1e-8, 1e-7]
}
metrics = gridsearch_and_metrics(nb, nb_clf, param_grid_nb, X_train, y_train, X_test, y_test, class_labels=y.unique())

metrics_list.append(metrics)


#%% Neural Networks

print("\n************************ NEURAL NETWORKS CLASSIFIER ************************")

nn = "Neural Networks"

nn_clf = MLPClassifier(random_state=5805)

param_grid_mlp = {
    'hidden_layer_sizes': [(50,), (100,), (100, 50), (200,)],
    'activation': ['tanh', 'relu'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive']
}
metrics = gridsearch_and_metrics(nn, nn_clf, param_grid_mlp, X_train, y_train, X_test, y_test, class_labels=y.unique())

metrics_list.append(metrics)

#%% Display the metrics for all the classifiers

metrics_table = pd.DataFrame(metrics_list).set_index('Model')
print(metrics_table)

print("\n************************ END OF PHASE III ************************")