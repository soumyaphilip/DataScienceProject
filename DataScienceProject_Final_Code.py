#!/usr/bin/env python
# coding: utf-8

# In[157]:


pip install imbalanced-learn


# In[212]:


pip install xgboost


# In[1]:


#Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from scipy import stats
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import xgboost as xgb  # XGBoost library
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier
from datetime import datetime
from sklearn.multiclass import OneVsRestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score


# In[2]:


#Load the csv dataset
landslides = pd.read_csv("/Users/soumya/Desktop/jupytersou/Global_Landslide_Catalog_Export.csv")


# In[3]:


print(landslides.head())


# In[4]:


landslides.info()


# In[5]:


landslides.describe()


# In[6]:


print(len(landslides))
print(len(landslides.columns))
print(landslides.columns)


# ## EDA

# In[7]:


# Summary statistics for numeric columns
print(landslides.describe())


# In[8]:


# Value counts for a categorical column (e.g., 'landslide_category')
print(landslides['landslide_category'].value_counts())


# In[9]:


# Histograms for numeric columns
landslides.hist(figsize=(12, 8))
plt.show()


# In[10]:


# Box plot for a numeric column (e.g., 'fatality_count')
sns.boxplot(data=landslides, x='fatality_count')
plt.show()


# In[11]:


# Correlation heatmap
correlation_matrix = landslides.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()


# In[12]:


# Convert event_date to datetime
landslides['event_date'] = pd.to_datetime(landslides['event_date'])

# Create function to determine season from month  
def get_season(month):
    if month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    elif month in [9, 10, 11]:
        return 'autumn'
    else:
        return 'winter'

# Extract month from event_date and apply season function
landslides['event_month'] = landslides['event_date'].dt.month  
landslides['event_season'] = landslides['event_month'].apply(get_season)
# Print seasons    
print(landslides['event_season'].value_counts())

print(landslides)


# ## Drop Unwanted Columns

# In[13]:


# List of columns to remove
columns_to_remove = ['event_date', 'event_month','source_name', 'source_link', 'event_id', 'event_description', 'storm_name',
                     'photo_link', 'notes', 'event_import_source', 'event_import_id', 'submitted_date', 'created_date',
                     'last_edited_date', 'location_description', 'admin_division_name','admin_division_population', 'country_code', 'country_name', 'event_time' ]

# Remove the unwanted columns
landslides_drop = landslides.drop(columns=columns_to_remove)

# Save the cleaned dataset to a new CSV file
landslides_drop.to_csv("Global_Landslide_Catalog_Export_column_drop.csv", index=False)
landslides = pd.read_csv("/Users/soumya/Desktop/jupytersou/Global_Landslide_Catalog_Export_column_drop.csv")
print(landslides.head(5))


# In[14]:


print(len(landslides))
print(len(landslides.columns))
print(landslides.columns)


# ## Preprocessing Steps

# In[15]:


# Check for missing values
missing_values = landslides.isnull().sum()
missing_values.head()
print(missing_values)


# In[16]:


# Multiple Imputation Technique - SimpleImputer

# Impute missing values 
imputer = SimpleImputer(strategy='most_frequent')
imputed_data = imputer.fit_transform(landslides)

# Columns with missing values
columns_with_missing_values = ['injury_count', 'location_accuracy', 'landslide_category', 'landslide_trigger', 'landslide_size', 'landslide_setting', 'fatality_count','gazeteer_closest_point', 'gazeteer_distance']

# Apply the imputer to fill missing values in specific columns
landslides[columns_with_missing_values] = imputer.fit_transform(landslides[columns_with_missing_values])


# In[17]:


# Create a new DataFrame with the imputed data
imputed_landslides = pd.DataFrame(imputed_data, columns=list(landslides.columns))
print(imputed_landslides.isnull().sum())


# In[18]:


# Create a dictionary to map the old landslide_category values to the new values
category_map = {
    "rock fall": "Falls",
    "snow_avalanche": "Falls",
    "topple": "Topples",
    "landslide": "Slides",
    "mudslide": "Slides",
    "translational slides": "Slides",
    "creep": "Slides",
    "debris flows": "Flows",
    "lahars": "Flows",
    "earth_flow": "Flows",
    "complex": "other",
    "other": "other",
    "unknown": "other",
    "riverbank_collapse": "other"
}

# Modify the landslide_category column using the category_map
imputed_landslides['landslide_category'] = imputed_landslides['landslide_category'].map(category_map)

# Save the modified DataFrame
imputed_landslides.to_csv("modified_landslide_category.csv", index=False)


# In[19]:


# Group the DataFrame by the landslide_category column and count the occurrences of each category
category_counts = imputed_landslides.groupby('landslide_category').size()


# In[20]:


# Check for duplicate rows
duplicate_rows = imputed_landslides[imputed_landslides.duplicated()]
# Display duplicate rows, if any
if not duplicate_rows.empty:
    print("Duplicate Rows found")
else:
    print("No Duplicate Rows found.")


# In[14]:


# Drop duplicate rows and keep the first occurrence
imputed_landslides = imputed_landslides.drop_duplicates()
# Reset the index if needed
imputed_landslides = imputed_landslides.reset_index(drop=True)
print(imputed_landslides.shape)
print(len(imputed_landslides.columns))
print(len(imputed_landslides))


# In[21]:


print(imputed_landslides.columns)


# ## Feature Engineering

# In[22]:


# Feature 1: Encode categorical features using LabelEncoder


# Encode categorical features using LabelEncoder
categorical_columns = imputed_landslides.select_dtypes(include=['object']).columns
label_encoder = LabelEncoder()

# Encode each categorical column individually
for col in categorical_columns:
    imputed_landslides[col] = label_encoder.fit_transform(imputed_landslides[col])
    
imputed_landslides.drop(imputed_landslides[imputed_landslides['landslide_category'] == 5].index, inplace=True)

# Print the encoded landslide_category values
print(imputed_landslides['landslide_category'].value_counts())


# In[24]:


# Feature 2: Create a binary feature indicating whether there were fatalities
imputed_landslides['has_fatalities'] = imputed_landslides['fatality_count'].apply(lambda x: 1 if x > 0 else 0)


# In[26]:


# Feature Selection

# Separate features and target variable
X = imputed_landslides.drop(columns=['landslide_category'])  # Features
y = imputed_landslides['landslide_category']  # Target variable
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use SelectKBest to select the top k features based on ANOVA F-statistics
k = 8  # Number of features to select
selector = SelectKBest(score_func=f_classif, k=k)
X_new = selector.fit_transform(X_train, y_train)
# Get the selected feature names
selected_feature_names = X.columns[selector.get_support()]
print(selected_feature_names)


# In[27]:


# REMOVE OUTLIERS

# Define a function to remove outliers using Z-Score
def remove_outliers_zscore(imputed_landslides, threshold=3):
    z_scores = np.abs(stats.zscore(imputed_landslides))
    imputed_landslides_no_outliers = imputed_landslides[(z_scores < threshold).all(axis=1)]
    return imputed_landslides_no_outliers

# Specify the columns where we want to detect and remove outliers
numeric_features = ['landslide_size', 'fatality_count', 'injury_count', 'gazeteer_distance', 'longitude', 'latitude']
#Remove outliers from the specified columns
imputed_landslides_no_outliers = remove_outliers_zscore(imputed_landslides[numeric_features])

# Merge the cleaned numeric features back with the original dataset
imputed_landslides_cleaned = pd.concat([imputed_landslides.drop(numeric_features, axis=1), imputed_landslides_no_outliers], axis=1)


# In[28]:


# Check target variable is balanced or not

# Count the occurrences of each category in the target variable
category_counts = imputed_landslides_cleaned['landslide_category'].value_counts()
# Plot the class distribution
plt.figure(figsize=(8, 6))
plt.bar(category_counts.index, category_counts.values)
plt.xlabel('Landslide Category')
plt.ylabel('Count')
plt.title('Imabalanced Landslide Category Distribution')
plt.xticks(category_counts.index)
plt.show()

# Count and print each landslide category along with the count
category_counts = imputed_landslides_cleaned['landslide_category'].value_counts()
for category, count in category_counts.items():
    print(f"Landslide Category {category}: Count {count}")


# In[30]:


# Define features (X) and target variable (y)
X = imputed_landslides_cleaned.drop('landslide_category', axis=1) 
y = imputed_landslides_cleaned['landslide_category']
# Initialize RandomOverSampler
ros = RandomOverSampler(random_state=42)
# Apply RandomOverSampler to balance the dataset
X_resampled, y_resampled = ros.fit_resample(X, y)
# Create a DataFrame for the balanced data
balanced_data = pd.concat([X_resampled, y_resampled], axis=1)
# Create a bar plot to visualize the distribution of categories in the balanced data
plt.figure(figsize=(8, 6))
balanced_data['landslide_category'].value_counts().plot(kind='bar')
plt.xlabel('Landslide Category')
plt.ylabel('Count')
plt.title('Balanced Landslide Category Distribution')
plt.show()


# ## Data Splitting and Training

# In[33]:


# Split the balanced dataset into features (X) and target variable (y)
X = balanced_data.drop('landslide_category', axis=1)  
y = balanced_data['landslide_category']
# Split the data into training (70%), validation (15%), and test (15%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

X_train = imputer.fit_transform(X_train)
X_val = imputer.transform(X_val)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

columns_to_fill = ['landslide_size', 'fatality_count', 'injury_count', 'gazeteer_distance', 'longitude', 'latitude']

# Filling NaN values in numeric columns with mean values
balanced_data[columns_to_fill] = balanced_data[columns_to_fill].fillna(balanced_data[columns_to_fill].mean())
# Handle missing values by imputing with mean (you can use other strategies)
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)


# ## Modelling

# In[34]:


# Dictionary to hold different classifiers
classifiers = {
    'Random Forest': OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42)),
    'Decision Tree': OneVsRestClassifier(DecisionTreeClassifier(random_state=42)),
    'Logistic Regression': OneVsRestClassifier(LogisticRegression(random_state=42, max_iter=1000)),
    'Gradient Boosting': OneVsRestClassifier(GradientBoostingClassifier(random_state=42)),
    'Support Vector Machine': OneVsRestClassifier(SVC(random_state=42, probability=True)),
    'XGBoost': OneVsRestClassifier(XGBClassifier(random_state=42))
}

# Lists to store results
results = []
metrics = ['Classifier', 'Accuracy', 'F1 Score', 'ROC AUC Score', 'Precision', 'Recall']

# Train, predict, and evaluate each classifier
for name, clf in classifiers.items():
    # Train the classifier
    clf.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = clf.predict(X_test)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(label_binarize(y_test, classes=y.unique()), label_binarize(y_pred, classes=y.unique()), average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    
    # Append results to the list
    results.append([name, accuracy, f1, roc_auc, precision, recall])
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    # Print the evaluation metrics for each classifier
    print(f"Classifier: {name}")
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"ROC AUC Score: {roc_auc}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print("--------------------------------------")
    
# Create a pandas DataFrame for results
results_df = pd.DataFrame(results, columns=metrics)

# Display results table
print(results_df)


# In[35]:


# Dictionary to hold different classifiers
classifiers = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Support Vector Machine': SVC(random_state=42, probability=True),
    'XGBoost': XGBClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

# Create a figure and axes to plot ROC curves
plt.figure(figsize=(8, 6))

# Plot ROC curve for each classifier separately
for name, clf in classifiers.items():
    # Train the classifier
    clf.fit(X_train, y_train)
    
    # Predict probabilities on the test set (for computing ROC curve)
    y_probs = clf.predict_proba(X_test)
    
    # Get the index for class 2
    class_2_index = np.where(clf.classes_ == 2)[0][0]
    
    # Calculate false positive rate and true positive rate for class 2
    fpr, tpr, _ = roc_curve(label_binarize(y_test, classes=clf.classes_)[:, class_2_index], y_probs[:, class_2_index])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')

# Plot the diagonal line (random classifier)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# In[36]:


# Handle missing values by imputing with mean (you can choose another strategy)
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Perform cross-validation for each classifier
cv_results = {}
for name, clf in classifiers.items():
    # Perform cross-validation with 5 folds
    scores = cross_val_score(clf, X_imputed, y, cv=5, scoring='accuracy')
    cv_results[name] = scores
    
    # Print cross-validation results for each classifier
    print(f"{name} Cross-validation results:")
    print(scores)
    print(f"Mean Accuracy: {np.mean(scores)}")
    print("--------------------------------------")

# Plotting cross-validation results (same as before)
plt.figure(figsize=(10, 6))
plt.boxplot(cv_results.values())
plt.xticks(range(1, len(classifiers) + 1), cv_results.keys(), rotation=45)
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Cross-validation scores for different models')
plt.show()


# In[37]:


# Handle missing values by imputing with mean (you can choose another strategy)
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Set the number of folds for cross-validation
num_folds = 5

# Initialize a figure for plotting
plt.figure(figsize=(10, 6))

# Perform cross-validation and plot results for each classifier separately
for name, clf in classifiers.items():
    # Perform cross-validation with 5 folds
    scores = cross_val_score(clf, X_imputed, y, cv=num_folds, scoring='accuracy')
    
    # Plot cross-validation results for each fold
    plt.plot(range(1, num_folds + 1), scores, marker='o', label=name)

# Plot settings and labels
plt.title('Cross-validation scores for Different Models')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.ylim(0.5, 1.0)  # Set y-axis limit for better visualization
plt.xticks(range(1, num_folds + 1))
plt.legend()
plt.grid(True)
plt.show()


# In[38]:


# Handle missing values by imputing with mean 
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Set the number of folds for cross-validation
num_folds = 5

# Perform cross-validation and plot results for each classifier separately
for name, clf in classifiers.items():
    # Initialize a figure for each classifier
    plt.figure(figsize=(8, 6))
    
    # Perform cross-validation with 5 folds
    scores = cross_val_score(clf, X_imputed, y, cv=num_folds, scoring='accuracy')
    
    # Plot cross-validation results for each fold
    plt.plot(range(1, num_folds + 1), scores, marker='o')
    
    # Plot settings and labels
    plt.title(f'Cross-validation scores for {name}')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.ylim(0.5, 1.0)  # Set y-axis limit for better visualization
    plt.xticks(range(1, num_folds + 1))
    plt.grid(True)
    plt.show()

