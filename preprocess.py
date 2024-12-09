import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
train_data_path = 'train_data.csv'
train_data = pd.read_csv(train_data_path)

# Extract features (landmark coordinates) and labels (class_name)
features = train_data.filter(regex='(x|y)$').values  # Use only *_x, *_y columns
labels = train_data['class_name'].values

# Normalize features
features = features / features.max(axis=0)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save the model
joblib.dump(clf, 'pose_classifier.pkl')
print("Model trained and saved.")
