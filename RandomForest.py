#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 10:34:48 2023

@author: greca.francesco@udc.es
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import numpy as np
import mne
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import matplotlib.pyplot as plt

# Directory containing EEG and annotation files
eeg_directory = 'DatabaseKcomplexes'
annotation_directory = 'DatabaseKcomplexes'

# Lists to store EEG data and their labels
eeg_data = []
kcomplex_labels = []

# Iterate through all the files in the directory
for i in range(1, 11):  # Iterate from 1 to 10 to cover files with different numbers
    eeg_filename = f"excerpt{i}.edf"
    txt_filename = f"Visual_scoring1_excerpt{i}.txt"

    eeg_file_path = os.path.join(eeg_directory, eeg_filename)
    annotation_file_path = os.path.join(annotation_directory, txt_filename)

    # Read the EEG signal
    raw = mne.io.read_raw_edf(eeg_file_path, preload=True)
    raw = raw.pick_channels(['CZ-A1'])

    # Read the annotation file
    kcomplex_info = []
    with open(annotation_file_path, 'r') as annotation_file:
        for line in annotation_file:
            # Try to convert the line into two floats (start time and duration)
            try:
                start_time, duration = map(float, line.strip().split())
                end_time = start_time + duration
                kcomplex_info.append((start_time, end_time))
            except ValueError:
                # Ignore lines that cannot be converted
                pass

    # Specify the step (in seconds) to advance during the loop
    step_sec = 0.5

    # Initialize lists to collect EEG signal fragments and their labels
    eeg_fragments = []
    labels = []

    # Create a loop that advances by 0.5 seconds at a time
    for start_sec in np.arange(0, raw.times[-1], step_sec):
        end_sec = start_sec + step_sec
        if end_sec > raw.times[-1]:
            break

        # Check if there is at least one K-Complex overlapping with the current fragment
        kcomplex_present = any(start_sec <= tempo <= end_sec or (
            start_sec <= tempo + durata <= end_sec) for tempo, durata in kcomplex_info)

        # Label the fragment as "K-Complex" (1) or "non-K-Complex" (0)
        labels.append(1 if kcomplex_present else 0)
        raw_temp = raw.copy().crop(tmin=start_sec, tmax=end_sec)

        eeg_fragments.append(raw_temp.get_data()[0])

    # Add the EEG data of this file to the overall list
    eeg_data.extend(eeg_fragments)
    kcomplex_labels.extend(labels)

# After collecting the data and labels, you can remove them from the original lists
eeg_data = np.array(eeg_data)
kcomplex_labels = np.array(kcomplex_labels)

# Standardize the input data (important for non-convolutional models)
scaler = StandardScaler()
eeg_data = scaler.fit_transform(eeg_data)

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(eeg_data, kcomplex_labels)

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Create the Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)  # You can adjust the number of estimators as needed

# Train the model using the training sets
clf.fit(X_train, y_train)

# Predict the response for the test dataset
y_pred = clf.predict(X_test)

# Model Accuracy: how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Model Precision: what percentage of positive tuples are labeled as such
print("Precision:", metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labeled as such
print("Recall:", metrics.recall_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labeled as such
print("F1-Score:", metrics.f1_score(y_test, y_pred))

print("Confusion Matrix \n", metrics.confusion_matrix(y_test, y_pred))

confusion_mat = metrics.confusion_matrix(y_test, y_pred)

import seaborn as sns  # For better-looking confusion matrix plots

# Plot Confusion Matrix
def plot_confusion_matrix(confusion_matrix):
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Non-K-Complex", "K-Complex"], yticklabels=["Non-K-Complex", "K-Complex"])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


plot_confusion_matrix(confusion_mat)

# Create lists to store training and testing accuracies
train_accuracies = []
test_accuracies = []

# Vary the number of estimators (trees) in the Random Forest
estimators_range = range(1, 10)  # You can adjust this range

for n_estimators in estimators_range:
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    clf.fit(X_train, y_train)
    
    # Training accuracy
    train_accuracy = clf.score(X_train, y_train)
    train_accuracies.append(train_accuracy)
    
    # Testing accuracy
    test_accuracy = clf.score(X_test, y_test)
    test_accuracies.append(test_accuracy)

# Plot training and testing accuracies
plt.figure(figsize=(8, 6))
plt.plot(estimators_range, train_accuracies, label='Training Accuracy')
plt.plot(estimators_range, test_accuracies, label='Testing Accuracy')
plt.xlabel('Number of Estimators (Trees)')
plt.ylabel('Accuracy')
plt.title('Training and Testing Accuracy vs. Number of Estimators')
plt.legend()
plt.grid(True)
plt.show()

import joblib

joblib.dump(clf, 'Kcomples_model_RandomForest.joblib')
print('Best model saved as kcomplex_model_RandomForest.joblib')