#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 10:34:48 2023

@author: greca.francesco@udc.es
"""

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import numpy as np
import mne
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
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
            try:
                start_time, duration = map(float, line.strip().split())
                end_time = start_time + duration
                kcomplex_info.append((start_time, end_time))
            except ValueError:
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

        labels.append(1 if kcomplex_present else 0)
        raw_temp = raw.copy().crop(tmin=start_sec, tmax=end_sec)
        eeg_fragments.append(raw_temp.get_data()[0])

    # Add the EEG data of this file to the overall list
    eeg_data.extend(eeg_fragments)
    kcomplex_labels.extend(labels)

# After collecting the data and labels, convert them to numpy arrays for the machine learning process
eeg_data = np.array(eeg_data)
kcomplex_labels = np.array(kcomplex_labels)

# Standardize the input data
scaler = StandardScaler()
eeg_data = scaler.fit_transform(eeg_data)

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(eeg_data, kcomplex_labels)

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Create the SVM Classifier
svm_clf = SVC(kernel='rbf', C = 55.0)  # You can change the kernel and other parameters as needed

# Train the model using the training sets
svm_clf.fit(X_train, y_train)

# Predict the response for the test dataset
y_pred = svm_clf.predict(X_test)

# Model Accuracy, Precision, Recall, F1-Score and Confusion Matrix
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))
print("Confusion Matrix \n", confusion_matrix(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
# Plot Confusion Matrix
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


import joblib
joblib.dump(svm_clf, 'Kcomples_model_SVM.joblib')
print('Best model saved as kcomplex_model_SVM.joblib')