#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 15:57:46 2023

@author: greca.francesco@udc.es
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import numpy as np
import mne
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
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

# Create a non-convolutional deep learning model
model = Sequential([
    Dense(128, activation='relu', input_shape=(eeg_data.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=30, batch_size=64, validation_split=0.2)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print("Test accuracy:", accuracy)

# Plot the training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Training and Validation Loss', fontsize=16)
plt.legend()
plt.show()

# Get the model predictions on the test dataset
y_pred_prob = model.predict(X_test)
y_pred_binary = (y_pred_prob > 0.5).astype(int)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary, zero_division=1)
recall = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)
confusion_matrix = confusion_matrix(y_test, y_pred_binary)
# For binary classification problems, calculate the Area Under the Receiver Operating Characteristics
roc_auc = roc_auc_score(y_test, y_pred_prob)

# Print all the metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Confusion Matrix:\n", confusion_matrix)
print("AUC-ROC:", roc_auc)

# Plot the Confusion Matrix
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(confusion_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(confusion_matrix.shape[0]):
    for j in range(confusion_matrix.shape[1]):
        ax.text(x=j, y=i, s=confusion_matrix[i, j], va='center', ha='center', size='xx-large')
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()



# Explore model predictions and actual labels
# for i in range(len(y_pred_binary)):
#     print("Example", i)
#     print("Real Label:", y_test[i])
#     print("Model Prediction:", y_pred_binary[i])

# Explore the distribution of K-Complex labels in the test dataset
# print("Distribution of K-Complex labels in the test dataset:")
# print(pd.Series(y_test).value_counts())

# Visualize some misclassifications
errors = np.where(y_pred_binary.squeeze() != y_test)[0]
for i in errors[:min(5, len(errors))]:
    plt.figure()
    plt.plot(X_test[i])
    plt.title(
        f"Error for example {i}. Real Label: {y_test[i]}, Prediction: {y_pred_binary[i]}")

# Creare la griglia dei parametri
param_grid = {
    'filters': [16, 32, 64],
    'kernel_size': [3, 5, 7],
    'activation': ['relu', 'tanh'],
}

# Inizializza le migliori impostazioni
best_params = None
best_score = 0.0

def build_model(filters, kernel_size, activation):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),  # Adjust the input shape as needed
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Esegui la ricerca manuale dei parametri
for filters in param_grid['filters']:
    for kernel_size in param_grid['kernel_size']:
        for activation in param_grid['activation']:
            print(f"Testing params: filters={filters}, kernel_size={kernel_size}, activation={activation}")
            
            # Valutazione del modello con cross-validation
            kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            cv_scores = []
            for train, test in kfold.split(X_train, y_train):
                model = build_model(filters, kernel_size, activation)  # Rebuild the model with new parameters
                model.fit(X_train[train].reshape(X_train[train].shape[0], X_train[train].shape[1], 1), y_train[train], epochs=10, batch_size=34, verbose=0)
                y_pred = (model.predict(X_train[test].reshape(X_train[test].shape[0], X_train[test].shape[1], 1)) > 0.5).astype(int)
                score = accuracy_score(y_train[test], y_pred)
                cv_scores.append(score)
            
            # Calcola il punteggio medio
            score_mean = np.mean(cv_scores)
            
            print(f"Score: {score_mean}")
            
            # Aggiorna le migliori impostazioni se necessario
            if score_mean > best_score:
                best_score = score_mean
                best_params = (filters, kernel_size, activation)

# Stampa i migliori parametri e il punteggio
print("Best Parameters:", best_params)
print("Best Score:", best_score)


# Save the best model
model.save('kcomplex_model_FCNN.h5')
print('Best model saved as kcomplex_model_FCNN.h5')
