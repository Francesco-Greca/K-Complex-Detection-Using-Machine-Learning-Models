##!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 15:57:46 2023

@author: greca.francesco@udc.es
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import os
import numpy as np
import mne
import matplotlib.pyplot as plt
from itertools import product
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# Directory contenente i file EEG ed i file di annotazione
eeg_directory = 'DatabaseKcomplexes'
annotation_directory = 'DatabaseKcomplexes'

# Lista per salvare i dati EEG e le relative etichette
eeg_data = []
kcomplex_labels = []

# Elabora tutti i file nella directory
for i in range(1, 11):  # Itera da 1 a 10 per coprire i file con numeri diversi
    eeg_filename = f"excerpt{i}.edf"
    txt_filename = f"Visual_scoring1_excerpt{i}.txt"
    
    eeg_file_path = os.path.join(eeg_directory, eeg_filename)
    annotation_file_path = os.path.join(annotation_directory, txt_filename)

    # Leggi il segnale EEG
    raw = mne.io.read_raw_edf(eeg_file_path, preload=True)
    raw = raw.pick_channels(['CZ-A1'])

    # Leggi il file di annotazione
    kcomplex_info = []
    with open(annotation_file_path, 'r') as annotation_file:
        for line in annotation_file:
            # Prova a convertire la riga in due float (timestamp di inizio e durata)
            try:
                start_time, duration = map(float, line.strip().split())
                end_time = start_time + duration
                kcomplex_info.append((start_time, end_time))
            except ValueError:
                # Ignora le righe che non possono essere convertite
                pass

    # Specifica il passo (in secondi) per avanzare durante il ciclo
    step_sec = 0.5

    # Inizializza una lista per raccogliere i frammenti dei segnali EEG e le relative etichette
    eeg_fragments = []
    labels = []

    # Crea un ciclo che avanza di 5 secondi alla volta
    for start_sec in np.arange(0, raw.times[-1], step_sec):
        end_sec = start_sec + step_sec
        if end_sec > raw.times[-1]:
            break

        # Verifica se c'è almeno un K-Complex che sovrappone il frammento attuale
        kcomplex_present = any(start_sec <= tempo <= end_sec or (
            start_sec <= tempo + durata <= end_sec) for tempo, durata in kcomplex_info)

        # Etichetta il frammento come "K-Complex" (1) o "non-K-Complex" (0)
        labels.append(1 if kcomplex_present else 0)
        raw_temp = raw.copy().crop(tmin=start_sec, tmax=end_sec)

        eeg_fragments.append(raw_temp.get_data()[0])

        # Aggiungi i dati EEG di questo file alla lista complessiva
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


# Define the CNN model
model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu',
           input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, epochs=20, batch_size=34, validation_split=0.2)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test.reshape(X_test.shape[0], X_test.shape[1], 1), y_test)
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
y_pred_prob = model.predict(X_test.reshape(X_test.shape[0], X_test.shape[1], 1))
y_pred_binary = (y_pred_prob > 0.5).astype(int)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary, zero_division=1)
recall = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)
conf_matrix = confusion_matrix(y_test, y_pred_binary)
# For binary classification problems, calculate the Area Under the Receiver Operating Characteristics
roc_auc = roc_auc_score(y_test, y_pred_prob)

# Prints all the metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Confusion Matrix:\n", conf_matrix)
print("AUC-ROC:", roc_auc)

# Plot the Confusion Matrix
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
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

param_grid = {
    'filters': [16, 32, 64],
    'kernel_size': [3, 5, 7],
    'activation': ['relu', 'tanh'],
}


# Creare la griglia dei parametri
param_grid_values = [param_grid['filters'], param_grid['kernel_size'], param_grid['activation']]
param_combinations = list(product(*param_grid_values))

# Inizializza le migliori impostazioni
best_params = None
best_score = 0.0

# Esegui la ricerca manuale dei parametri
for params in param_combinations:
    filters, kernel_size, activation = params
    print(f"Testing params: filters={filters}, kernel_size={kernel_size}, activation={activation}")
    
    # Valutazione del modello con cross-validation
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = []
    for train, test in kfold.split(X_train, y_train):
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
        best_params = params

# Stampa i migliori parametri e il punteggio
print("Best Parameters:", best_params)
print("Best Score:", best_score)

# Save the model
model.save('kcomplex_model_CNN.h5')
print('Model saved as kcomplex_model_CNN.h5')

