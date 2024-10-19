
# K-Complex Detection Using Machine Learning Models

This repository contains the code and resources for detecting K-Complexes in EEG data using machine learning models. K-Complexes are important waveforms in sleep studies and are useful for analyzing sleep patterns, particularly in sleep stage classification.
(Some files are not included for private license)

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Overview](#model-overview)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Introduction

K-Complexes are distinct waveforms seen in electroencephalogram (EEG) readings during sleep. Detecting them can be crucial for diagnosing and understanding various sleep disorders. This project aims to automatically detect K-Complexes using machine learning models.

## Project Structure

```bash
K-Complex-Detection-Using-Machine-Learning-Models/
│
├── data/                # EEG data for training and testing the models
├── models/              # Trained models and model-related scripts
├── notebooks/           # Jupyter notebooks for exploration and visualization
├── scripts/             # Python scripts for data preprocessing and model training
├── README.md            # Project documentation (this file)
└── requirements.txt     # Python dependencies
```

## Requirements

The project is built with Python and the required libraries are listed in \`requirements.txt\`. The key dependencies include:

- NumPy
- Pandas
- Scikit-learn
- TensorFlow or PyTorch (depending on the model used)
- Matplotlib (for visualization)
- MNE (for EEG data manipulation)

## Installation

To set up the environment for this project, follow these steps:

1. Clone this repository:

   ```bash
   git clone https://github.com/Francesco-Greca/K-Complex-Detection-Using-Machine-Learning-Models.git
   ```

2. Navigate to the project directory:

   ```bash
   cd K-Complex-Detection-Using-Machine-Learning-Models
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

To use the project, you can follow these steps:

1. **Preprocess the Data**: Run the preprocessing script to clean and prepare the EEG data for training.
   
   ```bash
   python scripts/preprocess_data.py
   ```

2. **Train a Model**: Choose a machine learning model and start training on the prepared dataset.
   
   ```bash
   python scripts/train_model.py --model <model_name>
   ```

3. **Evaluate the Model**: After training, you can evaluate the performance using:

   ```bash
   python scripts/evaluate_model.py --model <model_name>
   ```

## Model Overview

The project leverages several machine learning models for K-Complex detection, such as:

- **Random Forest**
- **Support Vector Machines (SVM)**
- **Convolutional Neural Networks (CNNs)**

Each model has its advantages depending on the dataset and requirements, and you can experiment with different models to optimize performance.

## Evaluation

To assess the performance of the models, metrics such as accuracy, precision, recall, and F1 score are used. These metrics provide insight into the model's ability to correctly identify K-Complexes in EEG data.

## Contributing

Contributions are welcome! If you'd like to improve this project, please:

1. Fork the repository.
2. Create a new branch (\`git checkout -b feature-branch\`).
3. Commit your changes (\`git commit -m 'Add new feature'\`).
4. Push to the branch (\`git push origin feature-branch\`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
