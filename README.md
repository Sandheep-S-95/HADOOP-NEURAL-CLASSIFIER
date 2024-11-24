# 💡HADOOP-NEURAL-CLASSIFIER 
![Neural Network MapReduce](https://github.com/Sandheep-S-95)

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/downloads/) [![MRJob](https://img.shields.io/badge/MRJob-0.7.4-green.svg)](https://mrjob.readthedocs.io/) [![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-orange.svg)](https://numpy.org/) [![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A distributed implementation of an Artificial Neural Network using Hadoop MapReduce and Python's MRJob library. This project demonstrates how to train a neural network in a distributed fashion using the MapReduce programming model.

## 📋 Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Notebook Usage](#notebook-usage)
- [Configuration](#configuration)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features
- Distributed training using MapReduce
- Configurable neural network architecture
- Support for multi-class classification
- Customizable hyperparameters
- L2 regularization for better generalization
- Real-time accuracy metrics

## 🔧 Prerequisites
```
python>=3.7
numpy
mrjob
jupyter
google-colab (for running in Colab)
```

## 🚀 Installation
```bash
git clone https://github.com/yourusername/hadoop-neural-classifier.git
cd hadoop-neural-classifier
```

## 📘 Notebook Usage
1. Local Jupyter:
```bash
jupyter notebook ANN_USING_MAPREDUCE.ipynb
```

2. Google Colab:
- Upload `ANN_USING_MAPREDUCE.ipynb` to Google Drive
- Open with Google Colab
- Run the following in first cell:
```python
!pip install mrjob
```
- Execute cells sequentially

3. Dataset Preparation:
```python
# In notebook
from sklearn.datasets import load_iris
iris = load_iris()
# Follow data preprocessing steps in notebook
```

## ⚙️ Configuration
```python
# Available parameters in notebook
params = {
    'learning_rate': 0.1,
    'num_iterations': 100,
    'hidden_layers': '5,5',
    'l2_regularization': 0.1
}
```

## 📊 Results
```json
{
    "Class 0": {
        "samples_processed": 50,
        "correct_predictions": 48,
        "accuracy": 0.96,
        "average_probabilities": [0.95, 0.03, 0.02]
    }
}
```


## 📄 License
MIT License

## 📫 Contact
Project Link: [https://github.com/Sandheep-S-95/HADOOP-NEURAL-CLASSIFIER](https://github.com/Sandheep-S-95)

## Repository Structure
```
HADOOP-NEURAL-CLASSIFIER/
│
├── ANN_USING_MAPREDUCE.ipynb    # Main Jupyter notebook
├── README.md                    # Project documentation
├── LICENSE                      # MIT License
```
