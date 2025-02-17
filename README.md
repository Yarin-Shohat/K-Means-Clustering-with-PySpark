# K-Means Clustering with PySpark

## Overview
This project implements the **K-Means clustering algorithm** using **Apache Spark** in **Python**. The algorithm groups data into **K clusters** based on feature similarity.

## Features
- Implementation of K-Means without using built-in ML libraries.
- Uses **PySpark** for distributed computing.
- **MinMaxScaler** from `scikit-learn` for data normalization.
- Supports **custom initial centroids** or random sampling.
- Uses **convergence threshold** and **iteration limits** as stopping criteria.

## Technologies Used
- **Python**
- **Apache Spark**
- **PySpark**
- **scikit-learn**
- **Databricks**

## Installation
### 1. Setup Databricks
- Create an account at [Databricks](https://community.cloud.databricks.com/login.html).
- Set up a **compute cluster** in Databricks.
- Upload your dataset (`Iris.csv`) to `/FileStore/tables/`.

### 2. Install Required Libraries
Ensure you have the necessary Python libraries installed:
```sh
pip install pyspark scikit-learn pandas
```

## Usage
### 1. Import Dependencies
```python
from pyspark.sql import SparkSession
from sklearn.preprocessing import MinMaxScaler
import random
```

### 2. Define K-Means Function
```python
def Kmeans(data, k, ct=0.0001, num_iterations=30, initial_centroids=None):
    """
    Runs the K-Means clustering algorithm on a given dataset.
    
    Parameters:
    - data (PySpark DataFrame): Dataset without target variable.
    - k (int): Number of clusters.
    - ct (float): Convergence threshold (default=0.0001).
    - num_iterations (int): Max iterations (default=30).
    - initial_centroids (list): List of tuples as initial centroids (default=None).
    
    Returns:
    - List of final centroid coordinates.
    """
    pass  # Implementation goes here
```

### 3. Load Data and Run K-Means
```python
spark = SparkSession.builder.appName("KMeans").getOrCreate()
data = spark.read.csv("/FileStore/tables/Iris.csv", header=True, inferSchema=True)

# Run K-Means
centroids = Kmeans(data, k=3)
print("Final centroids:", centroids)
```
