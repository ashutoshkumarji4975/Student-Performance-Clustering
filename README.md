# Student Performance Clustering

This is my major project based on Student Performance Clustering.

In this project, I have used Machine Learning to group students according to their performance. The system takes marks, attendance and study hours as input and predicts which performance group the student belongs to.

## Features

- Student Performance Prediction
- K-Means Clustering
- Student Dataset
- Performance Graph
- Cluster Count
- Simple Streamlit Interface

## Technologies Used

- Python
- Pandas
- Scikit-learn
- K-Means Clustering
- Matplotlib
- Streamlit

## Input

The system takes three main inputs:

- Marks
- Attendance
- Study Hours

## Working

First, the student dataset is loaded using Pandas.

After that, Marks, Attendance and Study Hours are selected as features.

K-Means Clustering is then applied with 3 clusters. Based on the cluster, students are shown as:

- Excellent Student
- Average Student
- Needs Improvement

The project also shows a graph to understand student performance.

## Project Structure

```text
Student-Performance-Clustering/
│
├── dataset/
│   └── students.csv
│
├── app.py
├── train_model.py
└── README.md
