import os
import pandas as pd

print("Current Folder:", os.getcwd())
print("Dataset Exists:", os.path.exists("dataset/students.csv"))

data = pd.read_csv("dataset/students.csv")
print(data)

from sklearn.cluster import KMeans

# Sirf numerical columns lena
X = data[["Marks", "Attendance", "StudyHours"]]

# KMeans Model
kmeans = KMeans(n_clusters=3, random_state=42)

# Model train karna
data["Cluster"] = kmeans.fit_predict(X)

# Result print karna
print(data)

import matplotlib.pyplot as plt

plt.scatter(data["Marks"], data["Attendance"], c=data["Cluster"], cmap="viridis")

plt.xlabel("Marks")
plt.ylabel("Attendance")
plt.title("Student Performance Clustering")

plt.show()