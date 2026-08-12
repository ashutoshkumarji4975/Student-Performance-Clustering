import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
history = []

# Dataset load
data = pd.read_csv("dataset/students.csv")

# Features
X = data[["Marks", "Attendance", "StudyHours"]]

# Model train
model = KMeans(n_clusters=3, random_state=42)
model.fit(X)
data["Cluster"] = model.labels_

st.sidebar.title("Student Performance App")

st.sidebar.write("""
### Features
✅ Student Performance Prediction
✅ K-Means Clustering
✅ Performance Graph
✅ Student Dataset
✅ Cluster Count
""")

st.title("🎓 Student Performance Clustering")
st.write("This Machine Learning application predicts student performance using the K-Means Clustering algorithm.")


marks = st.number_input("Enter Marks")
attendance = st.number_input("Enter Attendance")
study_hours = st.number_input("Enter Study Hours")

if st.button("Predict Cluster"):
    prediction = model.predict([[marks, attendance, study_hours]])
    cluster = prediction[0]
    if cluster == 0:
        st.success("🟢 Excellent Student")
    elif cluster == 1:
        st.warning("🟡 Average Student")
    else:
        st.error("🔴 Needs Improvement")
    history.append({
        "Mark": marks,
        "Attendance": attendance,
        "StudyHours": study_hours,
        "Cluster": cluster
    })
        
st.subheader("Student Performance Graph")

fig, ax = plt.subplots()
ax.scatter(data["Marks"], data["Attendance"],
           c=model.labels_, cmap="viridis")
ax.set_xlabel("Marks")
ax.set_ylabel("Attendance")
ax.set_title("Student Performance Clustering")

for i, name in enumerate(data["Name"]):
    ax.text(
        data["Marks"][i],
        data["Attendance"][i],
        name,
        fontsize=9
    )

st.pyplot(fig)
st.subheader("Student Dataset")
st.dataframe(data)

st.subheader("Cluster Count")
st.bar_chart(data["Cluster"].value_counts())

st.markdown("---")
st.caption("Developed by Ashutosh Kumar Bharti B.Tech CSE (AI & ML)| Student Performance Clustering using Python, Streamlit & K-Means")