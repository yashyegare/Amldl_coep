import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.datasets import make_blobs

# --- K-Means From Scratch Class ---
class KMeansScratch:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None

    def fit(self, X):
        # 1. Initialize centroids randomly
        idx = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[idx]

        for _ in range(self.max_iters):
            # 2. Assignment Step
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            # 3. Update Step
            new_centroids = np.array([X[labels == i].mean(axis=0) if len(X[labels == i]) > 0 
                                     else self.centroids[i] for i in range(self.k)])
            
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids
        
        return labels, self.centroids

    def get_inertia(self, X, labels, centroids):
        inertia = 0
        for i in range(self.k):
            points_in_cluster = X[labels == i]
            if len(points_in_cluster) > 0:
                inertia += np.sum((points_in_cluster - centroids[i])**2)
        return inertia

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Advanced ML: K-Means Lab", layout="wide")
st.title("📊 K-Means Clustering Assignment")
st.markdown("**COEP Tech M.Tech Advanced ML and DL Practical 1**")

# --- Sidebar Configuration ---
st.sidebar.header("📁 Data Configuration")
data_source = st.sidebar.radio("Select Data Source:", ("Synthetic Blobs", "Upload CSV"))

# Initialize X and k_value to avoid errors
X = None
k_value = 3

if data_source == "Synthetic Blobs":
    k_value = st.sidebar.slider("Select k (Number of Clusters)", 2, 8, 3)
    n_samples = st.sidebar.number_input("Number of Data Points", 100, 2000, 500)
    noise = st.sidebar.slider("Cluster Standard Deviation (Noise)", 0.5, 3.0, 1.0)
    # Generate Synthetic Data
    X, _ = make_blobs(n_samples=n_samples, centers=k_value, cluster_std=noise, random_state=42)

else:
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.write("Data Preview:")
        st.sidebar.dataframe(df.head(3))
        
        # Select Columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cols = st.sidebar.multiselect("Select 2 columns for 2D clustering", numeric_cols, default=numeric_cols[:2])
        
        if len(cols) == 2:
            X = df[cols].dropna().values
            k_value = st.sidebar.slider("Select k for CSV Data", 2, 10, 3)
        else:
            st.sidebar.warning("Please select exactly 2 numerical columns.")
            st.stop()
    else:
        st.info("👋 Awaiting CSV upload. Use the sidebar to upload your file.")
        st.stop()

# --- Execution Area ---
if st.sidebar.button('🚀 Run Analysis'):
    # Start Timer
    start_time = time.time()
    
    # Run the model
    model = KMeansScratch(k=k_value)
    final_labels, final_centers = model.fit(X)
    
    # End Timer
    execution_time = (time.time() - start_time) * 1000 # in ms

    # Layout: Top Row Metrics
    st.success(f"Algorithm converged successfully in {execution_time:.2f} ms")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Total Points", len(X))
    col_m2.metric("Clusters (k)", k_value)
    col_m3.metric("Execution Speed", f"{execution_time:.2f} ms")

    st.divider()

    # Layout: Visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Elbow Method Analysis")
        inertias = []
        k_range = range(1, 11)
        for i in k_range:
            km = KMeansScratch(k=i)
            l, c = km.fit(X)
            inertias.append(km.get_inertia(X, l, c))
        
        fig_elbow, ax_elbow = plt.subplots()
        ax_elbow.plot(k_range, inertias, 'go-', linewidth=2)
        ax_elbow.set_xlabel('Number of Clusters (k)')
        ax_elbow.set_ylabel('Inertia (WCSS)')
        ax_elbow.grid(True, alpha=0.3)
        st.pyplot(fig_elbow)

    with col2:
        st.subheader(f"2. Result (k={k_value})")
        fig_map, ax_map = plt.subplots()
        ax_map.scatter(X[:, 0], X[:, 1], c=final_labels, cmap='viridis', alpha=0.5, edgecolors='w')
        ax_map.scatter(final_centers[:, 0], final_centers[:, 1], c='red', marker='X', s=250, label="Centroids")
        ax_map.set_title("Cluster Assignments")
        ax_map.legend()
        st.pyplot(fig_map)

    # Cluster Summary
    st.divider()
    st.subheader("3. Cluster Summary Statistics")
    counts = pd.Series(final_labels).value_counts().sort_index()
    summary_list = []
    for i in range(len(final_centers)):
        summary_list.append({
            "Cluster ID": i,
            "Point Count": counts.get(i, 0),
            "Centroid X": round(final_centers[i][0], 4),
            "Centroid Y": round(final_centers[i][1], 4)
        })
    st.table(pd.DataFrame(summary_list))

    # Download CSV
    results_df = pd.DataFrame(X, columns=["Feature_1", "Feature_2"])
    results_df['Cluster_Label'] = final_labels
    csv_data = results_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Labeled Data (CSV)", data=csv_data, file_name='kmeans_results.csv')