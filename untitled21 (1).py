import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.utils import calculate_distance_matrix
from sklearn.impute import SimpleImputer
import random
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(layout="wide")
st.title("Aplikasi Klasterisasi Nilai Siswa (K-Means & K-Medoids)")

uploaded_file = st.file_uploader("Upload file CSV siswa (delimiter ;)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, delimiter=';')

    # Preprocessing
    for col in df.select_dtypes(include=['object']).columns:
        try:
            df[col] = df[col].str.replace('-', 'NaN').str.replace(',', '.').astype(float)
        except:
            pass

    imputer = SimpleImputer(strategy='mean')
    df[df.columns] = imputer.fit_transform(df)

    df["Pengetahuan_Sains"] = df[["IPA", "MTK", "BIN", "BING", "SUN", "PAI", "PKN"]].mean(axis=1)
    df["Pengetahuan_Sosial"] = df[["IPS", "BIN", "BING", "SUN", "PAI", "PKN"]].mean(axis=1)
    df["Nilai_Keterampilan_Tertinggi"] = df[["PNJ", "SBDY", "PRK"]].max(axis=1)

    def keterampilan_tertinggi(row):
        nilai = {k: row[k] for k in ["PNJ", "SBDY", "PRK"] if pd.notna(row[k])}
        return max(nilai, key=nilai.get) if nilai else np.nan

    df["Keterampilan_Tertinggi"] = df.apply(keterampilan_tertinggi, axis=1)

    # Fitur klasterisasi
    X = df[["Pengetahuan_Sains", "Pengetahuan_Sosial", "Nilai_Keterampilan_Tertinggi"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Pilih jumlah klaster
    k = st.slider("Pilih jumlah klaster (K)", 2, 10, 3)

    # K-Means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["Cluster_KMeans"] = kmeans.fit_predict(X_scaled)

    # K-Medoids
    initial_medoids = random.sample(range(len(X_scaled)), k)
    kmedoids_instance = kmedoids(data=X_scaled, initial_index_medoids=initial_medoids, method="pam")
    kmedoids_instance.process()
    clusters = kmedoids_instance.get_clusters()

    labels_kmedoids = np.zeros(len(X_scaled), dtype=int)
    for cid, cluster in enumerate(clusters):
        for idx in cluster:
            labels_kmedoids[idx] = cid
    df["Cluster_KMedoids"] = labels_kmedoids

    # Evaluasi DBI
    dbi_kmeans = davies_bouldin_score(X_scaled, df["Cluster_KMeans"])
    dbi_kmedoids = davies_bouldin_score(X_scaled, df["Cluster_KMedoids"])

    st.subheader("Evaluasi Davies-Bouldin Index")
    st.write(f"K-Means DBI: {dbi_kmeans:.4f}")
    st.write(f"K-Medoids DBI: {dbi_kmedoids:.4f}")

    # Visualisasi 3D
    st.subheader("Visualisasi 3D Klaster")
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(X["Pengetahuan_Sains"], X["Pengetahuan_Sosial"], X["Nilai_Keterampilan_Tertinggi"],
               c=df["Cluster_KMeans"], cmap='viridis', s=60)
    ax.set_title("K-Means")

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(X["Pengetahuan_Sains"], X["Pengetahuan_Sosial"], X["Nilai_Keterampilan_Tertinggi"],
                c=df["Cluster_KMedoids"], cmap='plasma', s=60)
    ax2.set_title("K-Medoids")
    st.pyplot(fig)

    # Tabel hasil akhir
    st.subheader("Tabel Hasil Klaster")
    st.dataframe(df[["Pengetahuan_Sains", "Pengetahuan_Sosial", "Nilai_Keterampilan_Tertinggi",
                    "Keterampilan_Tertinggi", "Cluster_KMeans", "Cluster_KMedoids"]])
