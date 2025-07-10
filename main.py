import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import KMeans
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.utils import calculate_distance_matrix
from sklearn.impute import SimpleImputer
from mpl_toolkits.mplot3d import Axes3D
import random

# Custom CSS for background and bright theme
page_bg_img = '''
<style>
body {
background-image: url("https://images.unsplash.com/photo-1508780709619-79562169bc64?auto=format&fit=crop&w=1400&q=80");
background-size: cover;
}
.stApp {
background-color: rgba(255, 255, 255, 0.9);
border-radius: 10px;
padding: 2rem;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

st.set_page_config(page_title="Student Clustering", layout="wide")
st.title("📊 Student Performance Clustering")
st.markdown("Upload data siswa dan lakukan klasterisasi menggunakan **K-Means** dan **K-Medoids** secara bersamaan.")

uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

# === 🧹 PREPROCESSING ===
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, delimiter=';')
    nilai_kolom = ["IPA", "IPS", "MTK", "BIN", "BING", "SUN", "PAI", "PKN", "PNJ", "SBDY", "PRK"]
    for col in nilai_kolom:
        df[col] = df[col].replace('-', np.nan).replace(',', '.', regex=True)
        df[col] = df[col].astype(float)

    imputer = SimpleImputer(strategy='mean')
    df[nilai_kolom] = imputer.fit_transform(df[nilai_kolom])

    df["Pengetahuan_Sains"] = df[["IPA", "MTK", "BIN", "BING", "SUN", "PAI", "PKN"]].mean(axis=1)
    df["Pengetahuan_Sosial"] = df[["IPS", "BIN", "BING", "SUN", "PAI", "PKN"]].mean(axis=1)
    df["Nilai_Keterampilan_Tertinggi"] = df[["PNJ", "SBDY", "PRK"]].max(axis=1)

    fitur = ["Pengetahuan_Sains", "Pengetahuan_Sosial", "Nilai_Keterampilan_Tertinggi"]
    X = df[fitur].values
    scaler = StandardScaler()
    # === 🔄 TRANSFORMASI ===
    X_scaled = scaler.fit_transform(X)

    df_scaled = pd.DataFrame(X_scaled, columns=fitur)
    st.subheader("📊 Hasil Transformasi (StandardScaler)")
    st.dataframe(df_scaled.head())
    csv_scaled = df_scaled.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Unduh Hasil Transformasi", data=csv_scaled, file_name='data_tertransformasi.csv', mime='text/csv')

    if 'Cluster_KMeans' in df.columns and 'Cluster_KMedoids' in df.columns:
        st.write("### Data dengan Hasil Klasterisasi", df[[*fitur, 'Cluster_KMeans', 'Cluster_KMedoids']].head())
    else:
        st.write("### Data Awal", df.head())

    # === 🔍 EVALUASI K: ELBOW METHOD ===
    distortions = []
    K_range = range(2, 11)
    for k_val in K_range:
        km = KMeans(n_clusters=k_val, random_state=42).fit(X_scaled)
        distortions.append(km.inertia_)

    st.subheader("📈 Grafik Elbow untuk Menentukan k Optimal")
    fig_elbow, ax_elbow = plt.subplots()
    ax_elbow.plot(K_range, distortions, 'bo-')
    ax_elbow.set_xlabel("Jumlah Klaster (k)")
    ax_elbow.set_ylabel("Inertia")
    ax_elbow.set_title("Elbow Method")
    st.pyplot(fig_elbow)

    k = st.slider("Pilih jumlah klaster (k):", min_value=2, max_value=10, value=3)

    # === 🤖 DATA MINING: K-MEANS & K-MEDOIDS CLUSTERING ===
    if st.button("Lakukan Klasterisasi"):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(X_scaled)
        df['Cluster_KMeans'] = kmeans.labels_
        dbi_kmeans = davies_bouldin_score(X_scaled, df['Cluster_KMeans'])

        data_size = X_scaled.shape[0]
        random.seed(42)
        initial_medoids = random.sample(range(data_size), k)
        kmedoids_instance = kmedoids(data=X_scaled, initial_index_medoids=initial_medoids, method="pam")
        kmedoids_instance.process()

        clusters = kmedoids_instance.get_clusters()
        labels = np.zeros(data_size, dtype=int)
        for cluster_id, cluster_indices in enumerate(clusters):
            for index in cluster_indices:
                labels[index] = cluster_id

        df['Cluster_KMedoids'] = labels
        dbi_kmedoids = davies_bouldin_score(X_scaled, df['Cluster_KMedoids'])

        st.success(f"Davies-Bouldin Index (K-Means): {dbi_kmeans:.4f} | K-Medoids: {dbi_kmedoids:.4f}")

        # === VISUALISASI CLUSTER ===
        st.subheader("📌 Visualisasi 3D Klasterisasi")
        fig1 = plt.figure(figsize=(10, 6))
        ax1 = fig1.add_subplot(121, projection='3d')
        ax1.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], c=df['Cluster_KMeans'], cmap='viridis')
        ax1.set_title("K-Means")
        ax1.set_xlabel("Pengetahuan Sains")
        ax1.set_ylabel("Pengetahuan Sosial")
        ax1.set_zlabel("Keterampilan Tertinggi")

        ax2 = fig1.add_subplot(122, projection='3d')
        ax2.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], c=df['Cluster_KMedoids'], cmap='plasma')
        ax2.set_title("K-Medoids")
        ax2.set_xlabel("Pengetahuan Sains")
        ax2.set_ylabel("Pengetahuan Sosial")
        ax2.set_zlabel("Keterampilan Tertinggi")

        st.pyplot(fig1)

        # === DIAGRAM PIE GABUNGAN ===
        st.subheader("🥧 Diagram Pie Gabungan: Dominasi Pengetahuan dan Keterampilan Tertinggi")
        df['Dominan_Pengetahuan'] = np.where(df['Pengetahuan_Sains'] >= df['Pengetahuan_Sosial'], 'Sains', 'Sosial')
        df['Asal_Keterampilan_Tertinggi'] = df[['PNJ', 'SBDY', 'PRK']].idxmax(axis=1)
        kombinasi_pie = df.groupby(['Dominan_Pengetahuan', 'Asal_Keterampilan_Tertinggi']).size()

        fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
        label_map = {"PNJ": "Pendidikan Jasmani dan Olahraga", "SBDY": "Seni Budaya", "PRK": "Prakarya"}
        labels = kombinasi_pie.index.map(lambda x: f"{x[0]} - {label_map.get(x[1], x[1])}")
        ax_pie.pie(kombinasi_pie.values, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
        ax_pie.axis('equal')
        st.pyplot(fig_pie)

else:
    st.info("Silakan unggah file CSV terlebih dahulu.")
