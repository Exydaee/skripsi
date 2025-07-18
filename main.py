# === Library dan Konfigurasi Awal ===
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
from matplotlib.cm import get_cmap

# Konfigurasi halaman dan latar belakang
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
st.title("📊 Klasterisasi Pengetahuan dan Keterampilan Siswa: Perbandingan K-Means vs K-Medoids")
st.markdown("Upload data siswa dan lakukan klasterisasi menggunakan **K-Means** dan **K-Medoids** secara bersamaan.")

# === Tahapan Knowledge Discovery in Databases (KDD) ===

# =====================================
# 1. SELEKSI DATA
# =====================================
uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])
if uploaded_file is not None:
    st.subheader("📌 Data Awal")
    nilai_kolom = ["IPA", "IPS", "MTK", "BIN", "BING", "SUN", "PAI", "PKN", "PNJ", "SBDY", "PRK"]
    

    df = pd.read_csv(uploaded_file, delimiter=';')
    
    st.write("📄 Data awal sebelum dilakukan proses apapun:")
    st.dataframe(df.head())
# =====================================
# 2. PREPROCESSING
# =====================================
    for col in nilai_kolom:
        df[col] = df[col].replace('-', np.nan).replace(',', '.', regex=True)
        df[col] = df[col].astype(float)

    imputer = SimpleImputer(strategy='mean')
    df[nilai_kolom] = imputer.fit_transform(df[nilai_kolom])

    # Pengelompokan nilai pengetahuan dan keterampilan
    df["Pengetahuan_Sains"] = df[["IPA", "MTK", "BIN", "BING", "SUN", "PAI", "PKN"]].mean(axis=1)
    df["Pengetahuan_Sosial"] = df[["IPS", "BIN", "BING", "SUN", "PAI", "PKN"]].mean(axis=1)
    df["Nilai_Keterampilan_Tertinggi"] = df[["PNJ", "SBDY", "PRK"]].max(axis=1)
    df["Keterampilan_Tertinggi"] = df[["PNJ", "SBDY", "PRK"]].idxmax(axis=1).replace({"PNJ": "Pendidikan Jasmani dan Olahraga", "SBDY": "Seni Budaya", "PRK": "Prakarya"})

    st.subheader("🧹 Preprocessing Data")
    st.write("📌 Data setelah konversi simbol dan imputasi nilai kosong:")
    st.dataframe(df[nilai_kolom].head())

    st.write("📌 Data setelah penambahan fitur Pengetahuan dan Keterampilan:")
    st.dataframe(df[["Pengetahuan_Sains", "Pengetahuan_Sosial", "Nilai_Keterampilan_Tertinggi", "Keterampilan_Tertinggi"]].head())

    # =====================================
    # 3. TRANSFORMASI
    # =====================================
    st.subheader("🔄 Transformasi Data")

    fitur = ["Pengetahuan_Sains", "Pengetahuan_Sosial", "Nilai_Keterampilan_Tertinggi"]
    X = df[fitur].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    st.write("Data hasil transformasi menggunakan StandardScaler:")
    st.write(pd.DataFrame(X_scaled, columns=fitur).head())

    # =====================================
    # 4. DATA MINING
    # =====================================
    st.subheader("🤖 Klasterisasi (Data Mining)")
    distortions = []
    K_range = range(2, 11)
    for k_val in K_range:
        km = KMeans(n_clusters=k_val, random_state=42).fit(X_scaled)
        distortions.append(km.inertia_)

    st.subheader("📈 Elbow Method untuk Menentukan k Optimal")
    fig_elbow, ax_elbow = plt.subplots()
    ax_elbow.plot(K_range, distortions, 'bo-')
    ax_elbow.set_xlabel("Jumlah Klaster (k)")
    ax_elbow.set_ylabel("Inertia")
    ax_elbow.set_title("Elbow Method")
    st.pyplot(fig_elbow)

    k = st.slider("Pilih jumlah klaster (k):", min_value=2, max_value=10, value=3)

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

        # =====================================
        # 5. EVALUASI
        # =====================================
        st.subheader("📊 Tahap 5: Evaluasi")
        st.success(f"Davies-Bouldin Index (K-Means): {dbi_kmeans:.4f} | K-Medoids: {dbi_kmedoids:.4f}")

        # =====================================
        # 6. INTERPRETASI DAN VISUALISASI
        # =====================================
        st.subheader("📌 Interpretasi dan Visualisasi")

        def generate_colors(n, cmap_name='tab10'):
            cmap = get_cmap(cmap_name)
            return [cmap(i % cmap.N) for i in range(n)]

        num_clusters_kmeans = len(df['Cluster_KMeans'].unique())
        num_clusters_kmedoids = len(df['Cluster_KMedoids'].unique())

        color_list_kmeans = generate_colors(num_clusters_kmeans, 'tab10')
        color_list_kmedoids = generate_colors(num_clusters_kmedoids, 'Set3')

        cluster_colors_kmeans = {i: color_list_kmeans[i] for i in range(num_clusters_kmeans)}
        cluster_colors_kmedoids = {i: color_list_kmedoids[i] for i in range(num_clusters_kmedoids)}

        df['Warna_KMeans'] = df['Cluster_KMeans'].map(cluster_colors_kmeans)
        df['Warna_KMedoids'] = df['Cluster_KMedoids'].map(cluster_colors_kmedoids)

        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(
            df["Pengetahuan_Sains"], df["Pengetahuan_Sosial"], df["Nilai_Keterampilan_Tertinggi"],
            c=df["Warna_KMeans"], s=60
        )
        ax1.set_title("3D Scatter Plot K-Means")
        ax1.set_xlabel("Pengetahuan Sains")
        ax1.set_ylabel("Pengetahuan Sosial")
        ax1.set_zlabel("Nilai Keterampilan Tertinggi")

        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(
            df["Pengetahuan_Sains"], df["Pengetahuan_Sosial"], df["Nilai_Keterampilan_Tertinggi"],
            c=df["Warna_KMedoids"], s=60
        )
        ax2.set_title("3D Scatter Plot K-Medoids")
        ax2.set_xlabel("Pengetahuan Sains")
        ax2.set_ylabel("Pengetahuan Sosial")
        ax2.set_zlabel("Nilai Keterampilan Tertinggi")

        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("📊 Distribusi Klaster K-Means")
        fig_pie_kmeans, ax_kmeans = plt.subplots()
        df['Cluster_KMeans'].value_counts().sort_index().plot.pie(
            autopct='%1.1f%%', ax=ax_kmeans,
            colors=[cluster_colors_kmeans[i] for i in sorted(cluster_colors_kmeans)]
        )
        ax_kmeans.set_ylabel('')
        ax_kmeans.set_title("Distribusi Klaster - KMeans")
        st.pyplot(fig_pie_kmeans)

        st.subheader("📊 Distribusi Klaster K-Medoids")
        fig_pie_kmedoids, ax_kmedoids = plt.subplots()
        df['Cluster_KMedoids'].value_counts().sort_index().plot.pie(
            autopct='%1.1f%%', ax=ax_kmedoids,
            colors=[cluster_colors_kmedoids[i] for i in sorted(cluster_colors_kmedoids)]
        )
        ax_kmedoids.set_ylabel('')
        ax_kmedoids.set_title("Distribusi Klaster - KMedoids")
        st.pyplot(fig_pie_kmedoids)

        st.subheader("🥧 Diagram Pie Gabungan")
        df['Gabungan'] = df.apply(
            lambda row: f"{'Sains' if row['Pengetahuan_Sains'] > row['Pengetahuan_Sosial'] else 'Sosial'} - {row['Keterampilan_Tertinggi']}", axis=1
        )
        gabungan_counts = df['Gabungan'].value_counts().sort_index()

        fig_pie, ax_pie = plt.subplots(figsize=(8, 8))
        ax_pie.pie(
            gabungan_counts.values,
            labels=gabungan_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=plt.cm.Paired.colors
        )
        ax_pie.axis('equal')
        plt.tight_layout()
        st.pyplot(fig_pie)

else:
    st.info("Silakan unggah file CSV terlebih dahulu.")
