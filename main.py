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
    st.subheader("📈 Elbow Method untuk Menentukan k Optimal")
    distortions = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(X_scaled)
        distortions.append(kmeans.inertia_)
        st.write(f"k = {k}, inertia = {kmeans.inertia_:.2f}")  # Untuk debugging dan validasi

    fig_elbow, ax_elbow = plt.subplots(figsize=(6, 4))
    ax_elbow.plot(K, distortions, 'bx-')
    ax_elbow.set_xlabel('Jumlah Klaster k')
    ax_elbow.set_ylabel('Inertia')
    ax_elbow.set_title('Metode Elbow untuk Menentukan k')
    st.pyplot(fig_elbow)

    with open("elbow_plot.png", "wb") as f:
        fig_elbow.savefig(f)
    with open("elbow_plot.png", "rb") as f:
        st.download_button("📥 Unduh Grafik Elbow", data=f, file_name="elbow_plot.png")

    k = st.number_input("Masukkan jumlah klaster (k) terbaik berdasarkan grafik elbow:", min_value=2, max_value=10, value=3, step=1)

    # === 🤖 DATA MINING: K-MEANS & K-MEDOIDS CLUSTERING ===
    if st.button("Lakukan Klasterisasi"):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(X_scaled)
        df['Cluster_KMeans'] = kmeans.labels_
        dbi_kmeans = davies_bouldin_score(X_scaled, df['Cluster_KMeans'])

        dist_matrix = calculate_distance_matrix(X_scaled)
        initial_medoids = list(range(k))  # Ubah agar hasil k-medoids stabil di semua environment
        kmedoids_instance = kmedoids(dist_matrix, initial_medoids, data_type='distance_matrix')
        kmedoids_instance.process()
        labels_medoid = np.zeros(len(X_scaled))
        for i, cluster in enumerate(kmedoids_instance.get_clusters()):
            for idx in cluster:
                labels_medoid[idx] = i
        df['Cluster_KMedoids'] = labels_medoid.astype(int)
        dbi_kmedoids = davies_bouldin_score(X_scaled, df['Cluster_KMedoids'])

        st.success(f"Davies-Bouldin Index (K-Means): {dbi_kmeans:.4f} | K-Medoids: {dbi_kmedoids:.4f}")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔹 K-Means Clustering")
            st.dataframe(df[[*fitur, 'Cluster_KMeans']].head())
            fig1, ax1 = plt.subplots()
            df['Cluster_KMeans'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, ax=ax1)
            ax1.axis('equal')
            st.pyplot(fig1)

            fig2 = plt.figure(figsize=(10, 8))
            ax2 = fig2.add_subplot(111, projection='3d')
            scatter_kmeans = ax2.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], c=df['Cluster_KMeans'], cmap='viridis')
            ax2.set_xlabel('Pengetahuan_Sains', labelpad=15)
            ax2.set_ylabel('Pengetahuan_Sosial', labelpad=15)
            ax2.set_zlabel('Nilai_Keterampilan_Tertinggi', labelpad=15)
            ax2.legend(*scatter_kmeans.legend_elements(), title="Cluster", loc="lower left", bbox_to_anchor=(1.05, 0.5))
            plt.tight_layout()
            st.pyplot(fig2)

        with col2:
            st.subheader("🔸 K-Medoids Clustering")
            st.dataframe(df[[*fitur, 'Cluster_KMedoids']].head())
            fig3, ax3 = plt.subplots()
            df['Cluster_KMedoids'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, ax=ax3)
            ax3.axis('equal')
            st.pyplot(fig3)

            fig4 = plt.figure(figsize=(10, 8))
            ax4 = fig4.add_subplot(111, projection='3d')
            scatter_kmedoids = ax4.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], c=df['Cluster_KMedoids'], cmap='plasma')
            ax4.set_xlabel('Pengetahuan_Sains', labelpad=15)
            ax4.set_ylabel('Pengetahuan_Sosial', labelpad=15)
            ax4.set_zlabel('Nilai_Keterampilan_Tertinggi', labelpad=15)
            ax4.legend(*scatter_kmedoids.legend_elements(), title="Cluster", loc="lower left", bbox_to_anchor=(1.05, 0.5))
            plt.tight_layout()
            st.pyplot(fig4)

        # === 📊 VISUALISASI LANJUTAN ===
        df['Dominan_Pengetahuan'] = np.where(df['Pengetahuan_Sains'] >= df['Pengetahuan_Sosial'], 'Sains', 'Sosial')
        df['Asal_Keterampilan_Tertinggi'] = df[['PNJ', 'SBDY', 'PRK']].idxmax(axis=1)
        kombinasi_pie = df.groupby(['Dominan_Pengetahuan', 'Asal_Keterampilan_Tertinggi']).size()

        st.subheader("🥧 Diagram Pie: Dominasi Pengetahuan vs Keterampilan Tertinggi")
        fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
        label_map = {"PNJ": "Pendidikan Jasmani dan Olahraga", "SBDY": "Seni Budaya", "PRK": "Prakarya"}
        labels = kombinasi_pie.index.map(lambda x: f"{x[0]} - {label_map.get(x[1], x[1])}")
        ax_pie.pie(kombinasi_pie.values, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
        ax_pie.axis('equal')
        st.pyplot(fig_pie)

else:
    st.info("Silakan unggah file CSV terlebih dahulu.")
