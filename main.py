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
import random
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

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, delimiter=';')
    st.write("### Data Awal", df.head())

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

    df["Keterampilan_Tertinggi"] = df.apply(
        lambda row: max([row["PNJ"], row["SBDY"], row["PRK"]]) if pd.notna(row["PNJ"]) and pd.notna(row["SBDY"]) and pd.notna(row["PRK"]) else np.nan,
        axis=1)

    df["Nilai_Keterampilan_Tertinggi"] = df[["PNJ", "SBDY", "PRK"]].max(axis=1)

    fitur = ["Pengetahuan_Sains", "Pengetahuan_Sosial", "Nilai_Keterampilan_Tertinggi"]
    X = df[fitur].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    st.subheader("📈 Elbow Method untuk Menentukan k Optimal")
    distortions = []
    K = range(2, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(X_scaled)
        distortions.append(kmeans.inertia_)

    fig_elbow, ax_elbow = plt.subplots()
    ax_elbow.plot(K, distortions, 'bx-')
    ax_elbow.set_xlabel('Jumlah Klaster k')
    ax_elbow.set_ylabel('Inertia')
    ax_elbow.set_title('Metode Elbow untuk Menentukan k')
    st.pyplot(fig_elbow)

    k = st.number_input("Masukkan jumlah klaster (k) terbaik berdasarkan grafik elbow:", min_value=2, max_value=10, value=3, step=1)

    if st.button("Lakukan Klasterisasi"):
        # K-Means
        kmeans = KMeans(n_clusters=k, random_state=42).fit(X_scaled)
        df['Cluster_KMeans'] = kmeans.labels_
        dbi_kmeans = davies_bouldin_score(X_scaled, df['Cluster_KMeans'])

        # K-Medoids
        dist_matrix = calculate_distance_matrix(X_scaled)
        initial_medoids = random.sample(range(len(X_scaled)), k)
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
            fig2 = plt.figure(figsize=(10,8))
            ax2 = fig2.add_subplot(111, projection='3d')
            scatter_kmeans = ax2.scatter(df[fitur[0]], df[fitur[1]], df[fitur[2]], c=df['Cluster_KMeans'], cmap='viridis')
            ax2.set_xlabel('Pengetahuan_Sains', labelpad=15)
            ax2.set_ylabel('Pengetahuan_Sosial', labelpad=15)
            ax2.set_zlabel('Nilai_Keterampilan_Tertinggi', labelpad=15)
            ax2.legend(*scatter_kmeans.legend_elements(), title="Cluster", loc="upper center", bbox_to_anchor=(0.5, 1.15))
            st.pyplot(fig2)

        with col2:
            st.subheader("🔸 K-Medoids Clustering")
            st.dataframe(df[[*fitur, 'Cluster_KMedoids']].head())
            fig3, ax3 = plt.subplots()
            df['Cluster_KMedoids'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, ax=ax3)
            ax3.axis('equal')
            st.pyplot(fig3)
            fig4 = plt.figure(figsize=(10,8))
            ax4 = fig4.add_subplot(111, projection='3d')
            scatter_kmedoids = ax4.scatter(df[fitur[0]], df[fitur[1]], df[fitur[2]], c=df['Cluster_KMedoids'], cmap='plasma')
            ax4.set_xlabel('Pengetahuan_Sains', labelpad=15)
            ax4.set_ylabel('Pengetahuan_Sosial', labelpad=15)
            ax4.set_zlabel('Nilai_Keterampilan_Tertinggi', labelpad=15)
            ax4.legend(*scatter_kmedoids.legend_elements(), title="Cluster", loc="upper center", bbox_to_anchor=(0.5, 1.15))
            st.pyplot(fig4)
else:
    st.info("Silakan unggah file CSV terlebih dahulu.")
