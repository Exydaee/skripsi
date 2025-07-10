import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
st.markdown("Upload data siswa dan lakukan klasterisasi menggunakan K-Means atau K-Medoids")

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

    cluster_method = st.selectbox("Pilih Metode Klasterisasi", ["K-Means", "K-Medoids"])
    k = st.slider("Jumlah Klaster (k)", 2, 10, 3)

    if st.button("Lakukan Klasterisasi"):
        if cluster_method == "K-Means":
            model = KMeans(n_clusters=k, random_state=42).fit(X_scaled)
            labels = model.labels_
        else:
            dist_matrix = calculate_distance_matrix(X_scaled)
            initial_medoids = random.sample(range(len(X_scaled)), k)
            kmedoids_instance = kmedoids(dist_matrix, initial_medoids, data_type='distance_matrix')
            kmedoids_instance.process()
            labels = np.zeros(len(X_scaled))
            for i, cluster in enumerate(kmedoids_instance.get_clusters()):
                for idx in cluster:
                    labels[idx] = i

        df['Cluster'] = labels.astype(int)
        dbi = davies_bouldin_score(X_scaled, df['Cluster'])

        st.success(f"Davies-Bouldin Index (DBI): {dbi:.4f} (semakin kecil semakin baik)")
        st.write("### Hasil Klasterisasi", df[[*fitur, 'Cluster']].head())

        # Pie Chart
        st.write("### Distribusi Klaster")
        fig1, ax1 = plt.subplots()
        df['Cluster'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, ax=ax1)
        ax1.axis('equal')
        st.pyplot(fig1)

        # 3D Scatter Plot
        st.write("### Visualisasi 3D")
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111, projection='3d')
        scatter = ax2.scatter(df[fitur[0]], df[fitur[1]], df[fitur[2]], c=df['Cluster'], cmap='viridis')
        ax2.set_xlabel(fitur[0])
        ax2.set_ylabel(fitur[1])
        ax2.set_zlabel(fitur[2])
        st.pyplot(fig2)
else:
    st.info("Silakan unggah file CSV terlebih dahulu.")
