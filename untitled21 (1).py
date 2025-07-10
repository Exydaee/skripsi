# Streamlit Web App Lengkap: Klasterisasi Siswa dengan K-Means & K-Medoids

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.impute import SimpleImputer
from sklearn_extra.cluster import KMedoids
from mpl_toolkits.mplot3d import Axes3D

# Konfigurasi halaman
st.set_page_config(page_title="📊 Klasterisasi Siswa SMP", layout="wide")
st.title("🎓 Klasterisasi Nilai Siswa SMP Berdasarkan Nilai Rapor")

# Sidebar untuk upload dan parameter
with st.sidebar:
    st.header("🔧 Pengaturan")
    uploaded_file = st.file_uploader("📂 Upload file CSV nilai siswa", type=["csv"])
    k = st.slider("🔢 Pilih Jumlah Klaster (K)", min_value=2, max_value=10, value=3)

# Jika file sudah diunggah
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, delimiter=';')

    # Pembersihan data
    for col in df.select_dtypes(include='object').columns:
        try:
            df[col] = df[col].str.replace('-', 'NaN').str.replace(',', '.').astype(float)
        except:
            pass

    imputer = SimpleImputer(strategy='mean')
    df[df.columns] = imputer.fit_transform(df)

    # Hitung fitur
    df["Pengetahuan_Sains"] = df[["IPA", "MTK", "BIN", "BING", "SUN", "PAI", "PKN"]].mean(axis=1)
    df["Pengetahuan_Sosial"] = df[["IPS", "BIN", "BING", "SUN", "PAI", "PKN"]].mean(axis=1)
    df["Keterampilan_Tertinggi"] = df[["PRK", "SBDY", "PNJ"]].max(axis=1)

    fitur = ["Pengetahuan_Sains", "Pengetahuan_Sosial", "Keterampilan_Tertinggi"]
    X = df[fitur]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans dan KMedoids
    kmeans = KMeans(n_clusters=k, random_state=42)
    df["Klaster_KMeans"] = kmeans.fit_predict(X_scaled)

    kmedoids = KMedoids(n_clusters=k, random_state=42)
    df["Klaster_KMedoids"] = kmedoids.fit_predict(X_scaled)

    dbi_kmeans = davies_bouldin_score(X_scaled, df["Klaster_KMeans"])
    dbi_kmedoids = davies_bouldin_score(X_scaled, df["Klaster_KMedoids"])

    # Predikat Akademik
    def klasifikasi_predikat(sains, sosial, keterampilan):
        rata2 = (sains + sosial + keterampilan) / 3
        if rata2 >= 90:
            return "Sangat Baik"
        elif rata2 >= 80:
            return "Baik"
        elif rata2 >= 70:
            return "Cukup"
        else:
            return "Perlu Bimbingan"

    df["Predikat"] = df.apply(lambda row: klasifikasi_predikat(
        row["Pengetahuan_Sains"], row["Pengetahuan_Sosial"], row["Keterampilan_Tertinggi"]), axis=1)

    df["Gabungan"] = df.apply(
        lambda row: f"{'Sains' if row['Pengetahuan_Sains'] > row['Pengetahuan_Sosial'] else 'Sosial'} - {['PRK', 'SBDY', 'PNJ'][np.argmax([row['PRK'], row['SBDY'], row['PNJ']])]}"
        if not pd.isna(row['PRK']) and not pd.isna(row['SBDY']) and not pd.isna(row['PNJ']) else "-",
        axis=1
    )

    # Tampilkan DBI dan Tabel
    st.subheader("📈 Evaluasi Klasterisasi")
    col1, col2 = st.columns(2)
    col1.metric("Davies-Bouldin Index - KMeans", f"{dbi_kmeans:.4f}")
    col2.metric("Davies-Bouldin Index - KMedoids", f"{dbi_kmedoids:.4f}")

    st.subheader("📊 Tabel Hasil Klasterisasi")
    st.dataframe(df[fitur + ["Klaster_KMeans", "Klaster_KMedoids", "Predikat", "Gabungan"]])

    # Visualisasi 2D
    st.subheader("🟠 Visualisasi 2D")
    col3, col4 = st.columns(2)
    with col3:
        fig1, ax1 = plt.subplots()
        sns.scatterplot(
            x=df["Pengetahuan_Sains"], y=df["Keterampilan_Tertinggi"],
            hue=df["Klaster_KMeans"], palette="tab10", ax=ax1)
        ax1.set_title("KMeans Clustering", fontsize=14)
        st.pyplot(fig1)

    with col4:
        fig2, ax2 = plt.subplots()
        sns.scatterplot(
            x=df["Pengetahuan_Sains"], y=df["Keterampilan_Tertinggi"],
            hue=df["Klaster_KMedoids"], palette="Set2", ax=ax2)
        ax2.set_title("KMedoids Clustering", fontsize=14)
        st.pyplot(fig2)

    # Visualisasi 3D
    st.subheader("🔵 Visualisasi 3D")
    fig3 = plt.figure(figsize=(12, 5))
    ax3 = fig3.add_subplot(121, projection='3d')
    ax3.scatter(df["Pengetahuan_Sains"], df["Pengetahuan_Sosial"], df["Keterampilan_Tertinggi"],
                c=df["Klaster_KMeans"], cmap="tab10")
    ax3.set_title("KMeans 3D")
    ax3.set_xlabel("Sains")
    ax3.set_ylabel("Sosial")
    ax3.set_zlabel("Keterampilan")

    ax4 = fig3.add_subplot(122, projection='3d')
    ax4.scatter(df["Pengetahuan_Sains"], df["Pengetahuan_Sosial"], df["Keterampilan_Tertinggi"],
                c=df["Klaster_KMedoids"], cmap="Set2")
    ax4.set_title("KMedoids 3D")
    ax4.set_xlabel("Sains")
    ax4.set_ylabel("Sosial")
    ax4.set_zlabel("Keterampilan")
    st.pyplot(fig3)

    # Pie Chart
    st.subheader("📌 Distribusi Klaster")
    col5, col6, col7 = st.columns(3)
    with col5:
        fig5, ax5 = plt.subplots()
        df["Klaster_KMeans"].value_counts().sort_index().plot.pie(
            autopct="%1.1f%%", ax=ax5, startangle=90, colors=sns.color_palette("tab10"))
        ax5.set_title("Distribusi KMeans")
        ax5.set_ylabel("")
        st.pyplot(fig5)

    with col6:
        fig6, ax6 = plt.subplots()
        df["Klaster_KMedoids"].value_counts().sort_index().plot.pie(
            autopct="%1.1f%%", ax=ax6, startangle=90, colors=sns.color_palette("Set2"))
        ax6.set_title("Distribusi KMedoids")
        ax6.set_ylabel("")
        st.pyplot(fig6)

    with col7:
        fig7, ax7 = plt.subplots()
        df["Gabungan"].value_counts().plot.pie(
            autopct='%1.1f%%', ax=ax7, startangle=90, colors=plt.cm.Paired.colors)
        ax7.set_title("Gabungan Pengetahuan & Keterampilan")
        ax7.set_ylabel("")
        st.pyplot(fig7)

    # Unduh hasil
    st.subheader("💾 Unduh Hasil")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", csv, "hasil_klasterisasi.csv", "text/csv")

else:
    st.info("📌 Silakan upload file CSV terlebih dahulu melalui sidebar.")
