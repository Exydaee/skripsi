import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.impute import SimpleImputer
from sklearn.metrics import davies_bouldin_score
from mpl_toolkits.mplot3d import Axes3D
import base64

# ======== FUNGSI TAMBAHAN UNTUK BACKGROUND =========
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .block-container {{
        background-color: rgba(255, 255, 255, 0.8);
        padding: 2rem;
        border-radius: 12px;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ======== ATUR PAGE =========
st.set_page_config(page_title="Klasterisasi Siswa SMP", layout="wide")
set_background("background.png")  # ganti dengan nama file gambar background kamu
st.markdown("<h1 style='text-align: center; color: navy;'>📊 Klasterisasi Siswa SMP Berdasarkan Nilai Rapor</h1>", unsafe_allow_html=True)

# ======== UPLOAD =========
uploaded_file = st.file_uploader("📁 Upload file CSV nilai siswa", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, delimiter=';')

    for col in df.select_dtypes(include='object').columns:
        try:
            df[col] = df[col].str.replace('-', 'NaN').str.replace(',', '.').astype(float)
        except:
            pass

    imputer = SimpleImputer(strategy='mean')
    df[df.columns] = imputer.fit_transform(df)

    df["Pengetahuan_Sains"] = df[["IPA", "MTK", "BIN", "BING", "SUN", "PAI", "PKN"]].mean(axis=1)
    df["Pengetahuan_Sosial"] = df[["IPS", "BIN", "BING", "SUN", "PAI", "PKN"]].mean(axis=1)
    df["Keterampilan_Tertinggi"] = df[["PRK", "SBDY", "PNJ"]].max(axis=1)

    fitur = ["Pengetahuan_Sains", "Pengetahuan_Sosial", "Keterampilan_Tertinggi"]
    X = df[fitur]
    X_scaled = StandardScaler().fit_transform(X)

    st.subheader("🔢 Pilih Jumlah Klaster")
    k = st.slider("Jumlah Klaster", 2, 10, 3)

    kmeans = KMeans(n_clusters=k, random_state=42)
    df["Klaster_KMeans"] = kmeans.fit_predict(X_scaled)

    kmedoids = KMedoids(n_clusters=k, random_state=42)
    df["Klaster_KMedoids"] = kmedoids.fit_predict(X_scaled)

    dbi_kmeans = davies_bouldin_score(X_scaled, df["Klaster_KMeans"])
    dbi_kmedoids = davies_bouldin_score(X_scaled, df["Klaster_KMedoids"])

    def klasifikasi_predikat(sains, sosial, keterampilan):
        r = (sains + sosial + keterampilan) / 3
        if r >= 90: return "Sangat Baik"
        elif r >= 80: return "Baik"
        elif r >= 70: return "Cukup"
        else: return "Perlu Bimbingan"

    df["Predikat"] = df.apply(lambda row: klasifikasi_predikat(
        row["Pengetahuan_Sains"], row["Pengetahuan_Sosial"], row["Keterampilan_Tertinggi"]), axis=1)

    df["Gabungan"] = df.apply(
        lambda row: f"{'Sains' if row['Pengetahuan_Sains'] > row['Pengetahuan_Sosial'] else 'Sosial'} - {['PRK','SBDY','PNJ'][np.argmax([row['PRK'], row['SBDY'], row['PNJ']])]}"
        if not pd.isna(row['PRK']) and not pd.isna(row['SBDY']) and not pd.isna(row['PNJ']) else "-", axis=1)

    # ======= TABEL & EVALUASI ========
    st.subheader("📋 Tabel dan Evaluasi")
    st.write(f"**Davies-Bouldin Index K-Means:** `{dbi_kmeans:.4f}`")
    st.write(f"**Davies-Bouldin Index K-Medoids:** `{dbi_kmedoids:.4f}`")
    st.dataframe(df[fitur + ["Klaster_KMeans", "Klaster_KMedoids", "Predikat", "Gabungan"]])

    # ======= VISUALISASI 2D ========
    st.subheader("📈 Visualisasi 2D")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.scatterplot(x=X["Pengetahuan_Sains"], y=X["Keterampilan_Tertinggi"], hue=df["Klaster_KMeans"], palette="tab10", ax=ax)
        ax.set_title("K-Means")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        sns.scatterplot(x=X["Pengetahuan_Sains"], y=X["Keterampilan_Tertinggi"], hue=df["Klaster_KMedoids"], palette="Set2", ax=ax)
        ax.set_title("K-Medoids")
        st.pyplot(fig)

    # ======= VISUALISASI 3D ========
    st.subheader("🧊 Visualisasi 3D")
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(X["Pengetahuan_Sains"], X["Pengetahuan_Sosial"], X["Keterampilan_Tertinggi"], c=df["Klaster_KMeans"], cmap="tab10")
    ax1.set_title("K-Means 3D")
    ax1.set_xlabel("Sains")
    ax1.set_ylabel("Sosial")
    ax1.set_zlabel("Keterampilan")

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(X["Pengetahuan_Sains"], X["Pengetahuan_Sosial"], X["Keterampilan_Tertinggi"], c=df["Klaster_KMedoids"], cmap="Set2")
    ax2.set_title("K-Medoids 3D")
    ax2.set_xlabel("Sains")
    ax2.set_ylabel("Sosial")
    ax2.set_zlabel("Keterampilan")
    st.pyplot(fig)

    # ======= PIE CHARTS ========
    st.subheader("📊 Pie Chart Distribusi")
    pie_cols = st.columns(3)
    with pie_cols[0]:
        fig, ax = plt.subplots()
        df["Klaster_KMeans"].value_counts().sort_index().plot.pie(autopct="%1.1f%%", ax=ax)
        ax.set_ylabel("")
        ax.set_title("Distribusi KMeans")
        st.pyplot(fig)

    with pie_cols[1]:
        fig, ax = plt.subplots()
        df["Klaster_KMedoids"].value_counts().sort_index().plot.pie(autopct="%1.1f%%", ax=ax)
        ax.set_ylabel("")
        ax.set_title("Distribusi KMedoids")
        st.pyplot(fig)

    with pie_cols[2]:
        fig, ax = plt.subplots()
        df["Gabungan"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax, colors=plt.cm.Paired.colors)
        ax.set_ylabel("")
        ax.set_title("Gabungan Pengetahuan & Keterampilan")
        st.pyplot(fig)

    # ======= UNDUH HASIL ========
    st.subheader("⬇️ Unduh Hasil")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "hasil_klasterisasi.csv", "text/csv")

else:
    st.info("Silakan upload file CSV terlebih dahulu.")
