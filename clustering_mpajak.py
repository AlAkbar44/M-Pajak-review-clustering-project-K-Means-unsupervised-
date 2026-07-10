import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # supaya bisa jalan tanpa GUI (headless)
import matplotlib.pyplot as plt

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

INPUT_FILE = "mpajak_reviews_raw.csv"
OUTPUT_FILE = "processed_mpajak_data.csv"

# Batas adjusted_score untuk dianggap "ada indikasi masalah" dan ikut di-cluster.
# <= COMPLAINT_THRESHOLD -> masuk proses clustering
# >  COMPLAINT_THRESHOLD -> langsung dianggap positif, skip clustering
COMPLAINT_THRESHOLD = 3

# Kata-kata generik/filler yang sering muncul tapi tidak menunjukkan topik spesifik.
# Ditambahkan di atas stopword bawaan Sastrawi.
CUSTOM_STOPWORDS = {
    "aja", "nya", "mau", "gak", "ga", "gk", "guna", "buat", "apa", "apk",
    "aplikasi", "kalo", "kok", "sih", "dong", "banget", "lah", "deh", "nih",
    "gitu", "yg", "yang", "kayak", "kaya", "udah", "sudah", "banyak", "bikin",
    "bisa", "baik", "nih", "min", "admin", "tolong", "mohon", "coba", "terus",
}

# ==================================================================
# 1. LOAD DATA
# ==================================================================
df = pd.read_csv(INPUT_FILE)
df["content"] = df["content"].fillna("")

print("=" * 60)
print("1. DATA UNDERSTANDING")
print("=" * 60)
print(f"Jumlah baris awal : {len(df)}")
print(f"Jumlah duplikat   : {df.duplicated(subset='content').sum()}")
print(f"Missing values:\n{df.isnull().sum()}")

df = df.drop_duplicates(subset="content")
df = df[df["content"].str.strip() != ""]
df = df.reset_index(drop=True)
print(f"Jumlah baris setelah drop duplikat/kosong: {len(df)}")

# ==================================================================
# 2. DATA CLEANING & RE-LABELING adjusted_score (rule berbasis kata kunci)
# ==================================================================
KEYWORDS_TECHNICAL = [
    "login", "masuk", "password", "sinkron", "sync", "data",
    "lemot", "lama", "hang", "otp", "verifikasi", "sms",
    "telepon", "pendaftaran", "gagal", "error"
]

def contains_technical_complaint(text):
    text_lower = str(text).lower()
    return any(k in text_lower for k in KEYWORDS_TECHNICAL)

def adjust_score(row):
    if row["score"] == 5 and contains_technical_complaint(row["content"]):
        return 1
    return row["score"]

df["adjusted_score"] = df.apply(adjust_score, axis=1)

# ==================================================================
# 3. TEXT CLEANING (lowercase, buang simbol, stopword bawaan + custom, stemming)
# ==================================================================
stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
stemmer = StemmerFactory().create_stemmer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)      # buang angka & simbol
    text = re.sub(r"\s+", " ", text).strip()   # rapikan spasi
    text = stopword_remover.remove(text)       # buang stopword bahasa Indonesia (Sastrawi)
    text = stemmer.stem(text)                  # stemming ke kata dasar
    # buang custom stopword tambahan
    words = [w for w in text.split() if w not in CUSTOM_STOPWORDS and len(w) > 2]
    return " ".join(words)

print("\nMembersihkan teks (stopword removal + stemming)... ini bisa makan waktu beberapa menit untuk 7000+ baris.")
df["clean_content"] = df["content"].apply(clean_text)

df = df[df["clean_content"].str.strip() != ""].reset_index(drop=True)
print(f"Jumlah baris setelah text cleaning: {len(df)}")

df["word_count"] = df["clean_content"].apply(lambda x: len(x.split()))

# ==================================================================
# 4. EDA
# ==================================================================
print("\n" + "=" * 60)
print("4. EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 60)

print("\nDistribusi score asli:")
print(df["score"].value_counts().sort_index())

print("\nDistribusi adjusted_score:")
print(df["adjusted_score"].value_counts().sort_index())

print("\nStatistik panjang komentar (jumlah kata setelah cleaning):")
print(df["word_count"].describe())

all_words = " ".join(df["clean_content"]).split()
word_freq = pd.Series(all_words).value_counts().head(20)

plt.figure(figsize=(10, 6))
word_freq.sort_values().plot(kind="barh", color="steelblue")
plt.title("20 Kata Paling Sering Muncul (setelah cleaning)")
plt.xlabel("Frekuensi")
plt.tight_layout()
plt.savefig("top20_words.png", dpi=150)
plt.close()
print("\nGrafik top 20 kata disimpan ke 'top20_words.png'")

try:
    from wordcloud import WordCloud
    wc = WordCloud(width=1000, height=500, background_color="white").generate(" ".join(all_words))
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("wordcloud_all.png", dpi=150)
    plt.close()
    print("Wordcloud disimpan ke 'wordcloud_all.png'")
except ImportError:
    print("Library 'wordcloud' tidak terinstall, skip pembuatan wordcloud (opsional).")

# ==================================================================
# 5. SPLIT: ULASAN BERMASALAH (di-cluster) vs POSITIF (skip clustering)
# ==================================================================
print("\n" + "=" * 60)
print("5. SPLIT DATA: KOMPLAIN vs POSITIF")
print("=" * 60)

is_complaint = df["adjusted_score"] <= COMPLAINT_THRESHOLD
df_complaint = df[is_complaint].copy().reset_index(drop=True)
df_positive = df[~is_complaint].copy().reset_index(drop=True)

print(f"Ulasan terindikasi masalah (adjusted_score <= {COMPLAINT_THRESHOLD}): {len(df_complaint)} "
      f"({len(df_complaint)/len(df)*100:.2f}%) -> akan di-cluster")
print(f"Ulasan positif (adjusted_score > {COMPLAINT_THRESHOLD}): {len(df_positive)} "
      f"({len(df_positive)/len(df)*100:.2f}%) -> langsung diberi category 'Positif/Tidak Ada Keluhan'")

# ==================================================================
# 6. FEATURE EXTRACTION -> TF-IDF (hanya pada subset komplain)
# ==================================================================
print("\n" + "=" * 60)
print("6. TF-IDF VECTORIZATION (subset komplain)")
print("=" * 60)

vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.5,   # buang kata yang muncul di >50% dokumen (kata generik/kurang diskriminatif)
)
X = vectorizer.fit_transform(df_complaint["clean_content"])
print(f"Ukuran matrix TF-IDF: {X.shape}")

# ==================================================================
# 7. CARI K OPTIMAL (Elbow Method + Silhouette Score)
# ==================================================================
print("\n" + "=" * 60)
print("7. MENENTUKAN K OPTIMAL (Elbow + Silhouette)")
print("=" * 60)

K_RANGE = range(2, 11)
inertias = []
silhouette_scores = []

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    sil = silhouette_score(X, labels, sample_size=min(2000, X.shape[0]), random_state=42)
    silhouette_scores.append(sil)
    print(f"K={k} | inertia={km.inertia_:.2f} | silhouette={sil:.4f}")

best_k = list(K_RANGE)[silhouette_scores.index(max(silhouette_scores))]
print(f"\n>> K optimal berdasarkan silhouette score tertinggi: K={best_k}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(list(K_RANGE), inertias, marker="o")
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("Jumlah Cluster (K)")
axes[0].set_ylabel("Inertia")

axes[1].plot(list(K_RANGE), silhouette_scores, marker="o", color="green")
axes[1].axvline(best_k, color="red", linestyle="--", label=f"K optimal = {best_k}")
axes[1].set_title("Silhouette Score")
axes[1].set_xlabel("Jumlah Cluster (K)")
axes[1].set_ylabel("Silhouette Score")
axes[1].legend()

plt.tight_layout()
plt.savefig("elbow_silhouette.png", dpi=150)
plt.close()
print("Grafik elbow & silhouette disimpan ke 'elbow_silhouette.png'")

# ==================================================================
# 8. FIT K-MEANS FINAL DENGAN K OPTIMAL (subset komplain)
# ==================================================================
print("\n" + "=" * 60)
print(f"8. FIT K-MEANS FINAL (K={best_k})")
print("=" * 60)

kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df_complaint["cluster"] = kmeans_final.fit_predict(X)

print("\nDistribusi jumlah ulasan per cluster (subset komplain):")
cluster_dist = df_complaint["cluster"].value_counts().sort_index()
for c, n in cluster_dist.items():
    print(f"  Cluster {c}: {n} ulasan ({n/len(df_complaint)*100:.2f}%)")

# ==================================================================
# 9. INTERPRETASI CLUSTER -> top terms tiap cluster
# ==================================================================
print("\n" + "=" * 60)
print("9. TOP TERMS PER CLUSTER (buat kamu kasih nama kategori)")
print("=" * 60)

terms = vectorizer.get_feature_names_out()
order_centroids = kmeans_final.cluster_centers_.argsort()[:, ::-1]

top_terms_report = []
for c in range(best_k):
    top_words = [terms[i] for i in order_centroids[c, :10]]
    line = f"Cluster {c}: {', '.join(top_words)}"
    print(line)
    top_terms_report.append(line)

with open("top_terms_per_cluster.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(top_terms_report))
print("\nDaftar top terms disimpan ke 'top_terms_per_cluster.txt'")

# ==================================================================
# 10. MAPPING CLUSTER -> NAMA KATEGORI (WAJIB DIISI MANUAL)
# ==================================================================
# >>> LIHAT hasil print 'TOP TERMS PER CLUSTER' di atas, lalu isi mapping ini <<<
CLUSTER_NAME_MAP = {
    0: "Keluhan Umum/Tidak Spesifik",
    1: "Keluhan Ekstrem (Makian)",
    2: "Kesulitan Bayar/Lapor Pajak",
    3: "Kualitas Aplikasi Buruk (General)",
    4: "Login",
    5: "Verifikasi/OTP",
    6: "Error/Crash Aplikasi",
    7: "Pendaftaran/Registrasi NPWP",
    8: "Error/Crash Aplikasi",
    9: "Lapor SPT Tahunan Sulit",
}

if CLUSTER_NAME_MAP:
    df_complaint["category"] = df_complaint["cluster"].map(CLUSTER_NAME_MAP).fillna("Belum Diberi Nama")
else:
    df_complaint["category"] = "Cluster " + df_complaint["cluster"].astype(str)
    print("\n[INFO] CLUSTER_NAME_MAP masih kosong. Kolom 'category' sementara diisi 'Cluster 0', 'Cluster 1', dst.")
    print("[INFO] Edit CLUSTER_NAME_MAP di script ini berdasarkan hasil top terms, lalu jalankan ulang.")

# Ulasan positif langsung diberi category tetap, tanpa cluster
df_positive["cluster"] = -1
df_positive["category"] = "Positif/Tidak Ada Keluhan"

# Gabungkan lagi jadi satu dataframe utuh
df = pd.concat([df_complaint, df_positive], ignore_index=True)

# ==================================================================
# 11. VALIDASI: RULE-BASED CATEGORY vs CLUSTER (Confusion Matrix)
# ==================================================================
print("\n" + "=" * 60)
print("11. VALIDASI CLUSTER VS RULE-BASED KEYWORD (Confusion Matrix)")
print("=" * 60)

RULE_CATEGORY_KEYWORDS = {
    "Verifikasi/OTP": ["otp", "verifikasi", "sms", "telepon"],
    "Sync Data": ["sinkron", "sync", "data tidak update", "data tidak sesuai", "data kosong"],
    "Login": ["login", "masuk", "password", "kata sandi", "pendaftaran", "daftar akun"],
    "Loading/Performa": ["lemot", "lama", "hang", "loading", "lambat", "force close", "crash", "macet"],
}

def classify_rule_based(text):
    text_lower = str(text).lower()
    for category, keywords in RULE_CATEGORY_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return category
    return "Lainnya"

df["rule_category"] = df["content"].apply(classify_rule_based)

# Confusion matrix hanya relevan untuk subset yang di-cluster
complaint_mask = df["cluster"] != -1
confusion = pd.crosstab(df.loc[complaint_mask, "rule_category"], df.loc[complaint_mask, "cluster"])
print("\nTabel silang (rule_category vs cluster, subset komplain):")
print(confusion.to_string())

plt.figure(figsize=(8, 6))
plt.imshow(confusion.values, cmap="Blues", aspect="auto")
plt.colorbar(label="Jumlah ulasan")
plt.xticks(range(len(confusion.columns)), [f"Cluster {c}" for c in confusion.columns])
plt.yticks(range(len(confusion.index)), confusion.index)
plt.xlabel("Hasil K-Means")
plt.ylabel("Rule-based Category")
plt.title("Confusion Matrix: Rule-based vs K-Means Cluster")
for i in range(confusion.shape[0]):
    for j in range(confusion.shape[1]):
        plt.text(j, i, confusion.values[i, j], ha="center", va="center",
                  color="white" if confusion.values[i, j] > confusion.values.max()/2 else "black")
plt.tight_layout()
plt.savefig("confusion_matrix_cluster_vs_rule.png", dpi=150)
plt.close()
print("\nHeatmap confusion matrix disimpan ke 'confusion_matrix_cluster_vs_rule.png'")

# ==================================================================
# 12. BOXPLOT: Distribusi Rating & Panjang Komentar per Cluster
# ==================================================================
print("\n" + "=" * 60)
print("12. BOXPLOT PER CLUSTER (subset komplain)")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
df_c = df[complaint_mask]
cluster_ids_sorted = sorted(df_c["cluster"].unique())

data_score = [df_c[df_c["cluster"] == c]["adjusted_score"].values for c in cluster_ids_sorted]
axes[0].boxplot(data_score)
axes[0].set_xticks(range(1, len(cluster_ids_sorted) + 1))
axes[0].set_xticklabels([f"C{c}" for c in cluster_ids_sorted])
axes[0].set_title("Distribusi adjusted_score per Cluster")
axes[0].set_xlabel("Cluster")
axes[0].set_ylabel("adjusted_score")

data_wc = [df_c[df_c["cluster"] == c]["word_count"].values for c in cluster_ids_sorted]
axes[1].boxplot(data_wc)
axes[1].set_xticks(range(1, len(cluster_ids_sorted) + 1))
axes[1].set_xticklabels([f"C{c}" for c in cluster_ids_sorted])
axes[1].set_title("Distribusi Panjang Komentar per Cluster")
axes[1].set_xlabel("Cluster")
axes[1].set_ylabel("Jumlah kata")

plt.tight_layout()
plt.savefig("boxplot_per_cluster.png", dpi=150)
plt.close()
print("Boxplot disimpan ke 'boxplot_per_cluster.png'")

# ==================================================================
# 13. RINGKASAN STATISTIK KATEGORI (full data: komplain + positif)
# ==================================================================
print("\n" + "=" * 60)
print("13. RINGKASAN STATISTIK KATEGORI")
print("=" * 60)
category_summary = (
    df["category"].value_counts().rename_axis("category").reset_index(name="jumlah")
)
category_summary["persentase (%)"] = (category_summary["jumlah"] / len(df) * 100).round(2)
print(category_summary.to_string(index=False))

# ==================================================================
# 14. RINGKASAN DISTRIBUSI RATING ASLI (score) & adjusted_score
# ==================================================================
print("\n" + "=" * 60)
print("14. RINGKASAN DISTRIBUSI RATING (score asli vs adjusted_score)")
print("=" * 60)

score_summary = (
    df["score"].value_counts().sort_index().rename_axis("rating").reset_index(name="jumlah")
)
score_summary["persentase (%)"] = (score_summary["jumlah"] / len(df) * 100).round(2)
print("\n-- Rating ASLI (score) --")
print(score_summary.to_string(index=False))

adj_summary = (
    df["adjusted_score"].value_counts().sort_index().rename_axis("adjusted_rating").reset_index(name="jumlah")
)
adj_summary["persentase (%)"] = (adj_summary["jumlah"] / len(df) * 100).round(2)
print("\n-- Rating SETELAH ADJUSTED (score 5 + komentar teknis -> jadi 1) --")
print(adj_summary.to_string(index=False))

jumlah_dikonversi = ((df["score"] == 5) & (df["adjusted_score"] == 1)).sum()
print(f"\nJumlah ulasan bintang 5 yang dikonversi jadi adjusted_score=1: {jumlah_dikonversi} "
      f"({jumlah_dikonversi/len(df)*100:.2f}% dari total data)")

# ==================================================================
# 15. PREVIEW & EXPORT
# ==================================================================
print("\n" + "=" * 60)
print("15. PREVIEW 10 BARIS PERTAMA")
print("=" * 60)
preview_cols = ["content", "score", "adjusted_score", "cluster", "category"]
print(df[preview_cols].head(10).to_string(index=False))

export_cols = ["reviewId", "userName", "content", "score", "adjusted_score",
               "cluster", "category", "at", "appVersion"]
export_cols = [c for c in export_cols if c in df.columns]
df[export_cols].to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print(f"\nSelesai! Dataset final disimpan ke '{OUTPUT_FILE}' ({len(df)} baris).")

# ==================================================================
# 16. KESIMPULAN OTOMATIS
# ==================================================================
print("\n" + "=" * 60)
print("16. KESIMPULAN")
print("=" * 60)

kesimpulan = []

cluster_terbesar = cluster_dist.idxmax()
pct_terbesar = cluster_dist.max() / len(df_complaint) * 100
top_words_terbesar = [terms[i] for i in order_centroids[cluster_terbesar, :5]]
kesimpulan.append(
    f"1. Di antara ulasan bermasalah, cluster paling dominan adalah Cluster {cluster_terbesar} "
    f"({cluster_dist.max()} ulasan, {pct_terbesar:.2f}% dari subset komplain), "
    f"dengan kata kunci utama: {', '.join(top_words_terbesar)}."
)

avg_score_per_cluster = df_complaint.groupby("cluster")["adjusted_score"].mean().sort_values()
cluster_terburuk = avg_score_per_cluster.index[0]
skor_terburuk = avg_score_per_cluster.iloc[0]
top_words_terburuk = [terms[i] for i in order_centroids[cluster_terburuk, :5]]
kesimpulan.append(
    f"2. Cluster dengan rata-rata adjusted_score paling rendah adalah Cluster {cluster_terburuk} "
    f"(rata-rata {skor_terburuk:.2f}), didominasi kata: {', '.join(top_words_terburuk)}."
)

kesimpulan.append(
    f"3. Ditemukan {jumlah_dikonversi} ulasan ({jumlah_dikonversi/len(df)*100:.2f}%) berating 5 bintang "
    f"namun isi komentarnya mengindikasikan keluhan teknis, sehingga adjusted_score diturunkan menjadi 1."
)

total_keluhan = (df["adjusted_score"] <= 2).sum()
kesimpulan.append(
    f"4. Secara keseluruhan, {total_keluhan} ulasan ({total_keluhan/len(df)*100:.2f}%) memiliki "
    f"adjusted_score rendah (<=2)."
)

kesimpulan.append(
    f"5. Dari total {len(df)} ulasan, {len(df_complaint)} ({len(df_complaint)/len(df)*100:.2f}%) "
    f"terindikasi ada masalah dan diproses lewat clustering, sedangkan {len(df_positive)} "
    f"({len(df_positive)/len(df)*100:.2f}%) murni positif tanpa indikasi keluhan teknis."
)

for k in kesimpulan:
    print(k)

with open("kesimpulan.txt", "w", encoding="utf-8") as f:
    f.write("KESIMPULAN ANALISIS ULASAN M-PAJAK\n")
    f.write("=" * 40 + "\n\n")
    f.write("\n\n".join(kesimpulan))
    f.write(f"\n\nCatatan: Jumlah cluster (K) ditentukan otomatis = {best_k} "
             "berdasarkan silhouette score tertinggi, dihitung hanya dari subset ulasan bermasalah.\n")
    f.write("Nama kategori tiap cluster masih perlu divalidasi/diisi manual di CLUSTER_NAME_MAP "
             "berdasarkan file 'top_terms_per_cluster.txt'.\n")

print("\nKesimpulan disimpan ke 'kesimpulan.txt'")