# Deteksi Transaksi Fraud Kartu Kredit Menggunakan Metode Klasifikasi

Proyek Data Science (IF-A) | Dosen pengampu: Dhimas Arief Dharmawan
Program Studi Informatika, Fakultas Teknik Industri, Universitas Pembangunan Nasional "Veteran" Yogyakarta, 2026

## Ringkasan

Proyek ini bertujuan mengenali transaksi fraud kartu kredit agar pelanggan tidak dibebankan transaksi yang tidak dilakukan. Tahap yang sudah dikerjakan meliputi business understanding, data gathering, exploratory data analysis (EDA), data cleaning, dan preprocessing. Pemodelan, evaluasi model, dan deployment belum dilakukan.

## Anggota Tim

| No | Nama | NIM |
|----|------|-----|
| 1 | Farabian Nabil Alauzi | 123240058 |
| 2 | Waladi Lintang Novianto | 123240065 |
| 3 | Raflin Madani | 123240178 |
| 4 | Raihan Buono Putra | 123240090 |

## Struktur Repositori

- `01_eda_representativitas_bias_uts.ipynb` - EDA, representativitas sampel acak, dan potensi bias (tugas UTS).
- `setupV2.ipynb` - Business understanding, data gathering, EDA, cleaning, dan preprocessing.
- `creditcard_1000.csv` - Dataset kerja (1.000 transaksi).
- `creditcard_1000_clean.csv` - Data bersih sebelum split dan scaling (hasil run notebook).
- `train_preprocessed.csv` - Data latih yang sudah diskalakan (hasil run notebook).
- `test_preprocessed.csv` - Data uji yang sudah diskalakan (hasil run notebook).
- `requirements.txt` - Daftar dependensi Python.
- `README.md` - Dokumentasi proyek.

## Sumber Data

Dataset yang digunakan adalah Credit Card Fraud Detection dari Kaggle, dipublikasikan oleh Machine Learning Group, Universite Libre de Bruxelles (ULB).

- Tautan: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Isi: transaksi kartu kredit pemegang kartu di Eropa selama dua hari pada September 2013.
- Lisensi Kaggle: Database Open Database, Contents Database Contents.
- Sitasi: Andrea Dal Pozzolo dkk., Calibrating Probability with Undersampling for Unbalanced Classification, IEEE CIDM 2015, serta publikasi terkait lainnya yang tercantum pada halaman Kaggle.

## Deskripsi Dataset

Dataset lengkap terdiri dari 284.807 transaksi dengan 31 kolom, di antaranya 492 transaksi fraud (0,172%). Karena sangat tidak seimbang, akurasi biasa tidak bermakna; disarankan menggunakan PR-AUC.

Catatan sampling pada repo ini: dataset kerja (`creditcard_1000.csv`) adalah subset 1.000 transaksi yang terdiri dari seluruh 492 transaksi fraud ditambah 508 transaksi normal yang dipilih secara acak (`random_state=42`). Duplikat bawaan dataset asli tidak dihapus agar dapat dinilai pada tahap cleaning. Subset ini hampir seimbang, sehingga hasil EDA dan model dari subset ini tidak boleh langsung dianggap mewakili kondisi nyata di mana proporsi fraud sekitar 0,17%.

Data dictionary ringkas:

| Kolom | Tipe | Deskripsi |
|-------|------|-----------|
| `Time` | Numerik | Detik sejak transaksi pertama dalam dataset |
| `V1` - `V28` | Numerik | Fitur hasil transformasi PCA (dianonimkan) |
| `Amount` | Numerik | Nominal transaksi (satuan mata uang tidak dinyatakan) |
| `Class` | Biner | 0 = normal, 1 = fraud |

Keterbatasan data: fitur `V1`-`V28` dianonimkan sehingga makna bisnisnya tidak dapat ditentukan; tidak ada ID transaksi; satuan mata uang pada `Amount` tidak dinyatakan; atribut kelompok sensitif tidak tersedia sehingga audit fairness terbatas.

## Cakupan Analisis per Notebook

- `setupV2.ipynb`: business understanding, data gathering, EDA, cleaning, preprocessing.
- `01_eda_representativitas_bias_uts.ipynb`: kualitas data (missing value, duplikasi, outlier), EDA distribusi dan korelasi, representativitas sampel acak, potensi bias dan keterbatasan. Tidak melakukan data preparation untuk modeling.

## Metodologi

1. Data gathering: sumber Kaggle, lisensi, subset 1.000 transaksi.
2. EDA: gambaran umum, distribusi kelas, distribusi `Amount` (berekor panjang ke kanan), distribusi `Time`, korelasi fitur dengan target (10 fitur dengan korelasi absolut terbesar), perbandingan distribusi fitur per kelas, missing value, duplikat, kandidat outlier dengan metode IQR.
3. Cleaning: missing value tidak ditemukan; duplikat dihapus (seluruhnya pada kelas fraud) untuk mencegah kebocoran data antara data latih dan uji; kandidat outlier dipertahankan karena dapat merupakan sinyal fraud; transaksi dengan `Amount = 0` dipertahankan.
4. Preprocessing: pemisahan fitur dan target, split stratified 80:20, `V1`-`V28` tidak diubah, `Amount` diskalakan dengan `RobustScaler`, `Time` dengan `StandardScaler`, scaler di-fit hanya pada data latih, verifikasi bebas kebocoran data.
5. Representativitas sampel: simple random sampling tanpa pengembalian dengan seed 42, indikator standardized mean difference (SMD) dan interval kepercayaan Wilson 95% untuk proporsi fraud, serta catatan bias.
## Temuan Utama

- Tidak ada missing value; terdapat baris duplikat (seluruhnya pada kelas fraud) dan banyak kandidat outlier terutama pada `Amount` dan beberapa fitur PCA.
- Korelasi absolut terbesar terhadap `Class`: `V14` (-0,751), `V4` (0,713), `V11` (0,687), `V12` (-0,685), `V10` (-0,632), `V16` (-0,600), `V3` (-0,567), `V17` (-0,562), `V9` (-0,561), `V2` (0,495).
- Duplikat dihapus; outlier dan transaksi `Amount = 0` dipertahankan.
- Data dibagi secara stratified 80:20 dan siap digunakan untuk tahap pemodelan.

## Prasyarat dan Cara Menjalankan

Prasyarat: Python 3, Jupyter Notebook/Lab atau Google Colab.

Instal dependensi:

    pip install -r requirements.txt

Letakkan `creditcard.csv` (dataset lengkap) pada folder `archive/` atau root proyek jika menjalankan notebook pertama. Notebook kedua menggunakan `creditcard_1000.csv` yang sudah tersedia.

Urutan eksekusi:

1. Jalankan `setupV2.ipynb` dari awal sampai akhir.
2. (Opsional) Jalankan `01_eda_representativitas_bias_uts.ipynb` untuk analisis representativitas dan bias (memerlukan dataset lengkap).

Catatan: notebook pertama menggunakan `SAMPLE_SIZE = 100_000` (rentang yang diizinkan 10.000-100.000).

## Output yang Dihasilkan

- `creditcard_1000_clean.csv` - data bersih sebelum split dan scaling.
- `train_preprocessed.csv` - data latih yang sudah diskalakan.
- `test_preprocessed.csv` - data uji yang sudah diskalakan.

## Keterbatasan dan Bias

- Subset hampir seimbang, sedangkan kondisi nyata memiliki proporsi fraud sekitar 0,17%.
- Fitur `V1`-`V28` dianonimkan, sehingga makna bisnis tidak dapat ditentukan.
- Tidak ada ID transaksi, sehingga baris identik tidak pasti merupakan kesalahan pencatatan.
- Korelasi bersifat statistik, bukan bukti sebab-akibat.
- Atribut kelompok sensitif tidak tersedia, sehingga audit fairness terbatas.
- Potensi kesalahan label, cakupan populasi, dan periode harus diverifikasi dari sumber asli.

Untuk evaluasi akhir disarankan menggunakan precision, recall, F1, PR-AUC, dan biaya kesalahan, bukan akurasi biasa.

## Status Proyek dan Pekerjaan Lanjutan

Tahap saat ini: data understanding, cleaning, dan preprocessing. Belum dilakukan: pemodelan, evaluasi, tuning, dan deployment.

Saran pekerjaan lanjutan: baseline model, cross-validation stratified, penanganan ketidakseimbangan kelas (class weight atau resampling), evaluasi PR-AUC, serta penentuan threshold berbasis biaya kesalahan.

## Lisensi dan Sitasi

Lisensi data mengikuti ketentuan Kaggle. Sitasi utama: Dal Pozzolo, Andrea; Caelen, Olivier; Johnson, Reid A.; Bontempi, Gianluca. Calibrating Probability with Undersampling for Unbalanced Classification. IEEE Symposium on Computational Intelligence and Data Mining (CIDM), 2015. Daftar publikasi lengkap tercantum pada halaman dataset Kaggle.

