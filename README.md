# Deteksi Transaksi Fraud Kartu Kredit Menggunakan Metode Klasifikasi

Proyek Akhir Praktikum Data Science 2026/2027, kelas IF-A, Program Studi Informatika, Universitas Pembangunan Nasional "Veteran" Yogyakarta.

**Dosen pengampu:** Dhimas Arief Dharmawan.

## Ringkasan

Proyek ini membandingkan **SVM linear dan Random Forest** untuk membantu analis mengenali transaksi kartu kredit yang berisiko fraud. Model menghasilkan penilaian yang dapat mendukung prioritas pemeriksaan transaksi. Dampak finansial pada operasional bank belum diukur.

Notebook utama adalah [03_proyek_akhir_svm_random_forest.ipynb](03_proyek_akhir_svm_random_forest.ipynb). Isinya mencakup Business Understanding, Data Gathering, EDA, Cleaning, Preprocessing, Seleksi Fitur, Modeling, Evaluasi, dan Konklusi. Hasil perhitungan, tabel, grafik, dan interpretasinya sudah tersimpan dalam notebook.

Keputusan pemodelan menggunakan **30 fitur prediktor**. **SVM linear dipilih berdasarkan PR-AUC validasi tertinggi**, sedangkan Random Forest memperoleh PR-AUC dan recall lebih tinggi pada sampel uji. Kedua hasil dilaporkan beserta batasannya.

## Anggota tim dan kontribusi

| Nama | NIM | Kontribusi |
|---|---|---|
| Farabian Nabil Alauzi | 123240058 | Perlu dilengkapi sesuai pekerjaan aktual. |
| Waladi Lintang Novianto | 123240065 | Perlu dilengkapi sesuai pekerjaan aktual. |
| Raflin Madani | 123240078 | Perlu dilengkapi sesuai pekerjaan aktual. |
| Raihan Buono Putra | 123240090 | Perlu dilengkapi sesuai pekerjaan aktual. |

Pembagian kontribusi perlu dikonfirmasi oleh anggota tim sebelum pengumpulan akhir.

## Pertanyaan proyek

1. Bagaimana memahami masalah bisnis dan kebutuhan pihak terkait dalam mendeteksi fraud kartu kredit?
2. Bagaimana kualitas dan karakteristik data memengaruhi keputusan cleaning dan preprocessing?
3. Bagaimana perubahan jumlah fitur memengaruhi PR-AUC Random Forest?
4. Bagaimana perbandingan SVM linear dan Random Forest dalam menemukan fraud serta membatasi alarm keliru?

## Sumber data dan ketentuan penggunaan

- **Dataset:** [Credit Card Fraud Detection, Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
- **Penerbit:** Machine Learning Group, Université Libre de Bruxelles (ULB).
- **Cakupan:** transaksi kartu kredit pemegang kartu di Eropa selama dua hari pada September 2013.
- **Dataset penuh:** 284.807 transaksi, 492 fraud dan 284.315 normal, dengan proporsi fraud sekitar 0,173%.
- **Metadata lisensi Kaggle:** `Database: Open Database, Contents: Database Contents`. Ketentuan penggunaan dan atribusi mengikuti halaman dataset.
- **Sitasi:** Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, dan Gianluca Bontempi. *Calibrating Probability with Undersampling for Unbalanced Classification*. IEEE Symposium on Computational Intelligence and Data Mining, 2015.

File [creditcard_1000.csv](creditcard_1000.csv) merupakan subset kerja yang berisi seluruh 492 fraud dan 508 transaksi normal yang dipilih acak dengan seed 42. Proporsi fraud pada subset adalah **49,2%**. Sampling ini memperkaya kelas fraud, sehingga hasil pada subset perlu dibaca sesuai komposisinya.

### Data dictionary

| Kolom | Tipe / unit | Deskripsi |
|---|---|---|
| `Time` | Numerik, detik | Waktu relatif sejak transaksi pertama pada dataset. |
| `Amount` | Numerik | Nilai transaksi. Satuan mata uang tidak dinyatakan pada file CSV. |
| `V1`-`V28` | Numerik | Komponen hasil transformasi PCA. Makna bisnis asli dan unitnya tidak tersedia. |
| `Class` | Biner | Label target, 0 untuk normal dan 1 untuk fraud. |

Totalnya **31 kolom**, terdiri dari 30 fitur input dan satu kolom target `Class`.

## Struktur repository

| File / folder | Fungsi |
|---|---|
| [03_proyek_akhir_svm_random_forest.ipynb](03_proyek_akhir_svm_random_forest.ipynb) | Notebook utama proyek akhir, memuat seluruh alur dari pemahaman bisnis hingga evaluasi. |
| [creditcard_1000.csv](creditcard_1000.csv) | Dataset kerja utama untuk menjalankan alur klasifikasi. |
| `requirements.txt` | Daftar dependensi pustaka Python. |
| `rujukan/panduan proyek/` | Panduan proyek akhir dan rubrik penilaian. |
| `rujukan/ppt/` | Berkas slide presentasi Progress 1 dan Progress 2. |
| `rujukan/modul/` | Modul praktikum SVM, Random Forest, dan regresi. |

Folder `rujukan` perlu disertakan saat membagikan repository agar tautan rujukan di notebook dapat dibuka. Notebook utama sudah memuat eksperimen seleksi fitur secara lengkap.

## Cara menjalankan

### 1. Siapkan lingkungan

Gunakan Python dan Jupyter Notebook/Lab. Unduh atau clone repository, lalu buka terminal pada folder proyek.

```bash
git clone https://github.com/RaflinMadani/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
python -m pip install -r requirements.txt
python -m jupyter notebook
```

Library yang digunakan adalah pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, dan IPython. Hasil tersimpan yang dilaporkan di bawah diperoleh pada Python 3.12 dan scikit-learn 1.9.1. Versi dependensi dalam `requirements.txt` dapat berbeda dari lingkungan pengujian tersebut; angka hasil dapat sedikit berubah antarversi library.

### 2. Jalankan notebook utama

1. Letakkan `creditcard_1000.csv` pada folder yang sama dengan notebook utama.
2. Buka `03_proyek_akhir_svm_random_forest.ipynb`.
3. Pilih **Restart Kernel and Run All Cells** atau **Run All** dari awal sampai akhir.
4. Baca penjelasan Markdown, tabel, dan grafik pada setiap bab. Satu eksekusi penuh memastikan variabel evaluasi terbentuk sebelum cell interpretasi dijalankan.

Notebook utama membaca CSV kerja dan membangun seluruh tahapan yang dibutuhkan dalam satu file.

### 3. Studi representativitas 100.000 transaksi

Analisis ini memerlukan file `creditcard.csv` lengkap yang dapat diunduh dari sumber Kaggle. Letakkan file tersebut pada salah satu lokasi berikut.

```text
Credit-Card-Fraud-Detection/archive/creditcard.csv
Credit-Card-Fraud-Detection/creditcard.csv
archive/creditcard.csv di folder induk repository
```

Jika file lengkap tersedia, notebook mengambil 100.000 transaksi secara acak dengan seed 42, lalu menghitung proporsi kelas, interval Wilson, SMD, dan statistik KS. Pada eksekusi yang tersimpan, **SMD absolut maksimum 0,0044** dan seluruh variabel memiliki |SMD| di bawah 0,10.

Jika file lengkap belum tersedia, cell menampilkan keterangan bahwa studi tersebut dilewati, kemudian analisis utama melanjutkan subset kerja. Studi 100.000 transaksi ini terpisah dari subset 1.000 transaksi yang diperkaya fraud.

## Alur pengerjaan dan keputusan

### Cleaning dan pembagian data

Audit dataset kerja menemukan **0 missing value** dan **19 baris duplikat berlebih**, seluruhnya berlabel fraud. Kelebihan baris identik dihapus agar salinan transaksi yang sama tidak tersebar antara data latih dan uji. Kandidat outlier dan transaksi `Amount = 0` dipertahankan karena data sumber belum memberikan bukti bahwa nilainya salah.

| Indikator | Sebelum cleaning | Sesudah cleaning |
|---|---:|---:|
| Jumlah transaksi | 1.000 | 981 |
| Normal | 508 | 508 |
| Fraud | 492 | 473 |
| Missing value | 0 | 0 |
| Duplikat berlebih | 19 | 0 |
| Kandidat outlier `Amount` | 116 | 113 |
| `Amount = 0` | 33 | 31 |

Data bersih dibagi secara **stratified 80:20** dengan seed 42, menghasilkan **784 transaksi latih dan 197 transaksi uji**. Data latih dipakai untuk memilih fitur, membandingkan model, dan memilih parameter. Data uji dipakai setelah keputusan tersebut ditetapkan.

### Preprocessing

- Semua prediktor sudah numerik sehingga encoding tambahan tidak diperlukan.
- **SVM** memakai Pipeline dengan `RobustScaler` untuk `Amount`, serta `StandardScaler` untuk `Time` dan `V1`-`V28`.
- **Random Forest** menggunakan nilai numerik asli.
- Parameter scaling dipelajari dari bagian latih pada setiap fold validasi.
- Rasio data bersih sekitar 51,78% normal dan 48,22% fraud. Eksperimen ini menggunakan `class_weight=None` dan tanpa resampling tambahan.

### Seleksi fitur

Tiga set fitur dibandingkan menggunakan Random Forest dan pembagian validasi yang sama. Top 10 dipilih berdasarkan korelasi Pearson absolut terhadap `Class` pada bagian latih masing-masing fold. Kandidat yang memiliki korelasi antarfitur absolut minimal 0,95 ditinjau untuk pengurangan.

| Set fitur | PR-AUC validasi rata-rata |
|---|---:|
| 30 fitur lengkap | **0,9822** |
| 10 kandidat | 0,9804 |
| 9 fitur setelah pengurangan | 0,9798 |

Versi ringkas memakai `V14`, `V4`, `V12`, `V11`, `V10`, `V16`, `V3`, `V9`, dan `V2`. `V17` dikeluarkan dari versi ini karena korelasinya dengan `V16` mencapai sekitar 0,955. `V16` dipertahankan karena korelasinya dengan `Class` lebih kuat.

**Keputusan pemodelan mempertahankan 30 fitur**, termasuk `Time` dan `Amount`, karena set lengkap memperoleh PR-AUC validasi tertinggi dalam eksperimen tersebut. Korelasi pada bab EDA dihitung dari 1.000 transaksi awal, sedangkan seleksi dipelajari dari data latih. Perbedaan basis perhitungan dapat menghasilkan angka korelasi yang sedikit berbeda.

### Model dan pencarian parameter

SVM merujuk pada Modul 6 dan Random Forest merujuk pada Modul 7 praktikum. Kedua model memakai 30 fitur dan lima fold stratified yang sama. DummyClassifier dengan skor konstan menjadi baseline.

| Model | Konfigurasi awal | Ruang parameter yang diuji | Konfigurasi terpilih |
|---|---|---|---|
| SVM linear | `kernel='linear'`, `C=1` | `C` = 0,1; 1; 10 | `C=0.1` |
| Random Forest | 100 pohon, kedalaman maksimal 5 | 100/300 pohon; kedalaman 5/`None` | 300 pohon, `max_depth=None` |

GridSearchCV memilih parameter berdasarkan rata-rata PR-AUC validasi pada 784 transaksi latih. Selisih skor dibaca bersama variasi antar-fold dan keterbatasan ukuran sampel.

## Hasil dan pemilihan model

| Metrik | SVM linear | Random Forest |
|---|---:|---:|
| PR-AUC validasi terbaik | **0,9849** | 0,9822 |
| Simpangan baku PR-AUC antar-fold | 0,0058 | 0,0056 |
| PR-AUC data uji | 0,9813 | **0,9852** |
| Precision fraud pada data uji | **98,80%** | 97,67% |
| Recall fraud pada data uji | 86,32% | **88,42%** |
| F1 fraud pada data uji | 0,9213 | **0,9282** |
| Fraud ditemukan (TP) | 82 | **84** |
| Fraud terlewat (FN) | 13 | **11** |
| Normal salah diberi alarm (FP) | **1** | 2 |

**Model terpilih adalah SVM linear berdasarkan validasi data latih.** Pada sampel uji, Random Forest menemukan dua fraud lebih banyak dengan satu alarm keliru tambahan. Hasil kedua model tetap dilaporkan; keputusan pemilihan mengikuti prosedur validasi sebelum pengujian. Selisih kecil pada sampel ini belum menetapkan keunggulan umum salah satu algoritma.

PR-AUC pada notebook dihitung menggunakan integrasi trapesium kurva precision-recall. Average Precision (AP) dilaporkan terpisah. Precision, recall, F1, dan confusion matrix menggunakan ambang prediksi bawaan model. Mengubah ambang mengubah keputusan kelas dan pasangan precision/recall pada titik tertentu; kurva PR dan luasnya tetap mengikuti skor prediksi model.

Grafik pendukung pada notebook mencakup confusion matrix kedua model, kurva precision-recall, serta feature importance Random Forest. Feature importance dibaca sebagai kontribusi terhadap pembagian data pada pohon. Fitur berkorelasi dapat berbagi nilai importance dan makna bisnis asli fitur PCA tetap tidak tersedia.

## Keterbatasan dan tindak lanjut

- Komposisi fraud pada sampel uji mendekati 48%, jauh lebih tinggi daripada sekitar 0,173% pada dataset penuh. Precision dan PR-AUC tersebut perlu dibaca sesuai distribusi sampel.
- Dataset mencakup dua hari pada September 2013 di Eropa. Pola transaksi, wilayah, dan waktu lain memerlukan pengujian tambahan.
- Identitas kartu dan atribut sensitif tidak tersedia. Ketergantungan antartransaksi dan fairness kelompok tidak dapat diperiksa secara lengkap.
- Label fraud mengikuti penyedia dataset dan belum diverifikasi ulang.
- Evaluasi berbasis biaya kerugian, koreksi prevalensi operasional, dan pemilihan threshold bisnis menjadi pekerjaan lanjutan. Proyek belum mengukur penghematan finansial pada bank.

## Rujukan dan kelengkapan pengumpulan

Panduan proyek akhir meminta notebook lengkap sampai konklusi, dataset, serta README dengan ringkasan, anggota dan kontribusi, sumber data, dan cara menjalankan. Notebook utama memuat tabel pemetaan bab terhadap panduan serta tautan ke PPT dan modul di folder `rujukan`.

Seluruh keluaran analisis, perbandingan model, tabel evaluasi, visualisasi grafik, dan konklusi tersimpan lengkap di dalam file notebook utama [03_proyek_akhir_svm_random_forest.ipynb](03_proyek_akhir_svm_random_forest.ipynb).

Dokumentasi teknis pendukung tersedia pada [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html), [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html), [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), dan [pemilihan threshold klasifikasi](https://scikit-learn.org/stable/modules/classification_threshold.html).
