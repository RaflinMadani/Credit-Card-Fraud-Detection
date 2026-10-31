# Deteksi Transaksi Fraud Kartu Kredit Menggunakan Metode Klasifikasi

Proyek UTS Data Science IF-A, Program Studi Informatika, Universitas Pembangunan Nasional "Veteran" Yogyakarta.

## Ringkasan Proyek

**Masalah.** Transaksi kartu kredit semakin banyak dilakukan secara digital, sehingga risiko penipuan (fraud) ikut meningkat dan menimbulkan kerugian finansial. Pemeriksaan manual atas seluruh transaksi tidak praktis, sehingga diperlukan pendekatan berbasis data untuk mengidentifikasi transaksi yang terindikasi fraud.

**Pertanyaan.**
1. Bagaimana memahami permasalahan bisnis dan kebutuhan analisis dalam mendeteksi transaksi kartu kredit yang terindikasi fraud?
2. Bagaimana memahami karakteristik, struktur, dan kualitas dataset yang digunakan?

**Metode.** Klasifikasi biner dengan target `Class` (0 = normal, 1 = fraud). Tahap yang dikerjakan saat ini adalah Business Understanding dan Data Preprocessing. Modelling dan evaluasi dikerjakan pada tahap berikutnya.

**Metrik.** PR-AUC sebagai metrik utama, serta precision, recall, dan F1 untuk kelas fraud. Accuracy tidak dijadikan acuan karena kelas sangat tidak seimbang.

## Anggota Kelompok dan Kontribusi

| No | Nama | NIM | Kontribusi |
| --- | --- | --- | --- |
| 1 | Farabian Nabil Alauzi | 123240058 | [isi kontribusi] |
| 2 | Waladi Lintang Novianto | 123240065 | [isi kontribusi] |
| 3 | Raflin Madani | 123240178 | [isi kontribusi] |
| 4 | Raihan Buono Putra | 123240090 | [isi kontribusi] |

## Sumber Data

- **Dataset:** Credit Card Fraud Detection, dipublikasikan oleh Machine Learning Group, Université Libre de Bruxelles (ULB).
- **Tautan:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **Lisensi:** "Database: Open Database, Contents: Database Contents" sesuai halaman Kaggle. Dipakai untuk kepentingan akademik.
- **Berkas di repository:** `creditcard_1000.csv`, berisi 1.000 baris dengan 31 kolom (`Time`, `V1` sampai `V28`, `Amount`, `Class`).

> **Catatan tentang sampel.** Dari 1.000 baris ini, 492 berlabel fraud (49,2%). Pada dataset lengkap, proporsi fraud hanya sekitar 0,17%. Sampel ini tidak mewakili proporsi tersebut, sehingga metrik dan baseline yang dihitung darinya tidak sama dengan kondisi sebenarnya. Sampel ini cocok untuk menguji alur kerja. Untuk menilai kemampuan deteksi fraud, gunakan dataset lengkap.

## Struktur Repository

```
.
+-- 01_business_understanding_preprocessing.ipynb
+-- creditcard_1000.csv
+-- requirements.txt
+-- .gitignore
+-- README.md
```

## Cara Menjalankan

1. Pastikan Python terpasang (proyek dikembangkan dengan Python 3.13.7). Disarankan memakai virtual environment.
2. Pasang dependensi:
   ```
   pip install -r requirements.txt
   ```
3. Jalankan Jupyter dari folder repository ini:
   ```
   jupyter notebook
   ```
4. Buka `01_business_understanding_preprocessing.ipynb` lalu pilih **Run All**. Notebook membaca `creditcard_1000.csv` dari folder yang sama. Folder `processed/` dibuat otomatis berisi hasil preprocessing.

## Ringkasan Preprocessing

- **Pembagian data.** Split 80:20 terstratifikasi dilakukan sebelum transformasi agar tidak terjadi data leakage.
- **Duplikat.** Duplikat dihapus dari data latih. Baris uji yang identik dengan data latih juga dibuang.
- **Outlier.** Dicatat dengan metode IQR tetapi tidak dihapus, karena transaksi fraud sering berada pada nilai ekstrem.
- **Transformasi.** `Amount` memakai `log1p` lalu `RobustScaler`. `Time` memakai `RobustScaler`. `V1` sampai `V28` dibiarkan karena sudah hasil PCA.
- **Pipeline.** Transformasi hanya di-fit pada data latih dan disimpan sebagai `processed/preprocessor.joblib`.

## Batasan

- Dataset hanya memuat transaksi pemegang kartu di Eropa selama dua hari pada September 2013.
- Fitur `V1` sampai `V28` telah dianonimkan dengan PCA sehingga makna bisnisnya tidak dapat dijelaskan secara langsung.
- Tidak tersedia informasi merchant, lokasi, perangkat, atau riwayat pemegang kartu.
- Sampel 1.000 baris tidak mewakili proporsi fraud pada data asli.
- Analisis bersifat offline dan tidak mencakup sistem deteksi real-time.
