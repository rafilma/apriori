# Laporan Proyek Machine Learning
### Nama : Rafil Moehamad Alif
### Nim : 211351116
### Kelas : Malam B

## Domain Proyek

Didasarkan pada dataset di kaggle yang mengambil contoh kasus di Toko Sukses Makmur Sentosa yang mana memiliki banyak pelanggan tetap yang membeli item yang sama secara berulang. Dalam kasus ini Toko Sukses Makmur Sentosa menjual 5 item yang secara prepetual menjadi primadona bagi para costumernya

## Business Understanding

Seperti di jelaskan di atas, ada 5 item yang menjadi pilihan customer ketika berbelanja di Toko Sukses Makmur Sentosa, sejatinya, tujuan di buatnya contoh model aplikasi ini agar si pengelola dapat menetukan Stock Barang, Diskon barang, dan bundle paket barang untuk customernya, karena dengan adanya aplikasi ini yang di buat dengan association rules dan algoritma apriori si pengelola di harapkan dapat memahami ketika si customer membeli suatu item, maka item lainya pasti di beli dengan menentukan nilai support nya.

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Dikarenakan Toko Sukses Makmur Sentosa adalah toko kecil, sehingga kadang pemilik warung kebingunan untuk menentukan stok item apa yang harus di perbanyak, membuat paket penjualan yang dapat menguntungkan dan memberi diskon pada customer yang menjadi langganan

### Goals
Menjelaskan tujuan dari pernyataan masalah:
- Dengan adanya contoh aplikasi ini, diharapkan pemilik Toko Sukses Makmur Sentosa dapat menentukan Stok barang yang harus di perbanyak, membuat paket bundle penjualan sesuai kebutuhan yang di dasari pada perhitungan nilai support.



## Data Understanding
Data ini diambil dari data penjualan Toko Sukses Makmur Sentosa yang terdiri dari 1289 transaksi dan 5 item yang berbeda serta 171 unique customer.

Contoh: [Penjualan Toko Sukses Makmur Sentosa](https://www.kaggle.com/datasets/bejopamungkas/transaksi-pembelian-penjualan-sembako).
 

### Variabel-variabel pada Dataset tersebut adalah sebagai berikut:
- nama.pembeli (bertipe string, unique name dari setiap customer)
- nama.barang (bertipe string, list item terjual dari toko tersebut)
- tanggal (bertipe datetime, tanggal terjadi nya transaksi dengan format Tahun-Bulan-Tanggal)
- Kuantum (bertipe integer, kuantitas barang yang di beli namun tidak digunakan pada algoritma ini)
- nominal (bertipe float, jumlah harga pada item yang di beli namun tidak digunakan pada algoritma ini)


## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

Pertama kita persiapkan dataset nya terlebih dahulu, jika sudah bisa lanjut kedalam import library
```bash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import networkx as nx
import plotly.graph_objects as go
from mlxtend.frequent_patterns import association_rules, apriori
import warnings
warnings.filterwarnings('ignore')
```
selanjutnya kita mount data tersebut
```bash
df=pd.read_csv('/content/project_uas/penjualan barang.csv')
```
maka ketika kita mengetik df akan muncul dataset yang kita pilih
```bash
df.head()
```
![Screenshot (85)](https://github.com/rafilma/apriori/assets/148635738/0127051b-bf00-4699-9156-0e1b1fddb469)


Selanjutnya kita bisa eksplore data tersebut seperti contoh :
```bash
df['nama.barang'].value_counts()
```
```bash
BERAS     836
DAGING    184
GULA      121
TEPUNG     77
MIGOR      71
Name: nama.barang, dtype: int64
```
Atau kita bisa mengecek apakah ada data kosong pada dataframe kita
```bash
df.isnull().sum()
```
```bash
Unnamed: 0      0
tanggal         0
nama.pembeli    0
nama.barang     0
kuantum         0
nominal         0
dtype: int64
```
Jika tidak ada data kosong, kita bisa lanjutkan sampai semua informasi yang kita butuhkan tercukupkan atau kita lanjutkan dalam visualisasi data

Berikut contoh visualisasi data :
```bash
plt.rcParams.update({'font.size': 12})
ax=df['nama.barang'].value_counts().plot.pie(autopct='%1.2f%%',shadow=True)
ax.set_title(label = "Penjualan Barang", fontsize = 24,color='Blue')
plt.axis('off')
```
![image](https://github.com/rafilma/apriori/assets/148635738/3618e1c0-4ef9-4a5d-9fd3-ae3dd3cfd24c)
 Maka kita bisa lihat persentase penjualan barang.
```bash
sns.countplot(x=df['nama.barang'], data=df)
plt.show()
```
![image](https://github.com/rafilma/apriori/assets/148635738/4a59bbce-cec8-4ad0-ac05-6c34e90d14cd)
kita bisa lihat dalam tampilan Chart nya

selanjutnya kita bisa melihat top customer
```bash
sns.countplot(x=df['nama.pembeli'], data=df)
plt.show()
```
![image](https://github.com/rafilma/apriori/assets/148635738/ac139a43-147a-4e70-bbfe-8cdd95f05aa2)

Karena jumlah customernya terlalu banyak dan visualisasi susah terbaca, maka kita coba pecah menjadi top 10 saja.

```bash
top_pembeli = df['nama.pembeli'].value_counts().nlargest(10)

plt.figure(figsize=(12, 6))
sns.countplot(x=df['nama.pembeli'], data=df, order=top_pembeli.index, palette='viridis')
plt.xlabel('Nama Pembeli')
plt.ylabel('Count')
plt.title('Top 10 Pembeli')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.show()
```

![image](https://github.com/rafilma/apriori/assets/148635738/70f2999d-215f-45db-871c-48ad0fad5fc5)

Kita coba pecah format tanggal yang ada menjadi kolom masing masing masing 
```bash
df['tanggal'] = pd.to_datetime(df['tanggal'], format= "%Y-%m-%d")
```
```bash
df["month"] = df['tanggal'].dt.month
df["day"] = df['tanggal'].dt.day
df["year"] = df['tanggal'].dt.year
df.head()
```
![Screenshot (86)](https://github.com/rafilma/apriori/assets/148635738/bd0840e9-521c-4590-a4f2-0262d3375772)

Terlihat kolom tanggal sudah terpecah
sekarang kita bisa menentukan penjualan berdasarkan tanggal
```bash
plt.figure(figsize=(8,5))
sns.countplot(x='day',data=df)
plt.title('Penjualan Item berdasarkan tanggal')
plt.show()
```
![image](https://github.com/rafilma/apriori/assets/148635738/c4c7e7c0-a849-48d4-adf1-d00c0db9c208)

atau kita bisa lihat penjualan berdasarkan bulan
```bash
plt.figure(figsize=(8,5))
sns.countplot(x='month',data=df)
plt.title('Penjualan Item berdasarkan bulan')
plt.show()
```
![image](https://github.com/rafilma/apriori/assets/148635738/c3148117-116e-4531-a949-329e52e5ec0c)


Selanjutnya kita coba satukan kolom nama pembeli dan nama barang saja
```bash
df["nama.barang"] = df["nama.barang"].apply(lambda item: item.lower())
df["nama.barang"] = df["nama.barang"].apply(lambda item: item.strip())
df = df[["nama.pembeli", "nama.barang"]].copy()
df.head()
```




## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

## Deployment
[Apriori](https://apriori-uas.streamlit.app/)
![image](https://github.com/rafilma/apriori/assets/148635738/55079ee0-1f04-4a81-b4d1-edf0c908ea23)


**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

