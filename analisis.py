import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import pandas as pd

# Baca file CSV
df = pd.read_csv('data_penjualan.csv')
# Periksa nilai yang hilang
print(df.isnull().sum())

# Periksa duplikat
print(df.duplicated().sum())

# Penanganan nilai yang hilang atau duplikat
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
import matplotlib.pyplot as plt
import seaborn as sns

# Misalnya, visualisasikan distribusi jumlah_pesanan
plt.figure(figsize=(10, 6))
sns.histplot(df['jumlah_pesanan'], bins=20, kde=True)
plt.title('Distribusi Jumlah Pesanan')
plt.xlabel('Jumlah Pesanan')
plt.ylabel('Frekuensi')
plt.show()

# Misalnya, lihat tren penjualan dari waktu ke waktu
plt.figure(figsize=(12, 6))
sns.lineplot(x='tanggal_pesanan', y='total_penjualan', data=df, estimator=sum)
plt.title('Tren Penjualan')
plt.xlabel('Tanggal Pesanan')
plt.ylabel('Total Penjualan')
plt.show()
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Pisahkan fitur dan target
X = df[['jumlah_pesanan', 'harga']]
y = df['keuntungan']

# Bagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi dan latih model regresi linear
model = LinearRegression()
model.fit(X_train, y_train)

# Lakukan prediksi
y_pred = model.predict(X_test)

# Evaluasi model
mse = mean_squared_error(y_test, y_pred)
print('MSE:', mse)
# Misalnya, lakukan validasi silang untuk penyetelan parameter
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
print('Cross-validated MSE:', -scores.mean())
# Misalnya, visualisasikan hasil prediksi
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted')
plt.show()

