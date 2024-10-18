import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Veri yükleme
df = pd.read_excel('veri_on_isleme_ve_ozellik_muhendisligi.xlsx')

# Eksik verileri doldurma (Gelir sütununda)
df.fillna(df['Gelir'].mean(), inplace=True)

# Meslek ve Cinsiyet sütunlarını sayısal verilere çevirme (Label Encoding)
le_meslek = LabelEncoder()
df['Meslek'] = le_meslek.fit_transform(df['Meslek'])

le_cinsiyet = LabelEncoder()
df['Cinsiyet'] = le_cinsiyet.fit_transform(df['Cinsiyet'])

# Giriş (X) ve Çıkış (y) verilerini belirleme
X = df[['Yaş', 'Meslek', 'Cinsiyet']]  # Girdi
y = df['Gelir']  # Çıktı

# Veri setini %80 eğitim ve %20 test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Lineer regresyon modeli oluşturma ve eğitme
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

# Lineer regresyon modeli ile test sonuçlarını değerlendirme
linear_accuracy = linear_model.score(X_test_scaled, y_test)
print(f"Lineer Regresyon Modelinin doğruluk oranı: {linear_accuracy * 100:.2f}%")

# Random Forest modeli oluşturma ve eğitme
rf_model = RandomForestRegressor(n_estimators=80, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Random Forest modeli ile test sonuçlarını değerlendirme
rf_accuracy = rf_model.score(X_test_scaled, y_test)
print(f"Random Forest Modelinin doğruluk oranı: {rf_accuracy * 100:.2f}%")

# Kullanıcıdan yaş, meslek ve cinsiyet bilgilerini alma
print("Lütfen tahmin için bilgilerinizi giriniz:")
yas = int(input("Yaş:"))
meslek = input("Meslek (Mühendis, Doktor, Öğretmen, Avukat):")
cinsiyet = input("Cinsiyet (Erkek, Kadın):")

# Meslek ve cinsiyeti sayısal değerlere çevirme
meslek_kod = le_meslek.transform([meslek])[0]
cinsiyet_kod = le_cinsiyet.transform([cinsiyet])[0]

# Yeni veriyi DataFrame olarak oluşturma
yeni_veri = pd.DataFrame([[yas, meslek_kod, cinsiyet_kod]], columns=['Yaş', 'Meslek', 'Cinsiyet'])

# Yeni veriyi ölçeklendirme
yeni_veri_scaled = scaler.transform(yeni_veri)

# Random Forest modeli ile tahmin yapma
tahmin = rf_model.predict(yeni_veri_scaled)
print(f'Tahmini ortalama maaş: {tahmin[0]:.2f} TL')
