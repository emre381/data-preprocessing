import pandas as pd 
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# veri yükeleme
df = pd.read_excel('veri_on_isleme_ve_ozellik_muhendisligi.xlsx')

df.fillna(df['Gelir'].mean(),inplace=True)

# cinsiyet ve meslek stunlarını sayısal verilere çevirme 
le=LabelEncoder()
df['Cinsiyet']=le.fit_transform(df['Cinsiyet'])
df['Meslek']=le.fit_transform(df['Meslek'])

# giriş ve çıktı verilerini giriniz
X = df[['Yaş', 'Meslek','Cinsiyet' ]] #GİRDİ
y =  df['Gelir'] #ÇIKTI

# veri setini 80/20 oranında train ve test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

# ölçeklendirme 
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test) #test verisini çevirmek bir hata olur zaten test 

# # modeli oluştur ve eğit
linear_model = LinearRegression()
linear_model.fit(X_train,y_train)

# Modeli test ve eğit 
y_pred = linear_model.predict(X_test)

# performansını değerlendir
mse=mean_squared_error(y_test,y_pred)
# performası değerlendir
rmse=mse**0.5 # kök ortalama
print(f"Linear reg rmse: {rmse:.2f}")

# daha karmaşık model ile  tekrar eğit
rf_model = RandomForestRegressor(n_estimators=100,random_state=1)
rf_model.fit(X_train,y_train)
y_pred_rf = rf_model.predict(X_test)
mse_rf=mean_squared_error(y_test,y_pred_rf) # bu işlemlerin printlerinde hata skorlarını görüyoruz ve sonuç olarak rand forest kullanınca daha az hata skoru elde ettik 
rmse_rf=mse_rf**0.5
print(f"Random Fores rmse: {rmse_rf:.2f}") 





# # # modeli test et
# linear_accuracy = linear_model.score(X_test,y_test)
# print(f"Modelin doğruluk oranı: {linear_accuracy*100:.2f}%")


# # daha karmaşık bir model kullan 

# rf_model=RandomForestRegressor(n_estimators=80,random_state=42) #n estimator 100 karar ağacı oluştur
# rf_model.fit(X_train,y_train) #modeli eğit
# rf_accuracy = rf_model.score(X_test,y_test)
# print(f"Modelin doğruluk oranı: {rf_accuracy*100:.2f}%")
# # # kullanıcıdan yaş ve meslek giridisini alma 
# print("Lütfen tahmin için bilgilerinizi giriniz:")
# yas= int(input("Yaş:"))
# meslek= input("meslek(Mühendis, Doktor, Öğretmen, Avukat):")
# cinsiyet=input("Cinsiyety(Erkek,Kadın):")
# # # kullanıcıdan alınan mesleği kodlayın (Label encoding)
# if cinsiyet =='Erkek':
#     cinsiyet_kod=0
# elif cinsiyet =='Kadın':
#     cinsiyet_kod=1
# else:
#     raise ValueError("Geçersiz cinsiyet değeri")

# meslek_kod=le.transform([meslek])[0]
# yeni_veri=pd.DataFrame([[yas,meslek_kod,cinsiyet_kod]],columns=['Yaş','Meslek','Cinsiyet'])
# yeni_veri_scaled=scaler.transform(yeni_veri) # burda randforest kullandığımız için veriyi scaler e çeviriyoruz

# tahmin=rf_model.predict(yeni_veri_scaled)
# print(f'tahmini ortalam maas: {tahmin[0]:.2f} TL')