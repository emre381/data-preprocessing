import pandas  as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# veriyi yükledik
df =pd.read_excel('teknolojik_urunler_zamanli.xlsx')
#  eksik verileri doldurma 
X=df[['Fiyat (TL)','Satış']]
# veriyi ölçeklendir
scaler=StandardScaler()
X_scaled = scaler.fit_transform(X) #ortalama 0 standart sapma 1 diyorum ve hepsi için sabitliyorum 
# Kmeans Modeli
kmeans=KMeans(n_clusters=3,random_state=42)
df['Küme']=kmeans.fit_predict(X_scaled)
# kümeleme

# görselleştirme
plt.figure(figsize=(10,8))
plt.scatter(df['Fiyat (TL)'],df['Satış'],c=df['Küme'],cmap='viridis')
plt.title('ürünler için analiz edilmiş kümeleme')
plt.xlabel("Fiyat (TL)")
plt.ylabel("Satış")
for i in range(len(df)):
    urun_adi_tarih_satis=f"{df['Ürün Adı'][i]} ({df['Tarih'][i].strftime('%d-%m-%Y')}) [{df['Satış'][i]} adet] {df['Fiyat (TL)'][i]} TL "

    plt.text(df['Fiyat (TL)'][i]+185,df['Satış'][i]+1,urun_adi_tarih_satis, fontsize=9,ha='left')
plt.show()
