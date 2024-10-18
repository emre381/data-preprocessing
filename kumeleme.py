import pandas  as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# veriyi yükledik
df =pd.read_excel('veri_on_isleme_ve_ozellik_muhendisligi.xlsx')
#  eksik verileri doldurma 
df.fillna(df['Gelir'].mean(), inplace=True)
# Kümelemeee
X=df[['Yaş','Gelir']]
# veriyi ölçeklendir
scaler=StandardScaler()
X_scaled = scaler.fit_transform(X)
# Kmeans Modeli
kmeans=KMeans(n_clusters=3,random_state=42)
kmeans.fit(X_scaled)
# Kümeyi tahmin et 
df['Küme']=kmeans.labels_
plt.figure(figsize=(8,6))
plt.scatter(df['Yaş'],df['Gelir'],c=df['Küme'],cmap='viridis')
plt.title('Kümeleme algoritması')
plt.xlabel("Yaş")
plt.ylabel("Gelir")
plt.show()
