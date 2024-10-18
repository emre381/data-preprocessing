import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_excel('veri_on_isleme_ve_ozellik_muhendisligi.xlsx')

# eksik gelir verileri ortalam ile doldur 
df.fillna(df['Gelir'].mean(),inplace=True)
# print(df)
le=LabelEncoder()
df['Cinsiyet']=le.fit_transform(df['Cinsiyet'])
# print(df)

scaler=StandardScaler()
# df[['Yaş','Gelir']]=scaler.fit_transform(df[['Yaş','Gelir']])
# print(df)

df['Gelir_Grubu']=pd.cut(df['Gelir'],bins=[0,3000,5000,7000],labels=['Düşük','Orta','Yüksek'])
# print(df[['Gelir','Gelir_Grubu']])
# df.drop('ID',axis=1,inplace=True)
# df.to_excel('Kategorik_Gelir.xlsx',index=False)
# print("İşlem tamamlandı")

plt.figure(figsize=(10,6))
plt.hist(df['Meslek'],bins=10,color='skyblue',edgecolor='black')
plt.title('Yaş Dağılımı')
plt.xlabel("Yaş")
plt.ylabel("Frekans")
# plt.show()

plt.figure(figsize=(10,6))
sns.countplot(x='Gelir_Grubu' , hue='Yaş',data=df)
plt.title('Gelir Grubu ve Cinsiyet Dağılı  ilişkisi')
plt.xlabel("Gelir Grubu")
plt.ylabel("Kişi sayısı")
plt.show()