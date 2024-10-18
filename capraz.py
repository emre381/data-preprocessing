import pandas as pd 
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score
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

def linearmodel():
    #  modeli oluştur ve çapraz olarak değerlendir 
    linear_model = LinearRegression()
    cv_scores = cross_val_score(linear_model,X_train,y_train,cv=5,scoring='neg_mean_squared_error')
    cv_rmse_scores = (-cv_scores)**0.5
    print(f"Linear R rg 5 katmanlı Cross Val Puanı {cv_rmse_scores.mean():.2f}")

def forestmodel():
    # random forest modeli ile çapraz doğrulama modelini ölçme 
    rf_model = RandomForestRegressor(n_estimators=100,random_state=42)
    cv_scores_rf = cross_val_score(rf_model,X_train,y_train,cv=5,scoring="neg_mean_squared_error")
    cv_rmse_scores_rf = (-cv_scores_rf)**0.5
    print(f"Random Reg 5 katmanlı Cross Val Puanı:{cv_rmse_scores_rf.mean():.2f}")