
### README.md

---

# Teknolojik Ürünler Kümeleme Projesi

Bu proje, teknolojik ürünlerin satış ve fiyat verilerine dayalı olarak K-Means algoritması kullanarak kümeleme (clustering) yapmayı amaçlamaktadır. Ürünler farklı kümelere atanarak benzerliklerine göre gruplandırılır ve satış ve fiyat ilişkisi üzerinden analiz yapılır.

## İçindekiler

- [Gereksinimler](#gereksinimler)
- [Kurulum](#kurulum)
- [Veri Seti](#veri-seti)
- [Proje Açıklaması](#proje-açıklaması)
- [Çalıştırma](#çalıştırma)
- [Sonuçlar](#sonuçlar)

## Gereksinimler

Projeyi çalıştırmak için aşağıdaki Python kütüphanelerine ihtiyaç duyulmaktadır:

- `pandas`
- `matplotlib`
- `scikit-learn`

Bu kütüphaneleri aşağıdaki komutla yükleyebilirsiniz:

```bash
pip install pandas matplotlib scikit-learn
```

## Kurulum

Projeyi yerel ortamınızda çalıştırmak için şu adımları izleyin:

1. Proje dosyalarını indirin veya klonlayın:

   ```bash
   git clone https://github.com/emre381/proje.git
   ```

2. Gerekli kütüphaneleri kurun (Gereksinimler bölümüne bakınız).

3. Veri setini `teknolojik_urunler_zamanli.xlsx` olarak proje dizinine ekleyin.

## Veri Seti

Veri seti, çeşitli teknolojik ürünlerin satış miktarlarını, fiyatlarını ve ürün adlarını içermektedir. Ayrıca, her satışın gerçekleştiği tarih de veri setinde yer almaktadır.

Veri setindeki temel sütunlar:

- **Fiyat (TL)**: Ürünün satış fiyatı.
- **Satış**: Satılan ürün miktarı.
- **Ürün Adı**: Ürünün ismi.
- **Tarih**: Ürünün satıldığı tarih.

## Proje Açıklaması

Bu projede KMeans algoritması kullanılarak teknolojik ürünler, fiyat ve satış değerlerine göre 3 kümeye ayrılır. Her ürün, fiyatı ve satış miktarı baz alınarak bir kümeye atanır. Veriler, `StandardScaler` kullanılarak ölçeklendirilir ve KMeans algoritması ile kümeleme yapılır.

Grafikte, her bir ürün farklı bir renk ile gösterilir ve ürün adı, tarih ve satış bilgileri her bir veri noktası üzerine eklenir.

## Çalıştırma

Proje dosyası olan `teknolojik.py` dosyasını çalıştırmak için şu adımları izleyin:

1. Terminalde proje dizinine gidin:

   ```bash
   cd proje_dizini
   ```

2. Python dosyasını çalıştırın:

   ```bash
   python teknolojik.py
   ```

Çalıştırdıktan sonra ürünlerin satış ve fiyatlarına göre gruplandırıldığı bir grafik ekranda görüntülenecektir.

## Sonuçlar

Bu projede, ürünler fiyat ve satış verileri üzerinden üç kümeye ayrılmıştır. Grafik üzerinde her bir ürün farklı renkte bir kümeye atanmıştır ve ürün adı, satış miktarı ve fiyat bilgileri grafik üzerinde gösterilmiştir. Bu sayede hangi ürünlerin benzer olduğunu ve fiyat-satış ilişkisi hakkında görsel bir analiz yapılabilmiştir.

---

