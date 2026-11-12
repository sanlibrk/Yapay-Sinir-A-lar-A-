# Adım adım okuma ve çalıştırma rehberi

Bu rehber, projeyi ilk kez açan bir öğrencinin hangi dosyaya bakacağını, kodu nasıl çalıştıracağını ve sonuçları nasıl anlatacağını gösterir. İki model de **bir yıl sonraki hastalık ilerleme puanını** tahmin eder. Basit model tek girdi, çoklu model 10 girdi kullanır.

## 1. Önce projede yolunu bul

1. [Depo ana sayfasını](https://github.com/sanlibrk/Yapay-Sinir-A-lar-A-) aç ve README'yi oku. Burada çalışmanın amacı ve kısa sonuçları bulunur.
2. Konuyu anlamak için [PDF raporuna](../rapor.pdf) bak. Ayrıntılı yöntem, denklemler ve kaynaklar için [bilimsel raporu](bilimsel_rapor.md) aç.
3. Kodu adım adım görmek için [notebook'u](../regresyon_odevi.ipynb), tek dosya olarak görmek için [regresyon.py](../regresyon.py) dosyasını aç.
4. Sayısal karşılaştırmayı [sonuclar.csv](../sonuclar.csv), görsel karşılaştırmayı [regresyon_grafikleri.png](../regresyon_grafikleri.png) üzerinden incele.

Yalnız sonuçları okumak istiyorsan kurulum yapman gerekmez. Deneyi yeniden çalıştırmak için aşağıdaki iki yoldan birini seç.

## 2. En kolay yol: Google Colab

Colab, Python kodunu tarayıcıda çalıştırır; bilgisayarına Python kurman gerekmez.

1. **[Notebook'u Google Colab'da aç](https://colab.research.google.com/github/sanlibrk/Yapay-Sinir-A-lar-A-/blob/main/regresyon_odevi.ipynb).** Çalıştırmak için Google hesabınla oturum açman istenebilir.
2. Üstte **Bağlan / Connect** düğmesini görüyorsan tıkla ve çalışma ortamının açılmasını bekle.
3. Menüden **Çalışma zamanı → Tümünü çalıştır** seç. İngilizce arayüzde karşılığı **Runtime → Run all** şeklindedir; menü adları arayüz diline göre değişebilir.
4. Hücrelerin tamamlanmasını bekle. Her hücrenin solunda çalıştırma numarası görünür. Dosya indirme veya veri yükleme adımı yoktur; veri seti scikit-learn içinde hazırdır.
5. Sonuç tablosunu ve dört parçalı grafiği notebook'un alt kısmında incele.

Notebook'taki altı kod hücresinin sırası şöyledir:

| Hücre | Ne yapar? |
|---|---|
| 1 | Kütüphaneleri yükler. |
| 2 | Veri setini, girdileri ve hedefi tanımlar. |
| 3 | Eğitim/test ayrımını yapar ve veri CSV'sini kaydeder. |
| 4 | Hata ölçme işlevini kurar; iki modeli eğitir ve tahminleri hesaplar. |
| 5 | Sonuçları ekrana yazar; CSV dosyalarını kaydeder. |
| 6 | Grafikleri üretir ve gösterir. |

Bir kütüphane bulunamadı hatası alırsan üst tarafa yeni bir kod hücresi ekleyip şunu çalıştır; ardından tüm hücreleri yeniden çalıştır:

```python
%pip install numpy pandas matplotlib scikit-learn
```

Çıktıları indirmek için soldaki **Dosyalar / Files** klasör simgesini aç, gerekirse yenile düğmesine bas. `sonuclar.csv` veya diğer dosyanın yanındaki üç noktadan **İndir / Download** seç. Colab'da üretilen dosyalar çalışma ortamındadır; oturum kapanmadan saklamak istediklerini indir. **Colab'daki değişiklikler GitHub deposuna otomatik yazılmaz.**

## 3. Bilgisayarında çalıştırma

### 3.1. Depoyu indir ve doğru klasöre gir

Bilgisayarında **Python 3.11 veya daha yeni bir sürüm** bulunmalı. Git kuruluysa terminalde:

```bash
git clone https://github.com/sanlibrk/Yapay-Sinir-A-lar-A-.git
cd Yapay-Sinir-A-lar-A-
```

Git kullanmıyorsan depo sayfasında **Code → Download ZIP** seç, ZIP'i çıkar ve çıkarılan proje klasöründe terminal aç. ZIP klasörünün adı genellikle `Yapay-Sinir-A-lar-A--main` olur. Adını kontrol et; Git ile indirilen klasör adıyla aynı olmak zorunda değildir.

Windows'ta Dosya Gezgini ile `regresyon.py` dosyasının bulunduğu klasöre gir. Adres çubuğuna `cmd` yazıp Enter'a bas: Komut İstemi o klasörde açılır. macOS/Linux'ta terminal açıp `cd` ile aynı klasöre git.

### 3.2. İstersen sanal ortam oluştur

Sanal ortam, bu projenin paketlerini diğer projelerden ayırır. Zorunlu değildir.

**Windows — Komut İstemi (CMD):**

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Ortam etkinleşince komut satırının başında genellikle `(.venv)` görünür.

### 3.3. Paketleri kur ve kodu çalıştır

Etkin sanal ortamda veya `python` komutu doğru Python sürümünü açıyorsa:

```bash
python --version
python -m pip install -r requirements.txt
python regresyon.py
```

Windows'ta `python` bulunamaz ama `py` çalışırsa aynı komutlarda `python` yerine `py` kullan. Sanal ortam oluşturmadan macOS/Linux'ta çalışıyorsan gerekirse `python` yerine `python3` yaz.

Terminalde 442 örneğin 353 eğitim ve 89 test örneğine ayrıldığı yazmalı. Sonuç dosyaları `regresyon.py` ile aynı klasöre kaydedilir; aynı adlı önceki çıktılar güncellenir. Grafik penceresi açılmaması normaldir: grafik `regresyon_grafikleri.png` dosyasına yazılır.

## 4. Kodun temel mantığını anla

```python
X = veri.data
y = veri.target
```

`X`, tahmin için kullanılan 10 özelliği; `y`, tahmin edilmeye çalışılan ilerleme puanını tutar. Hedef, doğrudan kan şekeri değeri değildir.

```python
ozellikler = {
    "Basit doğrusal regresyon": ["bmi"],
    "Çoklu doğrusal regresyon": list(X.columns),
}
```

Karşılaştırmanın farkı buradadır: ilk model yalnız BMI'ı, ikinci model tüm özellikleri görür. İkisi de aynı hedefi ve aynı eğitim/test kişilerini kullanır.

```python
model.fit(X_egitim, y_egitim)
model.predict(X_test)
```

`fit`, eğitim örneklerinden denklem katsayılarını öğrenir. `predict`, öğrenilmiş denklemle test girdilerine tahmin üretir. Kod, testteki gerçek hedefleri katsayı öğrenmek için kullanmaz. `LinearRegression` en küçük kareler çözümünü hesaplar; bu uygulamada epoch veya öğrenme hızı ayarlamazsın.

## 5. Çıktıları oku

| Dosya | İçeriği |
|---|---|
| `diabetes_veri_seti.csv` | Özellikler, hedef ve hangi satırın eğitim/test olduğu. |
| `sonuclar.csv` | İki modelin eğitim ve test hata ölçümleri. |
| `test_tahminleri.csv` | Her test kişisinin gerçek değeri, tahminleri ve artıkları. |
| `katsayilar.csv` | Öğrenilen sabit terimler ve özellik katsayıları. |
| `regresyon_grafikleri.png` | Regresyon doğrusu, gerçek–tahmin ve hata grafikleri. |

Testte basit modelin **RMSE'si 63,73**, çoklu modelin **53,85** bulundu. Çoklu modelin RMSE'si bu bölünmede yaklaşık **%15,50 daha düşük**. MAE sırasıyla **52,26** ve **42,79** puan. Hatanın küçük olması daha iyi tahmin demektir; sonuç, çoklu modelin her veri setinde üstün olacağını kanıtlamaz.

## 6. Sık karşılaşılan sorular

**“Dosya bulunamadı” hatası alıyorum.** Terminalin `regresyon.py` ve `requirements.txt` bulunan klasörde olduğundan emin ol. Windows'ta `dir`, macOS/Linux'ta `ls` ile dosyaları görebilirsin.

**`ModuleNotFoundError` alıyorum.** Kurulum ile çalıştırmayı aynı Python ortamında yap. `python -m pip install -r requirements.txt` komutunu tekrar çalıştır.

**Neden eğitim ve test ayrı?** Eğitim verisi katsayıları öğrenmek, test verisi daha önce görülmemiş örneklerdeki hatayı ölçmek içindir.

**Neden accuracy yok?** Burada sınıf değil sayısal değer tahmin edilir. MSE, RMSE, MAE ve R² kullanılır. R²=0,4526, tahminlerin %45,26'sının doğru olduğu anlamına gelmez.

**MSE neden büyük?** Hatalar karesini alarak toplanır; birimi puanın karesidir. RMSE ve MAE hedefle aynı birimdedir.

**Çoklu regresyonda birden fazla çıktı mı var?** Bu çalışmada hayır: girdiler birden fazla, hedef tektir.

## 7. Hocaya altı cümlede anlat

1. “442 kişinin bulunduğu Diabetes veri setinde bir yıl sonraki hastalık ilerleme puanını tahmin ettim.”
2. “Basit regresyonda yalnız BMI'ı, çoklu regresyonda 10 özelliği kullandım.”
3. “İki model için aynı 353 eğitim ve 89 test örneğini kullandım.”
4. “Katsayıları eğitim verisiyle öğrenip testte MSE, RMSE, MAE ve R² hesapladım.”
5. “Bu bölünmede RMSE 63,73'ten 53,85'e düştü; çoklu model daha düşük hata verdi.”
6. “Tek bir veri ayrımıyla genel üstünlük veya nedensellik sonucu çıkarmadım.”
