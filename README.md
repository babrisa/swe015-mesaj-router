# Öğrenci İşleri Yönlendirici

İstinye Üniversitesi'ne gelen Türkçe öğrenci mesajlarını doğru birime yönlendirir. Mesajı normalize eder, öğrenci numarasını bulur, niyeti güven skoruyla tahmin eder ve talep kartı yazar. Eşik altındaki mesajlar personele gider, selamlaşmalar kuralla cevaplanır. Arayüz dört modeli yan yana gösterir.

SWE015 Use Case 2. Ödev alan ve dil değişikliğine izin veriyor ("You may change domain details (language, ...)").

## Kurulum ve çalıştırma

Python 3.12 gerekir. Tek komut:

```bash
./run.sh
```

Komut sanal ortamı kurar, paketleri yükler, model yoksa eğitir ve uygulamayı http://127.0.0.1:7860 adresinde açar. İlk çalıştırmada BERTurk indirilir ve eğitim yapılır.

Ağırlıklar repoda yok. `./run.sh` onları yeniden üretir. Hazır ağırlıklar: [HF LINK]

Testler için:

```bash
.venv/bin/python -m pytest
```

## Sonuçlar

Dört model aynı test setinde ölçüldü (218 mesaj: 120 kapsam içi, 98 kapsam dışı). Donanım: Apple M5, sadece CPU, 1 iş parçacığı, 200 tek mesajlık çalıştırmanın ortalaması. Kaynak: `reports/results.csv`, `reports/hardware.txt`.

| Model | Hassasiyet | Accuracy | OOS recall | Boyut (MB) | Gecikme (ms) |
|---|---|---:|---:|---:|---:|
| tfidf | FP64 | 70,6% | 100,0% | 0,7 | 0,19 |
| fasttext_ffn | FP32 | 60,6% | 85,7% | 9,8 | 0,06 |
| berturk (öğretmen) | FP32 | 87,2% | 93,9% | 422,0 | 12,42 |
| ogrenci_int8 (damıtılmış + quantize) | INT8 | 87,2% | 90,8% | 136,6 | 12,37 |

Damıtma ve quantization boyutu 3,1 kat küçülttü, accuracy değişmedi, gecikme aynı kaldı. Kapsam dışı yakalama 3 puan düştü.

Dört model de aynı sabit eşikle (0,8) ölçüldü. Bu eşik BERTurk modellerine uygun, ama TF-IDF daha düşük güven verdiği için onun doğru bildiği birçok mesaj da personele gidiyor.

Eğitim Mac'te (MPS) her çalıştırmada birebir aynı sonucu vermez. BERTurk accuracy değeri farklı çalıştırmalarda %87 ile %92 arasında görüldü. Tablo tek bir çalıştırmanın sonucudur.

## Tasarım kararları

- Öğretmen 12 katmanlı BERTurk, öğrenci 6 katmanlı hali. Öğrenci damıtma ile eğitilir, sonra PyTorch dynamic INT8 quantization uygulanır.
- fastText vektörleri hazır indirilmez, `gensim` ile eğitim mesajlarından öğrenilir.
- Kapsam dışı için ayrı sınıf yok, yalnızca `config.yaml` içindeki sabit güven eşiği (0,8) var.
- Niyet ve birim eşlemesi `config.yaml` içinde.

Mimari diyagram: [docs/architecture.md](docs/architecture.md)

## Veri

- Kapsam içi mesajlar bu proje için yazıldı (AI taslağı). Gerçek öğrenci mesajı, numarası, adı veya e-postası yok. Numaralar sahtedir.
- Kapsam dışı mesajların çoğu MASSIVE 1.1 `tr-TR` veri setinden (Amazon, CC BY 4.0), kalanı yazıldı. Bunlar eğitime girmedi.
- Eğitim 351, doğrulama 130, test 218 mesaj. Test seti ayrı ve farklı tarzda yazıldı.

## AI kullanımı

Kod, testler, dokümanlar ve kapsam içi mesajların taslağı Claude Code (Claude Opus 5.5) ile hazırlandı. Tüm sonuçlar bu repodaki kod çalıştırılarak ölçüldü.
