# Mimari

Her şey tek bir Python sürecinde, Mac CPU üzerinde çalışır (Apple M5). Eğitim MPS kullanır. Kutulardaki adlar koddaki dosya ve fonksiyon adlarıyla aynıdır.

```mermaid
flowchart TD
    K[Kullanıcı] --> A["app.py: yonlendir()"]
    A --> R["src/router.py: route(mesaj, esik)"]
    R --> S["src/preprocess.py: selamlasma_cevabi()"]
    S -- selamlaşma --> C1[Hazır cevap kartı]
    S -- diğer --> N["src/preprocess.py: normalize(), numara_bul()"]
    N --> M1["tfidf (FP64)"]
    N --> M2["fasttext_ffn (FP32)"]
    N --> M3["berturk, öğretmen (FP32)"]
    N --> M4["ogrenci_int8, damıtılmış öğrenci (INT8), üretim modeli"]
    M1 --> T["app.py: Dört model yan yana tablosu"]
    M2 --> T
    M3 --> T
    M4 --> T
    M4 --> D["src/router.py: karar(olasilik, esik)"]
    D -- eşik üstü --> B["config.yaml: intents -> birim"]
    D -- eşik altı --> P[Personele aktar]
    B --> C2["src/router.py: kart_yaz()"]
    P --> C2
    C2 --> A
```

Görsel hali: [architecture.png](architecture.png)

| Bileşen | Dosya | Hassasiyet | Nerede |
|---|---|---|---|
| Arayüz | `app.py` | | Mac CPU |
| Ön işleme | `src/preprocess.py` | | Mac CPU |
| tfidf | `models/tfidf.joblib` | FP64 | Mac CPU |
| fasttext_ffn | `models/fasttext.kv`, `models/ffn.pt` | FP32 | Mac CPU |
| berturk (öğretmen, 12 katman) | `models/bert/` | FP32 | Mac CPU |
| ogrenci_int8 (öğrenci, 6 katman) | `models/ogrenci_int8.pt` | INT8 | Mac CPU |
