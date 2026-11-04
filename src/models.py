"""Dört modelin eğitimi ve tahmin fonksiyonları."""
import json

import joblib
import numpy as np
import torch
import yaml
from gensim.models import FastText
from gensim.models.fasttext import FastTextKeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from torch import nn
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from preprocess import normalize

CONFIG = yaml.safe_load(open("config.yaml", encoding="utf-8"))
INTENTLER = list(CONFIG["intents"])
BILINMEYEN = len(INTENTLER)  # kapsam dışı mesajların etiketi; hiçbir model bu sınıfı öğrenmez
torch.manual_seed(42)


def veri_yukle(ad):
    """data/ad.json dosyasını okur; normalize metinleri ve sayısal etiketleri döndürür."""
    kayitlar = json.load(open(f"data/{ad}.json", encoding="utf-8"))
    metinler = [normalize(k["text"]) for k in kayitlar]
    etiketler = [INTENTLER.index(k["intent"]) if k["intent"] in INTENTLER else BILINMEYEN for k in kayitlar]
    return metinler, np.array(etiketler)


def parcalara_bol(kayit_sayisi):
    """Eğitim için kayıtları karıştırır ve 16'lık parçalara böler."""
    sira = torch.randperm(kayit_sayisi).tolist()
    return [sira[i:i + 16] for i in range(0, kayit_sayisi, 16)]


def kodla(tokenizer, metinler):
    """Metinleri BERT'in anlayacağı sayılara çevirir, en fazla 64 parça alır."""
    return tokenizer(metinler, truncation=True, padding=True, max_length=64, return_tensors="pt")


def tfidf_egit(metinler, etiketler):
    """Klasik temel model: harf n-gram tf-idf + logistic regression (B)."""
    vektorlestirici = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)
    model = make_pipeline(vektorlestirici, LogisticRegression(C=30, max_iter=2000)).fit(metinler, etiketler)
    joblib.dump(model, "models/tfidf.joblib")


def ortalama_vektor(kelime_vektorleri, metinler):
    """Her mesajın fastText kelime vektörlerinin ortalamasını alır (A)."""
    sonuc = []
    for metin in metinler:
        vektorler = [kelime_vektorleri.get_vector(k, norm=True) for k in metin.split()]
        sonuc.append(np.mean(vektorler or [np.zeros(50, dtype=np.float32)], axis=0))
    return torch.tensor(np.stack(sonuc)) * 10  # ortalamalar küçük, ölçekleme SGD'nin öğrenmesini kolaylaştırır


def ffn_kur():
    """Bir gizli katmanlı feedforward ağı kurar (A)."""
    return nn.Sequential(nn.Linear(50, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, len(INTENTLER)))


def ffn_egit(metinler, etiketler):
    """fastText'i eğitim mesajlarından öğrenir, ağı SGD ve backpropagation ile eğitir (A, B)."""
    cumleler = [m.split() for m in metinler]
    fasttext = FastText(cumleler, vector_size=50, window=4, min_count=1, sg=1, min_n=2, max_n=5,
                        bucket=50000, epochs=100, seed=42, workers=1)
    fasttext.wv.save("models/fasttext.kv")
    x = ortalama_vektor(fasttext.wv, metinler)
    y = torch.tensor(etiketler)
    ag = ffn_kur()
    optimizer = torch.optim.SGD(ag.parameters(), lr=0.05, momentum=0.9)
    for _ in range(200):
        for parca in parcalara_bol(len(metinler)):
            kayip = nn.functional.cross_entropy(ag(x[parca]), y[parca])
            optimizer.zero_grad()
            kayip.backward()  # backpropagation: gradyanlar hesaplanır
            optimizer.step()
    torch.save(ag.state_dict(), "models/ffn.pt")


def damitma_kaybi(ogrenci_cikti, ogretmen_cikti, sicaklik=3):
    """Öğrencinin öğretmenin yumuşatılmış olasılıklarına benzemesini sağlayan kayıp (D)."""
    hedef = torch.softmax(ogretmen_cikti / sicaklik, dim=1)
    tahmin = torch.log_softmax(ogrenci_cikti / sicaklik, dim=1)
    return nn.functional.kl_div(tahmin, hedef, reduction="batchmean") * sicaklik ** 2


def bert_egit(metinler, etiketler, klasor, katman=12, tur=8, ogretmen=None):
    """BERTurk'ü ince ayarlar (C). Öğretmen verilirse öğrenci damıtma ile eğitilir (D)."""
    cihaz = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["model_name"], num_labels=len(INTENTLER), num_hidden_layers=katman).to(cihaz)
    if ogretmen:
        ogretmen = AutoModelForSequenceClassification.from_pretrained(ogretmen).to(cihaz).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    y = torch.tensor(etiketler)
    for _ in range(tur):
        for parca in parcalara_bol(len(metinler)):
            girdi = kodla(tokenizer, [metinler[j] for j in parca]).to(cihaz)
            cikti = model(**girdi, labels=y[parca].to(cihaz))
            kayip = cikti.loss
            if ogretmen:
                with torch.no_grad():
                    ogretmen_cikti = ogretmen(**girdi).logits
                kayip = 0.3 * kayip + 0.7 * damitma_kaybi(cikti.logits, ogretmen_cikti)
            optimizer.zero_grad()
            kayip.backward()
            optimizer.step()
    model.cpu().save_pretrained(klasor)
    tokenizer.save_pretrained(klasor)


def quantize_et(model):
    """Linear katman ağırlıklarını FP32'den INT8'e çevirir (D). Etkinleştirmeler çalışırken ölçeklenir."""
    # Motor işletim sistemine göre değişir, bu yüzden desteklenen listeden seçilir.
    torch.backends.quantized.engine = torch.backends.quantized.supported_engines[-1]
    return torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


def ogrenci_quantize_kaydet():
    """Öğrenci modelin INT8 halini models/ogrenci_int8.pt dosyasına yazar (D)."""
    model = AutoModelForSequenceClassification.from_pretrained("models/ogrenci")
    torch.save(quantize_et(model).state_dict(), "models/ogrenci_int8.pt")


def bert_tahminci(model, tokenizer):
    """Bir BERT modelini 'metin listesi -> olasılıklar' fonksiyonuna çevirir."""
    def tahmin(metinler):
        sonuc = []
        with torch.no_grad():
            for i in range(0, len(metinler), 32):
                sonuc.append(torch.softmax(model(**kodla(tokenizer, metinler[i:i + 32])).logits, dim=1))
        return torch.cat(sonuc).numpy()
    return tahmin


def modelleri_yukle():
    """Kayıtlı dört modeli yükler; her biri için tahmin fonksiyonu döndürür."""
    tfidf = joblib.load("models/tfidf.joblib")
    kelime_vektorleri = FastTextKeyedVectors.load("models/fasttext.kv")
    ag = ffn_kur()
    ag.load_state_dict(torch.load("models/ffn.pt"))
    ag.eval()  # dropout tahminde kapalı

    def ffn_tahmin(metinler):
        with torch.no_grad():
            return torch.softmax(ag(ortalama_vektor(kelime_vektorleri, metinler)), dim=1).numpy()

    tokenizer = AutoTokenizer.from_pretrained("models/bert")
    bert = AutoModelForSequenceClassification.from_pretrained("models/bert").eval()
    bos = AutoModelForSequenceClassification.from_config(AutoConfig.from_pretrained("models/ogrenci")).eval()
    int8 = quantize_et(bos)
    int8.load_state_dict(torch.load("models/ogrenci_int8.pt", weights_only=False))
    return {"tfidf": tfidf.predict_proba, "fasttext_ffn": ffn_tahmin,
            "berturk": bert_tahminci(bert, tokenizer), "ogrenci_int8": bert_tahminci(int8, tokenizer)}
