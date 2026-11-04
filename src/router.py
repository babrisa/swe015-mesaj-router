"""Bir mesajı uçtan uca yönlendirir: ön işleme, quantize model, eşik, birim, kart."""
import numpy as np

from models import CONFIG, INTENTLER, modelleri_yukle
from preprocess import normalize, numara_bul, selamlasma_cevabi

PERSONEL = "Personele aktar"
MODELLER = {}  # modeller ilk mesajda bir kez yüklenir


def yukle():
    """Modelleri bir kez belleğe alır, tekrar çağrılırsa yeniden yüklemez."""
    if not MODELLER:
        MODELLER.update(modelleri_yukle())
    return MODELLER


def karar(olasilik, esik):
    """Olasılıklardan niyet, güven ve birimi seçer; eşik altı personele gider."""
    i = int(np.argmax(olasilik))
    guven = float(olasilik[i])
    if guven < esik:
        return "unknown", guven, PERSONEL
    return INTENTLER[i], guven, CONFIG["intents"][INTENTLER[i]]["birim"]


def kart_yaz(niyet, birim, numaralar):
    """Talep kartı metnini oluşturur; yalnızca 10 haneli numarayı yazar."""
    if niyet == "unknown":
        return "Mesajınız personele aktarıldı."
    numara = next((n for n in numaralar if len(n) == 10), "belirtilmedi")
    return f"{CONFIG['intents'][niyet]['ad']} talebi alındı, öğrenci no: {numara}, yönlendirilen birim: {birim}"


def route(mesaj, esik=CONFIG["threshold"]):
    """Mesajı yönlendirir ve sonucu sözlük olarak döndürür."""
    metin = normalize(mesaj)
    cevap = selamlasma_cevabi(mesaj)
    if cevap:
        return {"niyet": "greeting", "guven": 1.0, "birim": "-", "kart": cevap, "metin": metin, "tablo": []}
    tablo = []
    for ad, tahmin in yukle().items():  # dört modelin tahmini yan yana gösterilir
        niyet, guven, birim = karar(tahmin([metin])[0], esik)
        tablo.append([ad, niyet, round(guven, 3), birim])
    niyet, guven, birim = tablo[-1][1:]  # son satır üretim modeli: ogrenci_int8
    return {"niyet": niyet, "guven": guven, "birim": birim, "metin": metin, "tablo": tablo,
            "kart": kart_yaz(niyet, birim, numara_bul(mesaj))}
