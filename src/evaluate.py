"""Dört modeli aynı test setinde ölçer; reports/results.csv ve hardware.txt yazar."""
import csv
import os
import platform
import subprocess
import time

import numpy as np
import torch

from models import BILINMEYEN, CONFIG, modelleri_yukle, veri_yukle

CALISTIRMA = 200  # gecikme için tek mesajlık çalıştırma sayısı
DOSYALAR = {"tfidf": ["models/tfidf.joblib"], "fasttext_ffn": ["models/fasttext.kv", "models/ffn.pt"],
            "berturk": ["models/bert/model.safetensors"], "ogrenci_int8": ["models/ogrenci_int8.pt"]}
HASSASIYET = {"tfidf": "FP64", "fasttext_ffn": "FP32", "berturk": "FP32", "ogrenci_int8": "INT8"}


def gecikme_olc(tahmin, metinler):
    """Tek mesajlık tahminin ortalama süresini milisaniye olarak ölçer."""
    tahmin(metinler[:1])  # ilk çalıştırma ısınmadır, sayılmaz
    baslangic = time.perf_counter()
    for i in range(CALISTIRMA):
        tahmin([metinler[i % len(metinler)]])
    return (time.perf_counter() - baslangic) / CALISTIRMA * 1000


def olc(tahmin, metinler, etiketler, esik):
    """Doğruluk ve kapsam dışı yakalama oranını hesaplar."""
    olasilik = tahmin(metinler)
    karar = olasilik.argmax(axis=1)
    karar[olasilik.max(axis=1) < esik] = BILINMEYEN  # eşik altı -> personele aktar
    dis = etiketler == BILINMEYEN
    return (karar == etiketler).mean(), (karar[dis] == BILINMEYEN).mean()


if __name__ == "__main__":
    torch.set_num_threads(1)  # tüm modeller aynı koşulda: tek CPU iş parçacığı
    metinler, etiketler = veri_yukle("test")
    esik = CONFIG["threshold"]
    os.makedirs("reports", exist_ok=True)
    with open("reports/results.csv", "w", newline="") as dosya:
        yazici = csv.writer(dosya)
        yazici.writerow(["model", "hassasiyet", "accuracy", "oos_recall", "esik", "boyut_mb", "gecikme_ms", "calistirma"])
        for ad, tahmin in modelleri_yukle().items():
            dogruluk, yakalama = olc(tahmin, metinler, etiketler, esik)
            boyut = sum(os.path.getsize(d) for d in DOSYALAR[ad]) / 2**20
            satir = [ad, HASSASIYET[ad], f"{dogruluk:.4f}", f"{yakalama:.4f}", esik, f"{boyut:.1f}",
                     f"{gecikme_olc(tahmin, metinler):.2f}", CALISTIRMA]
            yazici.writerow(satir)
            print(satir)
    mac = platform.system() == "Darwin"
    komut = ["sysctl", "-n", "machdep.cpu.brand_string"]
    islemci = subprocess.run(komut, capture_output=True, text=True).stdout.strip() if mac else platform.processor()
    with open("reports/hardware.txt", "w") as dosya:
        dosya.write(f"{islemci}, {platform.platform()}\n")
        dosya.write(f"Sadece CPU, 1 is parcacigi, tek mesaj, tokenizer dahil, normalize haric. Test: {len(etiketler)} mesaj.\n")
