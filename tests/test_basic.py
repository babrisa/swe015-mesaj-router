"""Temel testler. Model ağırlığı gerektirmez."""
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, "src")
from preprocess import edit_distance, normalize, numara_bul, selamlasma_cevabi, terim_duzelt  # noqa: E402
from router import PERSONEL, karar, kart_yaz  # noqa: E402


@pytest.mark.parametrize("numara", ["2401010123", "240101123", "1808019999", "2609100001"])
def test_gecerli_numara(numara):
    assert numara_bul(f"no {numara} lütfen") == [numara]


@pytest.mark.parametrize("numara", ["3001010123", "2411010123", "24010101", "1234567890"])
def test_gecersiz_numara(numara):
    assert numara_bul(numara) == []


def test_cevresinde_rakam_olan_numara_yakalanmaz():
    assert numara_bul("24010101234") == []
    assert numara_bul("x2401010123y") == ["2401010123"]


def test_normalizasyon():
    assert normalize("  ÖĞRENCİ belgesi   lazım no 2401010123 ") == "öğrenci belgesi lazım no <no>"
    assert normalize("çoooook acil") == "çok acil"
    assert normalize("IŞIK") == "ışık"


def test_edit_distance():
    assert edit_distance("kitap", "kitap") == 0
    assert edit_distance("kitap", "katip") == 2
    assert edit_distance("", "abc") == 3


@pytest.mark.parametrize("yanlis,dogru", [("transkirpt", "transkript"), ("diplomma", "diploma"), ("harc", "harç"), ("ogrenci", "öğrenci")])
def test_terim_duzeltme(yanlis, dogru):
    assert terim_duzelt(yanlis) == dogru


@pytest.mark.parametrize("kelime", ["bursa", "kayıp", "dert", "hava"])
def test_normal_kelimeye_dokunulmaz(kelime):
    assert terim_duzelt(kelime) == kelime


def test_selamlasma_kurali():
    assert selamlasma_cevabi("Merhaba!")
    assert selamlasma_cevabi("nasılsın?")
    assert selamlasma_cevabi("merhaba transkript istiyorum") is None


def test_test_sizintisi():
    oku = lambda ad: [normalize(k["text"]) for k in json.load(open(f"data/{ad}.json", encoding="utf-8"))]  # noqa: E731
    test = set(oku("test"))
    assert not test & set(oku("train")) and not test & set(oku("val"))


def test_egitimde_kapsam_disi_yok():
    assert "unknown" not in {k["intent"] for k in json.load(open("data/train.json", encoding="utf-8"))}


def test_esik_altinda_personele_aktar():
    olasilik = np.full(10, 0.1)
    niyet, guven, birim = karar(olasilik, 0.5)
    assert niyet == "unknown" and birim == PERSONEL and guven == pytest.approx(0.1)


def test_esik_ustunde_birime_gider():
    olasilik = np.zeros(10)
    olasilik[6] = 0.9  # harc-odeme
    assert karar(olasilik, 0.5)[0] == "harc-odeme" and karar(olasilik, 0.5)[2] == "mali-isler"


def test_kart_numara_kurali():
    assert "2401010123" in kart_yaz("transkript", "ogrenci-isleri", ["2401010123"])
    assert "belirtilmedi" in kart_yaz("transkript", "ogrenci-isleri", ["240101123"])
    assert "belirtilmedi" in kart_yaz("transkript", "ogrenci-isleri", [])
