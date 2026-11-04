"""Klasik NLP adımları: regex normalizasyon, numara bulma, edit distance, selamlaşma."""
import re

# Öğrenci numarası: yıl (18-26) + 0 + 6 veya 7 rakam. Çevresinde başka rakam olamaz.
NUMARA = re.compile(r"(?<!\d)(1[89]|2[0-6])0\d{6,7}(?!\d)")
TERIMLER = ["transkript", "diploma", "mezuniyet", "öğrenci", "belgesi", "harç", "kayıt", "dondurma",
            "yatay", "geçiş", "burs", "staj", "şifre", "ödeme", "ders", "seçimi", "ois"]
SELAM = {
    "merhaba": "Merhaba! Öğrenci işleriyle ilgili nasıl yardımcı olabilirim?",
    "selam": "Selam! Size nasıl yardımcı olabilirim?",
    "teşekkürler": "Rica ederim! Başka bir konuda yardımcı olabilir miyim?",
    "nasılsın": "İyiyim, teşekkürler. Size nasıl yardımcı olabilirim?",
}
SADELESTIR = str.maketrans("çğıöşü", "cgiosu")


def kucuk_harf(metin):
    """Türkçe kurala göre küçük harfe çevirir (I -> ı, İ -> i)."""
    return metin.replace("İ", "i").replace("I", "ı").lower()


def sade(kelime):
    """Karşılaştırma için Türkçe harfleri sadeleştirir (ç -> c gibi)."""
    return kelime.translate(SADELESTIR)


def edit_distance(a, b):
    """İki kelime arasındaki en az ekleme, silme ve değiştirme sayısını bulur."""
    # Tablonun her satırında sadece bir önceki satır gerekir, bu yüzden tek liste tutulur.
    onceki = list(range(len(b) + 1))
    for i, harf_a in enumerate(a, 1):
        simdiki = [i]
        for j, harf_b in enumerate(b, 1):
            maliyet = 0 if harf_a == harf_b else 1
            simdiki.append(min(onceki[j] + 1, simdiki[j - 1] + 1, onceki[j - 1] + maliyet))
        onceki = simdiki
    return onceki[-1]


def terim_duzelt(kelime):
    """Yanlış yazılmış belge terimini sözlükteki doğru haline çevirir."""
    if len(kelime) < 3 or kelime in TERIMLER:
        return kelime
    for terim in TERIMLER:
        # Kısa terimlerde sadece harf farkına izin var ("bursa" "burs" olmasın).
        izin = 0 if len(terim) <= 5 else (1 if len(terim) <= 7 else 2)
        if edit_distance(sade(kelime), sade(terim)) <= izin:
            return terim
    return kelime


def numara_bul(metin):
    """Metindeki öğrenci numaralarını liste olarak döndürür."""
    return [m.group(0) for m in NUMARA.finditer(metin)]


def normalize(metin):
    """Mesajı modellerin göreceği sade forma getirir, numarayı <no> ile gizler."""
    metin = NUMARA.sub(" <no> ", metin or "")
    metin = kucuk_harf(metin)
    metin = re.sub(r"([^\W\d_])\1{2,}", r"\1", metin)  # çoooook -> çok
    metin = re.sub(r"[^\W\d_]+", lambda m: terim_duzelt(m.group(0)), metin)
    return re.sub(r"\s+", " ", metin).strip()


def selamlasma_cevabi(metin):
    """ELIZA tarzı kural: mesajın tamamı selamlaşmaysa hazır cevabı döndürür."""
    temiz = kucuk_harf(metin).strip(" .,!?")
    return SELAM.get(temiz)
