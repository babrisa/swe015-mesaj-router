"""Dört modeli eğitir ve models/ altına kaydeder."""
import os

from models import bert_egit, ffn_egit, ogrenci_quantize_kaydet, tfidf_egit, veri_yukle

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    metinler, etiketler = veri_yukle("train")
    tfidf_egit(metinler, etiketler)
    ffn_egit(metinler, etiketler)
    bert_egit(metinler, etiketler, "models/bert")  # öğretmen: 12 katman
    bert_egit(metinler, etiketler, "models/ogrenci", katman=6, tur=12, ogretmen="models/bert")  # öğrenci: 6 katman
    ogrenci_quantize_kaydet()
