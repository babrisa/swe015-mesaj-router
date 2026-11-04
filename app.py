"""Gradio arayüzü: quantize öğrenci BERTurk ile yönlendirir, dört modeli yan yana gösterir."""
import sys

import gradio as gr

sys.path.insert(0, "src")
from models import CONFIG  # noqa: E402
from router import route  # noqa: E402

ORNEKLER = ["öğrenci belgesi lazım no 2401010123", "harç ödememi yapamadım", "transkirpt nasıl alınıyor, numaram 240101123",
            "merhaba", "yarın hava nasıl olacak"]


def yonlendir(mesaj, esik):
    """Arayüzden gelen mesajı route() ile işler, kutulara dağıtır."""
    s = route(mesaj, esik)
    return s["kart"], s["birim"], s["niyet"], s["guven"], s["metin"], s["tablo"]


with gr.Blocks(title="Mesaj Yönlendirici") as demo:
    gr.Markdown("# Mesaj Yönlendirici\nİstinye Üniversitesi · Üretim modeli: **Damıtılmış BERTurk, INT8 (CPU)**")
    mesaj = gr.Textbox(label="Öğrenci mesajı", lines=2)
    esik = gr.Slider(0, 1, value=CONFIG["threshold"], step=0.01, label="Güven eşiği")
    buton = gr.Button("Yönlendir", variant="primary")
    kart = gr.Textbox(label="Talep kartı")
    with gr.Row():
        birim = gr.Textbox(label="Birim")
        niyet = gr.Textbox(label="Niyet")
        guven = gr.Number(label="Güven")
    metin = gr.Textbox(label="Normalize metin")
    tablo = gr.Dataframe(headers=["Model", "Niyet", "Güven", "Birim"], label="Dört model yan yana")
    gr.Examples(ORNEKLER, inputs=mesaj)
    cikti = [kart, birim, niyet, guven, metin, tablo]
    buton.click(yonlendir, [mesaj, esik], cikti, api_name="yonlendir")
    mesaj.submit(yonlendir, [mesaj, esik], cikti)

if __name__ == "__main__":
    demo.launch()
