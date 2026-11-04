#!/bin/sh
# Sifirdan calistirir: ortam, paketler, model indirme (olmazsa egitim), uygulama.
set -e
cd "$(dirname "$0")"
[ -d .venv ] || python3.12 -m venv .venv
.venv/bin/pip install -q -r requirements.txt
[ -f models/ogrenci_int8.pt ] || .venv/bin/hf download BarisYildiz/swe015-mesaj-router --local-dir models || .venv/bin/python src/train.py
.venv/bin/python app.py
