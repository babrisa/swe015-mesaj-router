#!/bin/sh
# macOS: Finder'da cift tiklayinca kurar ve uygulamayi acar. Hazir olunca tarayici acilir.
cd "$(dirname "$0")"
( until curl -s -o /dev/null http://127.0.0.1:7860; do sleep 2; done; open http://127.0.0.1:7860 ) &
./run.sh
