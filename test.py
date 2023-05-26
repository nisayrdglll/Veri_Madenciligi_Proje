import heapq
import re
import sys
import nltk
import io

from Desktop.anahtarKelime.ozet import  summarize_text

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from keywords import AnahtarKelimeler
from nltk.corpus import stopwords


with open('deneme.txt', 'r', encoding="utf8") as f1:
    corpus_1 = f1.read()

stopWords = stopwords.words('english')

# Anahtar kelimeleri hesaplamak
anahtar_kelimeler = AnahtarKelimeler(corpus=corpus_1, stop_words=stopWords, alpha=0.8)
d = anahtar_kelimeler.anahtar_kelimeleri_al(corpus_1, n=20)
for i in d:
    print("Anahtar Kelime: %s\nSkor: %f" % (i[0], i[1]))

# Metin özetini almak

summary = summarize_text(corpus_1)
print("Metnin ozeti:")
print(summary)
