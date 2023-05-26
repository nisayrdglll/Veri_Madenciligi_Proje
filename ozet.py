import re
import bs4 as bs
import urllib.request
import nltk
import heapq

with open(r'C:\Users\user\Desktop\anahtarKelime\deneme.txt', 'r', encoding='utf-8') as file:
    content = file.read()


    def summarize_text(text):
        with open(r'C:\Users\user\Desktop\anahtarKelime\deneme.txt', 'r', encoding='utf-8') as file:
            article_text = file.read()

        article_text = re.sub(r'\[[0-9]*\]', ' ', text)
        article_text = re.sub(r'\s+', ' ', article_text)

        # Özel karakterleri ve rakamları kaldırma
        formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text)
        formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)

        sentence_list = nltk.sent_tokenize(article_text)

        stopwords = nltk.corpus.stopwords.words('turkish')

        word_frequencies = {}
        for word in nltk.word_tokenize(formatted_article_text):
            if word not in stopwords:
                if word not in word_frequencies.keys():
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1

        maximum_frequency = max(word_frequencies.values())

        for word in word_frequencies.keys():
            word_frequencies[word] = (word_frequencies[word] / maximum_frequency)

        sentence_scores = {}
        for sent in sentence_list:
            for word in nltk.word_tokenize(sent.lower()):
                if word in word_frequencies.keys():
                    if len(sent.split(' ')) < 30:
                        if sent not in sentence_scores.keys():
                            sentence_scores[sent] = word_frequencies[word]
                        else:
                            sentence_scores[sent] += word_frequencies[word]

        summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)
        summary = ' '.join(summary_sentences)
        return summary
