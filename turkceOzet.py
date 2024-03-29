import openai

openai.api_key = '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'


def translate_text(text, target_language="en"):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Translate the following text to {target_language}: \"{text}\"",
        max_tokens=200,  # Increase the value to cover a longer text
        temperature=0.3,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    if 'choices' in response and len(response.choices) > 0:
        translation = response.choices[0].text.strip()
        return translation

    return ""
def translate_turkce(text, target_language="tr"):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Translate the following text to {target_language}: \"{text}\"",
        max_tokens=200,  # Increase the value to cover a longer text
        temperature=0.3,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    if 'choices' in response and len(response.choices) > 0:
        translation = response.choices[0].text.strip()
        return translation

    return ""


def showPaperSummary(paperContent):
    summaries = []  # Özetlenmiş metinleri saklamak için bir liste oluşturuldu
    for page in paperContent:
        text = page.extract_text()
        translation = translate_text(text, target_language="en")
        #GPT-3 modelini kullanarak özeti çıkarır.
        response = openai.Completion.create(
            engine="text-davinci-003", #GPT-3 modeli
            prompt=translation,#giriş metni
            max_tokens=300,#max token sayısı
            temperature=0.3, #rastgelelik ölçümü
            top_p=1.0,# tüm olasılık dağılım sınırı
            frequency_penalty=0.0, #Sıklık cezası
            presence_penalty=0.0 #Varlık cezası
        )

        if 'choices' in response and len(response.choices) > 0:
            summary = response.choices[0].text.strip()
            print("İngilizce Özet:")
            print(summary)
            summaries.append(summary)  # Özetlenmiş metni listeye ekler
            translation = translate_turkce(summary, target_language="tr")
            print("Türkçe Özet:")
            print(translation)
        else:
            print("Özetleme yapılamadı.")

    return summaries