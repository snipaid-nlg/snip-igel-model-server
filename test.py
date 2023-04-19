# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import requests

model_inputs = {
    'task_prefix': '### Nachfolgend ist eine Anweisung, die eine Aufgabe beschreibt, zusammen mit einer Eingabe, die weiteren Kontext liefert. Schreiben Sie eine Antwort, die die Aufgabe angemessen erfüllt.\nAnweisung: Was sind die wichtigsten Keywords?\n### Eingabe:',
    'document': 'Immerhin über 10.000 Produkte machen das schwedische Unternehmen Ikea zu einem der größten Möbel- und Einrichtungshäuser der Welt. Schon länger gibt es die App Ikea Place, mit deren Hilfe ihr Möbel aus dem Katalog in 3D per Augmented Reality in euer Zuhause einzublenden. 3D und AR direkt aus der Suche heraus: So könnt ihr unproblematisch sehen, ob sich ein Einrichtungsgegenstand in eurer Wohnung tatsächlich so gut macht wie anhand des Katalogbilds erwartet. Mit den Erweiterungen der Google-Suche wird diese App künftig nicht mehr der exklusive Zugang zu den 3D-Modellen der Schweden bleiben. Denn wie 9to5google berichtet, hat Google nun den gesamten Katalog über die eigene Suchfunktion verfügbar gemacht. Bei der Suche nach einem IKEA-Artikel wie z. B. einem BROR-Servierwagen auf dem Mobiltelefon wird unter dem entsprechenden Angebot im Webshop ein Umschalter für die 3D-Ansicht angezeigt.',
    'prompt': '\n### Antwort:',
    'params': {
        'min_new_tokens': 3,
        'max_new_tokens': 20,
        'top_p': 0.9,
        'top_k': 0,
        'temperature': 1,
        'do_sample': True,
        'early_stopping': True,
        'length_penalty': 1,
        'eos_token_id': 2,
        'pad_token_id': 2,
    }
}

res = requests.post('http://localhost:8000/', json = model_inputs)

print(res.json())