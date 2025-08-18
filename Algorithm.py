import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np

# NLTK vorbereiten
nltk.download('punkt')
nltk.download('stopwords')

# Spacy Modell laden
nlp = spacy.load("de_core_news_sm")

# Stoppwörter definieren
stop_words = set(stopwords.words('german'))  # alternativ: deutsch, falls du NLTK DE-Stoppwörter nutzt

def preprocess_text(text):
    # 1. Kleinbuchstaben
    text = text.lower()

    # 2. Entferne Tags, Sonderzeichen, Zahlen
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-äöüßz\s]', '', text)

    # 3. Tokenisierung (SpaCy statt NLTK)
    doc = nlp(text)
    tokens = [token.text for token in doc]

    # 4. Entferne Stoppwörter
    tokens = [token for token in tokens if token not in stop_words]

    # 5. Lemmatisierung
    lemmatized = [token.lemma_ for token in nlp(" ".join(tokens)) if token.lemma_ != "-PRON-"]

    return lemmatized

def process_json_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed_data = []
    tfidf_texts = []  # <-- Neue Liste für TF-IDF Texte

    for entry in data:
        text_parts = []

        for key in ["service_name", "title", "description"]:
            if key in entry and isinstance(entry[key], str) and entry[key].strip():
                text_parts.append(entry[key])

        if text_parts:
            combined_text = " ".join(text_parts)
            clean_tokens = preprocess_text(combined_text)
            processed_data.append({
                "original": combined_text,
                "processed": clean_tokens
            })
            tfidf_texts.append(" ".join(clean_tokens))  # <-- hier sammeln für TF-IDF

    return processed_data, tfidf_texts  # <-- beide Rückgaben


if __name__ == "__main__":
    input_path = "C:\\Users\\dxschecht\\Desktop\\output_cleaned.json"
    processed_data, tfidf_input = process_json_file(input_path)

####### HIer beginnt der Satz-Embeddingteil #######

    # Modell laden
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  # Gut für deutsch & englisch

     #Embeddings berechnen
    embeddings = model.encode(tfidf_input, show_progress_bar=True)

     #Optional: Embeddings mit Originaltexten speichern
    vectorized_data = [
        {
            "original": processed_data[i]["original"],
            "processed": processed_data[i]["processed"],
            "embedding": embeddings[i].tolist()  # JSON-kompatibel
        }
        for i in range(len(processed_data))
    ]

     #Speichern der Embeddings
    with open("C:\\Users\\dxschecht\\Desktop\\Bonn_Phase_1_Embeddings.json", "w", encoding="utf-8") as f_vec:
        json.dump(vectorized_data, f_vec, ensure_ascii=False, indent=4)

####### Hier endet der Satz-Embeddingteil #######

####### Das hier ist der TDIDF-Vektorisierungsteil ########

    # Speichern der verarbeiteten Tokens
    #with open("C:\\Users\\dxschecht\\Desktop\\Bonn_Phase_1.json", "w", encoding="utf-8") as f_out:
     #   json.dump(processed_data, f_out, ensure_ascii=False, indent=4)

    # TF-IDF Verarbeitung
    #from sklearn.feature_extraction.text import TfidfVectorizer
    import pandas as pd

    #vectorizer = TfidfVectorizer()
    #tfidf_matrix = vectorizer.fit_transform(tfidf_input)

    #df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
   # df_tfidf.to_csv("C:\\Users\\dxschecht\\Desktop\\Bonn_TFIDF.csv", index=False)

########### Hier hört der TDIDF-Vektorisierungsteil auf #######
    print("Phase 1 & 2 abgeschlossen.")