import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bertopic import BERTopic

# -------------------------------
# 1. JSON-Datei laden
# -------------------------------
with open("C:\\Users\\dxschecht\\Desktop\\Bonn_Phase_1_Embeddings.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# -------------------------------
# 2. Texte und Embeddings extrahieren
# -------------------------------
texts = [item["original"] for item in data]
embeddings = np.array([item["embedding"] for item in data])

# -------------------------------
# 3. BERTopic-Modell erstellen
# -------------------------------
topic_model = BERTopic(language="multilingual")
topics, probs = topic_model.fit_transform(texts, embeddings)

# -------------------------------
# 4a. Visualisierung: Top-Wörter pro Thema
# -------------------------------
fig1 = topic_model.visualize_barchart(top_n_topics=10)
fig1.write_html("C:\\Users\\dxschecht\\Desktop\\bertopic_barchart.html")

# -------------------------------
# 4b. Visualisierung: Interaktive Themenübersicht
# -------------------------------
fig2 = topic_model.visualize_topics()
fig2.write_html("C:\\Users\\dxschecht\\Desktop\\bertopic_overview.html")

# -------------------------------
# 5. Statische Matplotlib-Grafik mit Top-Wörtern
# -------------------------------
df = topic_model.get_document_info(texts)
topic_counts = df["Topic"].value_counts().sort_index()
labels = [
    f"Thema {i}: {topic_model.get_topic(i)[0][0]}" if i != -1 else "Outlier"
    for i in topic_counts.index
]

plt.figure(figsize=(12, 6))
plt.bar(labels, topic_counts.values)
plt.title("Wichtige Themen nach Häufigkeit")
plt.ylabel("Anzahl der Dokumente")
plt.xlabel("Themen (ID: Top-Wort)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("C:\\Users\\dxschecht\\Desktop\\bertopic_static_plot.png")
plt.show()

# -------------------------------
# 6. Ergebnisse exportieren (bereinigt)
# -------------------------------
df["Top_Wort"] = df["Topic"].apply(
    lambda x: topic_model.get_topic(x)[0][0] if x != -1 else "Outlier"
)
df_clean = df[["Document", "Topic", "Top_Wort", "Probability"]]
df_clean.to_csv("C:\\Users\\dxschecht\\Desktop\\Bonn_Embeddings_CLEAN.csv", index=False)

print("Themenanalyse abgeschlossen und gespeichert.")
