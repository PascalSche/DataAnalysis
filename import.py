import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF

# === 1. TF-IDF-Matrix laden ===
csv_path = "C:\\Users\\dxschecht\\Desktop\\Bonn_TFIDF.csv"  # anpassen
tfidf_df = pd.read_csv(csv_path)

# === 2. NMF-Modell anwenden ===
n_topics = 20  # Anzahl der Themen (anpassbar)
nmf_model = NMF(n_components=n_topics, random_state=42)
W = nmf_model.fit_transform(tfidf_df)
H = nmf_model.components_

# === 3. Top-Wörter je Thema anzeigen ===
feature_names = tfidf_df.columns
n_top_words = 10

for topic_idx, topic in enumerate(H):
    top_indices = topic.argsort()[:-n_top_words - 1:-1]
    top_words = [feature_names[i] for i in top_indices]
    print(f"\n🧠 Thema {topic_idx + 1}: {', '.join(top_words)}")

# === 4. Visualisierung mit Matplotlib ===
fig, axes = plt.subplots(n_topics, 1, figsize=(10, n_topics * 2.5))
for i, topic in enumerate(H):
    top_indices = topic.argsort()[:-n_top_words - 1:-1]
    top_words = [feature_names[i] for i in top_indices]
    top_weights = topic[top_indices]

    ax = axes[i]
    ax.barh(top_words[::-1], top_weights[::-1])
    ax.set_title(f"Thema {i + 1}")
    ax.set_xlabel("Gewicht (Relevanz)")
    ax.grid(True)

plt.tight_layout()
plt.show()
