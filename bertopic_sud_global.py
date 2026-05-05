import argparse
import re
from pathlib import Path

import nltk
import pandas as pd
import spacy
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from hdbscan import HDBSCAN
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP


CUSTOM_STOPWORDS = [
    "être", "suis", "es", "est", "sommes", "êtes", "sont", "été", "sera", "serait",
    "avoir", "ai", "as", "a", "avons", "avez", "ont", "eu", "aura", "aurait", "rien",
    "faire", "fait", "font", "dire", "dit", "disent", "aller", "va", "vont",
    "plusieurs", "certains", "quelques", "selon", "depuis", "après", "avant",
    "pendant", "vers", "entre", "cependant", "néanmoins", "toutefois", "afin", "alors", "aussi",
    "aujourd", "hui", "hier", "demain", "fois", "cas", "exemple", "outre", "croire",
    "celui", "celle", "ceux", "celles", "lequel", "laquelle", "lesquels", "lesquelles",
    "ceci", "cela", "ça", "vouloir", "valoir", "pouvoir", "devoir", "devenir", "moins", "premier", "venir",
    "tout", "plus", "un", "deux", "pays", "comme", "ben",
]

VIP_NAMES = [
    "biden", "joe", "trump", "donald", "harris", "kamala", "blinken", "antony", "mohammed", "riyad",
    "françois", "macron", "emmanuel", "scholz", "olaf", "sunak", "rishi", "starmer", "keir", "von",
    "der", "leyen", "ursula", "poutine", "vladimir", "zelensky", "volodymyr", "lavrov", "sergueï",
    "meloni", "xi", "jinping", "modi", "narendra", "netanyahou", "nétanyahou", "benjamin", "erdogan",
    "recep", "tayyip", "lula", "da", "silva", "luiz", "inácio", "bolsonaro", "jair", "ramaphosa",
    "cyril", "millei", "javier",
]

PHRASES_TO_PROTECT = {
    "Union européenne": "Union_européenne",
    "États-Unis": "États_Unis",
    "Etats-Unis": "Etats_Unis",
    "Nations unies": "Nations_unies",
    "Fonds monétaire international": "Fonds_monétaire_international",
    "Banque mondiale": "Banque_mondiale",
    "Organisation mondiale du commerce": "Organisation_mondiale_du_commerce",
    "Cour pénale internationale": "Cour_pénale_internationale",
    "Cour internationale de justice": "Cour_internationale_de_justice",
    "Union africaine": "Union_africaine",
    "Sud global": "Sud_global",
}

PROTECTED_ABBRS = {"ue", "onu", "fmi", "bm", "omc", "cpi", "cij", "ua", "g7", "g20", "g77", "un"}


def build_stopwords() -> set[str]:
    nltk.download("stopwords", quiet=True)
    french_stopwords = stopwords.words("french")
    return set(french_stopwords + CUSTOM_STOPWORDS + VIP_NAMES)


def preprocess_text(text: str, nlp, all_stopwords: set[str]) -> str:
    for old, new in PHRASES_TO_PROTECT.items():
        text = text.replace(old, new)
        text = text.replace(old.lower(), new.lower())

    doc = nlp(text)
    tokens = [
        token.lemma_.lower()
        for token in doc
        if (len(token.lemma_) > 2 or token.lemma_.lower() in PROTECTED_ABBRS)
        and not token.is_punct
        and token.lemma_.lower() not in all_stopwords
    ]
    return " ".join(tokens)


def run_topic_modeling(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("1/5 Read data...")
    df = pd.read_excel(args.input)
    df = df.dropna(subset=[args.text_column])
    docs = df[args.text_column].astype(str).tolist()

    print("2/5 Lemmatization...")
    all_stopwords = build_stopwords()
    nlp = spacy.load(args.spacy_model, disable=["parser", "ner"])
    processed_docs = [preprocess_text(doc, nlp, all_stopwords) for doc in docs]

    print("3/5 Parameter settings...")
    vectorizer_model = CountVectorizer(stop_words=list(all_stopwords), min_df=args.min_df)
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
    hdbscan_model = HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=3,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    representation_model = {
        "Main": MaximalMarginalRelevance(diversity=0.3),
        "KeyBERT": KeyBERTInspired(),
        "MMR": MaximalMarginalRelevance(diversity=0.3),
    }

    topic_model = BERTopic(
        embedding_model=args.embedding_model,
        vectorizer_model=vectorizer_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        representation_model=representation_model,
    )

    print("4/5 Data training...")
    topics, probs = topic_model.fit_transform(processed_docs)

    mmr_labels = topic_model.generate_topic_labels(nr_words=3, separator=", ", aspect="MMR")
    topic_model.set_topic_labels(mmr_labels)

    print("5/5 Generate results...")
    topic_info = topic_model.get_topic_info()
    topic_info.to_excel(output_dir / "full_report.xlsx", index=False)

    doc_info = topic_model.get_document_info(processed_docs)
    doc_info["Original_Text"] = docs

    wide_dict = {}
    for topic_id in doc_info["Topic"].unique():
        texts = doc_info[doc_info["Topic"] == topic_id]["Original_Text"].tolist()
        wide_dict[f"Topic {topic_id}"] = pd.Series(texts)

    wide_df = pd.DataFrame(wide_dict)
    sorted_cols = sorted(wide_df.columns, key=lambda x: int(x.replace("Topic ", "")))
    wide_df = wide_df[sorted_cols]
    wide_df.to_excel(output_dir / "all_docs_by_topic_original_text.xlsx", index=False)

    if len(topic_info) > 1:
        fig_bar = topic_model.visualize_barchart(
            top_n_topics=args.top_n_topics,
            n_words=5,
            custom_labels=True,
        )

        for annotation in fig_bar.layout.annotations:
            match = re.match(r"^\s*(-?\d+)", annotation.text)
            if match:
                topic_id = match.group(1)
                annotation.text = f"Topic {topic_id}"
                annotation.font.size = 14

        fig_bar.write_html(output_dir / "bar_chart_of_topic_terms.html")

    print("Generating scatter plot...")
    try:
        top_topic_indices = topic_info[topic_info["Topic"] != -1].head(args.top_n_topics)["Topic"].tolist()
        fig_docs = topic_model.visualize_documents(
            docs,
            topics=top_topic_indices,
            width=1000,
            height=750,
            custom_labels=True,
        )
        fig_docs.update_traces(marker=dict(size=4.5, opacity=0.8, line=dict(width=0.5, color="White")))
        fig_docs.write_html(output_dir / "2d_scatter_plot.html")
    except Exception as exc:
        print(f"Error while plotting: {exc}")

    print(f"Done. Results saved in: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BERTopic analysis for a French Global South media corpus.")
    parser.add_argument("--input", required=True, help="Path to the Excel corpus file.")
    parser.add_argument("--text-column", default="Texte", help="Name of the column containing article text.")
    parser.add_argument("--output-dir", default="outputs/bertopic", help="Directory for generated reports and plots.")
    parser.add_argument("--spacy-model", default="fr_core_news_md", help="spaCy French model to use.")
    parser.add_argument("--embedding-model", default="paraphrase-multilingual-mpnet-base-v2", help="BERTopic embedding model.")
    parser.add_argument("--min-df", type=int, default=10, help="Minimum document frequency for CountVectorizer.")
    parser.add_argument("--min-cluster-size", type=int, default=15, help="Minimum HDBSCAN cluster size.")
    parser.add_argument("--top-n-topics", type=int, default=16, help="Number of topics to visualize.")
    return parser.parse_args()


if __name__ == "__main__":
    run_topic_modeling(parse_args())
