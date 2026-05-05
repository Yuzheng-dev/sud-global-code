import argparse
import os
import re
from collections import defaultdict
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import spacy
import torch
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from transformers import AutoModel, AutoTokenizer


CUSTOM_STOPWORDS = {
    "être", "suis", "es", "est", "sommes", "êtes", "sont", "été", "sera", "serait", "sûr", "sûrs",
    "avoir", "ai", "as", "a", "avons", "avez", "ont", "eu", "aura", "aurait", "rien", "ailleurs", "partir",
    "faire", "fait", "font", "dire", "dit", "disent", "aller", "va", "vont", "tantôt", "particulier", "tel",
    "plusieurs", "certains", "quelques", "selon", "depuis", "après", "avant", "majoritairement", "net",
    "pendant", "vers", "entre", "cependant", "néanmoins", "toutefois", "afin", "alors", "aussi", "nette",
    "aujourd", "hui", "hier", "demain", "fois", "cas", "exemple", "outre", "croire", "dits", "référer",
    "autrement", "celui", "celle", "ceux", "celles", "lequel", "laquelle", "lesquels", "lesquelles", "dit",
    "dudit", "ceci", "cela", "ça", "vouloir", "valoir", "pouvoir", "devoir", "devenir", "moins", "premier",
    "venir", "tout", "plus", "un", "deux", "pays", "comme", "ben", "très", "beaucoup", "trop", "peu", "gré",
    "duquel", "toujours", "jamais", "souvent", "parfois", "non", "oui", "où", "quand", "comment", "pourquoi",
    "parallèle", "quel", "quelle", "quels", "quelles", "dont", "sans", "sous", "sur", "dans", "pour", "par",
    "avec", "contre", "chez", "grand", "petit", "autre", "autres", "bien", "mal", "déjà", "encore", "enfin",
    "ensuite", "puis", "donc", "car", "parce", "que", "qui", "quoi", "quelque", "chose", "mon", "ton", "son",
    "notre", "votre", "leur", "mes", "tes", "ses", "nos", "vos", "leurs", "majeur", "monde", "année", "ans",
    "cet", "cette", "ainsi", "jour", "faut", "mieux", "reste", "ferait", "majeurs", "ici", "autant", "peut",
    "veux", "tous", "désormais", "temps", "surtout", "toute", "toutes", "doit", "backward", "notamment",
    "lors", "jusqu", "mois", "également", "côté", "nouvelle", "nouveau",
}


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_stopwords() -> set[str]:
    nltk.download("stopwords", quiet=True)
    return set(stopwords.words("french")).union(CUSTOM_STOPWORDS)


def build_yearly_vectors(
    docs: list[str],
    tokenizer,
    model,
    nlp,
    device: torch.device,
    all_stopwords: set[str],
    min_frequency: int,
) -> KeyedVectors | None:
    word_vectors_dict = defaultdict(list)

    with torch.no_grad():
        for index, doc_text in enumerate(docs):
            clean_text = doc_text.lower().replace("sud global", "sud_global").replace("suds", "sud")
            inputs = tokenizer(
                clean_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                return_offsets_mapping=True,
            )
            model_inputs = {key: value.to(device) for key, value in inputs.items() if key != "offset_mapping"}
            outputs = model(**model_inputs)
            embeddings = outputs.last_hidden_state[0].cpu().numpy()

            spacy_doc = nlp(clean_text)
            for token in spacy_doc:
                lemma = token.lemma_.strip().lower()
                is_valid_word = (
                    len(lemma) > 2
                    and lemma not in all_stopwords
                    and re.match(r"^[a-zàâçéèêëîïôûùüÿñæœ_]+$", lemma)
                )
                if not is_valid_word:
                    continue

                start_token = inputs.char_to_token(token.idx)
                end_token = inputs.char_to_token(token.idx + len(token.text) - 1)
                if start_token is None or end_token is None:
                    continue

                word_vec = embeddings[start_token : end_token + 1].mean(axis=0)
                word_vectors_dict[lemma].append(word_vec)

            if (index + 1) % 100 == 0:
                print(f"   Processed {index + 1} articles...")

    vocab, vecs = [], []
    for word, vectors in word_vectors_dict.items():
        if len(vectors) >= min_frequency:
            vocab.append(word)
            vecs.append(np.mean(vectors, axis=0))

    if not vocab:
        return None

    keyed_vectors = KeyedVectors(vector_size=768)
    keyed_vectors.add_vectors(vocab, vecs)
    return keyed_vectors


def run_word_vector_analysis(args: argparse.Namespace) -> None:
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device()
    print(f"Current compute device in use: {device}")

    print(f"Loading BERT model: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.eval()

    print("Loading spaCy French lemmatization model...")
    nlp = spacy.load(args.spacy_model, disable=["ner", "parser"])
    all_stopwords = build_stopwords()

    df = pd.read_excel(args.input)
    df["Year"] = df[args.date_column].astype(str).str.extract(r"^(\d{4})")

    rows = []
    years = [year.strip() for year in args.years.split(",") if year.strip()]
    for year in years:
        docs = df[df["Year"] == year][args.text_column].dropna().astype(str).tolist()
        if not docs:
            print(f"\nNo documents found for {year}; skipping.")
            continue

        print(f"\nProcessing year {year} ({len(docs)} reports in total)...")
        keyed_vectors = build_yearly_vectors(
            docs=docs,
            tokenizer=tokenizer,
            model=model,
            nlp=nlp,
            device=device,
            all_stopwords=all_stopwords,
            min_frequency=args.min_frequency,
        )
        if keyed_vectors is None:
            print(f"No vocabulary met min-frequency={args.min_frequency} for {year}; skipping.")
            continue

        target = "sud_global" if "sud_global" in keyed_vectors.key_to_index else "sud"
        if target not in keyed_vectors.key_to_index:
            print(f"Neither 'sud_global' nor 'sud' appears in the yearly vocabulary for {year}; skipping.")
            continue

        print(f"\n[{year}] Core semantic associated words for '{target}' (Top {args.topn}):")
        similarities = keyed_vectors.most_similar(target, topn=args.topn)
        for rank, (word, similarity) in enumerate(similarities, start=1):
            rows.append(
                {
                    "year": year,
                    "target_word": target,
                    "rank": rank,
                    "neighbor": word,
                    "similarity": similarity,
                }
            )
        for index in range(0, len(similarities), 5):
            print(", ".join([f"{word} ({similarity:.3f})" for word, similarity in similarities[index : index + 5]]))

    if rows:
        output_path = output_dir / "word_vector_neighbors.csv"
        pd.DataFrame(rows).to_csv(output_path, index=False)
        print(f"\nDone. Results saved in: {output_path}")
    else:
        print("\nNo similarity results were generated.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract yearly BERT word-vector neighbors for 'sud global'.")
    parser.add_argument("--input", required=True, help="Path to the Excel corpus file.")
    parser.add_argument("--text-column", default="Texte", help="Name of the column containing article text.")
    parser.add_argument("--date-column", default="Date", help="Name of the column containing article dates.")
    parser.add_argument("--years", default="2022,2023,2024,2025", help="Comma-separated years to analyze.")
    parser.add_argument("--output-dir", default="outputs/word_vectors", help="Directory for generated CSV output.")
    parser.add_argument("--model-name", default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    parser.add_argument("--spacy-model", default="fr_core_news_sm", help="spaCy French model to use.")
    parser.add_argument("--min-frequency", type=int, default=2, help="Minimum yearly occurrences for a lemma.")
    parser.add_argument("--topn", type=int, default=50, help="Number of similar words to report per year.")
    parser.add_argument("--hf-endpoint", default=None, help="Optional Hugging Face endpoint mirror, e.g. https://hf-mirror.com.")
    return parser.parse_args()


if __name__ == "__main__":
    run_word_vector_analysis(parse_args())
