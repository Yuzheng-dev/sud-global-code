# Sud Global Text Analysis

Python scripts for analyzing a French media corpus about the Global South.

## Contents

- `bertopic_sud_global.py`: topic modeling with BERTopic, UMAP, HDBSCAN, and French lemmatization.
- `word_vector_sud_global.py`: yearly BERT-based word-vector analysis for terms related to `sud_global`.
- `full_report_public.xlsx`: public topic report with representative full-text documents removed.

The full news discourse corpus is not publicly included because releasing all collected reporting texts may raise copyright concerns. If you need access to the full `Texte` column and the related `Date` metadata for research purposes, please contact the author at yzwu25@stu.pku.edu.cn.

The scripts expect an Excel file with at least:

- `Texte`: article text
- `Date`: article date, required by `word_vector_sud_global.py`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download fr_core_news_md
python -m spacy download fr_core_news_sm
```

## Run BERTopic

```bash
python bertopic_sud_global.py --input "path/to/corpus.xlsx" --output-dir outputs/bertopic
```

Main outputs:

- `full_report.xlsx`
- `all_docs_by_topic_original_text.xlsx`
- `bar_chart_of_topic_terms.html`
- `2d_scatter_plot.html`

## Run Word-Vector Analysis

```bash
python word_vector_sud_global.py --input "path/to/corpus.xlsx" --output-dir outputs/word_vectors
```

To use a Hugging Face mirror:

```bash
python word_vector_sud_global.py --input "path/to/corpus.xlsx" --hf-endpoint https://hf-mirror.com
```

Main output:

- `word_vector_neighbors.csv`
