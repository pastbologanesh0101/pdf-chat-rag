# AI PDF Chat (RAG)

A small, fully working, **fully local** retrieval-augmented generation (RAG) pipeline over
PDFs — chunking, TF-IDF embedding, cosine-similarity retrieval, and cited extractive
answers. **No LLM API is called anywhere in this project** (no OpenAI/Anthropic/etc key
required, no network call at query time).

## What "RAG without an LLM" means here

RAG has two halves:

1. **Retrieval** — turn a document into searchable chunks, embed them, and find the
   chunks most relevant to a query. This is the hard engineering part, and it is what
   this project implements in full: PDF ingestion, overlapping chunking with page
   provenance, TF-IDF vector indexing, and cosine-similarity search.
2. **Generation** — hand the retrieved chunks to a large language model and have it
   write a fluent, synthesized answer grounded in that context.

This project deliberately implements **only the first half**, honestly. Instead of a
generative step, it performs **extractive answer synthesis**: it finds the sentence(s)
within the retrieved chunks that most overlap with your query and returns them verbatim,
with a citation. Nothing is generated; every word of the "answer" comes straight out of
the source PDF. This is a legitimate, self-contained way to demonstrate the retrieval
engineering that RAG is built on, without requiring an API key or making any external
network call.

**Where a real LLM call would go:** see the module docstring at the top of
[`rag/synthesize.py`](rag/synthesize.py). In short, `synthesize_answer()` is the function
you would replace with a call like:

```python
prompt = build_prompt(query, retrieved_chunks)
answer = anthropic_client.messages.create(
    model="claude-...",
    messages=[{"role": "user", "content": prompt}],
)
```

Everything upstream of that function — ingest, chunk, index, retrieve — is unchanged
either way; that upstream pipeline *is* the "retrieval-augmented" part of RAG, and it is
fully implemented and running locally here.

## Architecture

```
 PDF file
    │
    ▼
 ingest.py      extract per-page text          (pypdf)
    │
    ▼
 chunker.py     split each page's text into overlapping word-window
                chunks; every chunk keeps its 1-indexed source page number
    │
    ▼
 index.py       TF-IDF vectorize all chunks; cosine-similarity search
                returns the top-K chunks for a query               (scikit-learn)
    │
    ▼
 synthesize.py  pick the sentence(s) within the top-K chunks that most
                overlap the query (TF-IDF + cosine similarity again,
                at sentence granularity) and return them with a
                "(page X, chunk Y)" citation — or a clear
                "no relevant content found" message if nothing matches
    │
    ▼
 chat.py        CLI: prints the answer + citations; --interactive
                starts a REPL for follow-up questions against the
                same indexed document
```

`ingest -> chunk -> embed/index -> retrieve -> synthesize` — the standard RAG pipeline,
with a template/extractive synthesis step standing in for an LLM call.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

(`reportlab` is only needed to regenerate the bundled sample PDFs; `pypdf` and
`scikit-learn` are the only runtime dependencies of `chat.py` itself.)

## Run

```bash
python chat.py sample_docs/hydroponics_guide.pdf "What is the main conclusion?"
```

```
Q: What is the main conclusion?
A: The main conclusion of this guide is that a simple deep water culture setup, built
from a plastic tote and an aquarium air pump, is the cheapest reliable way for a
beginner to start growing leafy greens indoors year round. (page 1, chunk 0)

Top retrieved chunks:
  [chunk 0 | page 1 | score 0.316] Backyard Hydroponics: A Practical Guide Hydroponics
  is a method of growing plants without soil, in which the roots sit d...
```

Another example, against the second sample document:

```bash
python chat.py sample_docs/lighthouse_history.pdf "When was the Pharos of Alexandria built?"
```

```
Q: When was the Pharos of Alexandria built?
A: The earliest known lighthouse, the Pharos of Alexandria, was built around 280 BC and
is believed to have stood over 100 meters tall, making it one of the tallest structures
in the ancient world. (page 1, chunk 0)
```

And a query with no relevant content in the document, showing the honest fallback
instead of a hallucinated answer:

```bash
python chat.py sample_docs/lighthouse_history.pdf "What is the capital of France?"
```

```
Q: What is the capital of France?
A: No relevant content found for this query in the document.
```

### Interactive mode

```bash
python chat.py sample_docs/hydroponics_guide.pdf --interactive
```

Ask follow-up questions against the same indexed document; type `quit` or `exit` to leave.

### Options

```
python chat.py document.pdf "question" \
    --top-k 3          # number of chunks to retrieve (default 3)
    --chunk-size 120    # words per chunk (default 120)
    --overlap 30         # overlapping words between consecutive chunks (default 30)
```

## Sample documents

`sample_docs/hydroponics_guide.pdf` and `sample_docs/lighthouse_history.pdf` are small,
original, multi-page PDFs generated by `sample_docs/generate_samples.py` (via
`reportlab`) specifically for this project — not copied from anywhere. Regenerate them
with:

```bash
python sample_docs/generate_samples.py
```

## Tests

```bash
pip install pytest
pytest tests/ -v
```

23 unit/integration tests cover: PDF text extraction (non-empty text, correct page
count), chunk size/overlap behavior, per-chunk page-number correctness, genuinely
relevant retrieval on a crafted deterministic document, citation formatting, empty and
very-short document handling, and the "no relevant content found" fallback for
off-topic queries.

CI (`.github/workflows/tests.yml`) runs the suite on every push/PR against Python 3.11
and 3.12.

## Project layout

```
chat.py                    CLI entry point
rag/
  ingest.py                PDF -> per-page text (pypdf)
  chunker.py                per-page text -> overlapping, page-tagged chunks
  index.py                  TF-IDF index + cosine similarity search (scikit-learn)
  synthesize.py              extractive answer + citation
sample_docs/
  generate_samples.py        regenerates the bundled sample PDFs
  hydroponics_guide.pdf
  lighthouse_history.pdf
tests/                       23 unit/integration tests
.github/workflows/tests.yml  CI
```

## License

MIT — see [LICENSE](LICENSE).
