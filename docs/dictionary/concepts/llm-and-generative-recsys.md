# LLMs and generative recommenders

## Why it matters

Since 2023, two new ideas have reshaped recommendation research and industry:

1. **Large language models (LLMs)** used directly as recommenders, or to understand items and users from text.
2. **Generative recommenders:** models that *generate* the next item (or its identifier) the way a language
   model generates the next word, and that keep improving as they get bigger (scaling laws).

## The landscape

| Approach | Idea | Examples |
|---|---|---|
| LLM as a recommender | describe the user's history in a prompt; ask the LLM to recommend or rank | P5 (2022), TALLRec (2023) |
| LLM as a zero-shot re-ranker | give the LLM a short candidate list and the history; it reorders them | Hou et al. (2023) |
| LLM as an encoder | turn item text (titles, descriptions, reviews) into embeddings for a two-tower model | sentence encoders, LLM embeddings |
| Generative retrieval | give items short "semantic IDs" and generate the next ID token by token | TIGER (2023) |
| Generative recommenders at scale | treat user actions as a sequence modelling problem with a new architecture and scale it up | HSTU (Meta, 2024), followed by industrial systems at other companies |

## Intuition

- **Why LLMs?** They bring world knowledge ("people who like *Dune* often like *Foundation*") and read text,
  which helps with new items and sparse data, and they can explain in natural language.
- **Why not only LLMs?** They are expensive per request, struggle with huge catalogs (they cannot score
  millions of items), can recommend items that do not exist, and do not automatically know your users'
  behaviour. In practice they are often used as re-rankers or encoders next to classic retrieval.
- **Why generative recommenders?** Scaling. HSTU's authors showed that quality keeps improving with more
  compute and data, as it does for language models.

## A small example: LLM re-ranking

Prompt: "The user watched *Toy Story*, *Up*, and *Inside Out*. Rank these candidates: *Heat*, *Coco*,
*Alien*." A capable LLM answers *Coco* first, using what it knows about Pixar films. No interaction data
about *Coco* was needed. Retrieval (which produced the three candidates) still came from a classic model.

## What recbench includes today

| In recbench | Relation to this landscape |
|---|---|
| [HSTU](../algorithms/hstu.md) | a small, faithful-in-spirit generative recommender (attention verified against Meta's code) |
| [TIGER-lite](../algorithms/tiger-lite.md) | semantic IDs via residual quantisation, without generative retrieval (unranked) |
| [Text hash tower](../algorithms/text-hash-tower.md) | the "encoder" pattern with a hashed bag of words instead of an LLM |

## What is on the roadmap

- An LLM zero-shot re-ranker on top of a retriever (for example EASE top-50 → LLM).
- Replacing hashed text with pretrained sentence or LLM embeddings in the content towers.
- A faithful TIGER (RQ-VAE on item content, encoder-decoder, beam search).

See the [roadmap](../../results/roadmap.md).

## Pitfalls

- **Hallucinated items:** an LLM can recommend titles that are not in your catalog. Constrain it to candidates.
- **Leakage through pretraining:** an LLM may already "know" famous public datasets, inflating results.
- **Cost:** one LLM call per user per request adds up quickly; measure it like any other method.

## Check your understanding

??? question "Why are LLMs usually used as re-rankers rather than retrievers?"
    They cannot score millions of items per request. A cheap retriever narrows the catalog to a few dozen
    candidates, which fit in a prompt.

??? question "What makes TIGER 'generative'?"
    It produces the next item's semantic ID token by token with a decoder, instead of scoring every item
    with a dot product.

## Further reading

- Geng et al. (2022), [Recommendation as Language Processing (P5)](https://arxiv.org/abs/2203.13366).
- Bao et al. (2023), [TALLRec](https://arxiv.org/abs/2305.00447).
- Hou et al. (2023), [Large Language Models are Zero-Shot Rankers for Recommender Systems](https://arxiv.org/abs/2305.08845).
- Rajput et al. (2023), [TIGER](https://arxiv.org/abs/2305.05065); Zhai et al. (2024), [HSTU](https://arxiv.org/abs/2402.17152).
