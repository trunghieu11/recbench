<!-- generated -->

| Method | Rung | Status | One-line idea |
|---|---|---|---|
| [Random](random.md) | 0 | bake-off | Random scores: the floor. |
| [MostPopular](most-popular.md) | 0 | bake-off | Recently popular items for everyone. |
| [ItemKNN](itemknn.md) | 1 | bake-off | Items similar (by co-occurrence) to what you used. |
| [SLIM (ElasticNet)](slim.md) | 1 | bake-off | A sparse, non-negative item-to-item weight matrix learned by regression. |
| [RP3beta](rp3beta.md) | 1 | bake-off | Two-step random walks between items, with popular items turned down. |
| [EASE^R](ease.md) | 1 | bake-off | One learned item-to-item weight matrix, solved in closed form. |
| [SANSA](sansa.md) | 1 | bake-off | EASE's weights approximated sparsely, so it scales to large catalogs. |
| [iALS](ials.md) | 2 | bake-off | User and item vectors fitted by weighted least squares. |
| [BPR-MF](bpr-mf.md) | 2 | bake-off | User and item vectors trained so that used items outrank random ones. |
| [PureSVD](puresvd.md) | 2 | bake-off | A truncated SVD of the interaction matrix: the simplest factorisation. |
| [MultVAE](multvae.md) | 2 | bake-off | A variational autoencoder that rebuilds a user's whole history. |
| [RecVAE](recvae.md) | 2 | bake-off | MultVAE with a stronger encoder and a composite prior. |
| [SimpleX](simplex.md) | 2 | bake-off | Matrix factorisation with a cosine contrastive loss and many negatives. |
| [DirectAU](directau.md) | 2 | bake-off | Vectors trained to align used pairs and spread everything else out. |
| [LightGCN](lightgcn.md) | 3 | held back | Vectors smoothed over the user-item graph. |
| [GF-CF](gfcf.md) | 3 | bake-off | A training-free graph filter: normalised co-occurrence plus a low-pass part. |
| [UltraGCN](ultragcn.md) | 3 | bake-off | The effect of infinite graph convolutions, as a weighted loss without message passing. |
| [XSimGCL](xsimgcl.md) | 3 | held back | LightGCN plus a contrastive loss that spreads vectors out. |
| [Turbo-CF](turbocf.md) | 3 | bake-off | A training-free polynomial graph filter, computed on a GPU. |
| [GRU4Rec](gru4rec.md) | 4 | bake-off | A recurrent network reads your clicks in order (the authors' official code). |
| [SASRec](sasrec.md) | 4 | bake-off | A causal Transformer reads your sequence and predicts the next item. |
| [V-SKNN](vsknn.md) | 4 | bake-off | Past sessions similar to your current one, with recent items counting more. |
| [BERT4Rec](bert4rec.md) | 4 | held back | Fill-in-the-blank training on sequences, like BERT for text. |
| [S3-Rec](s3rec.md) | 4 | held back | Self-supervised pre-training with item attributes, then fine-tuning. |
| [TIGER-lite](tiger-lite.md) | 4 | held back | A next-item model with semantic-ID codes (experimental). |
| [HSTU](hstu.md) | 4 | held back | Meta's generative sequence model with pointwise attention and time-aware bias. |
| [LightGBM re-ranker](lgbm-rerank.md) | 5 | bake-off | Cheap models propose candidates; boosted trees re-order them with many features. |
| [DIN](din.md) | 5 | held back | Attention over your history, conditioned on the candidate item. |
| [DCN-V2](dcnv2.md) | 5 | held back | Learned feature crosses for click prediction, over the whole catalog. |
| [DCN-V2 re-ranker](dcnv2-rerank.md) | 5 | bake-off | The same candidates, re-ordered by a neural network that learns feature crosses. |
| [Text hash tower](text-hash-tower.md) | 6 | held back | Items understood from their words; can recommend brand-new items. |
| [Multimodal tower](multimodal-tower.md) | 6 | held back | Text plus image colour features. |
| [Text-embedding kNN](text-knn.md) | 6 | bake-off | Items whose descriptions (encoded by a pretrained model) match your recent items. |
| [Recombee](recombee.md) | 7 | managed | A managed recommendation API, benchmarked like the rest. |
| [Other managed services](managed-services.md) | 7 | docs only | Amazon Personalize, Google's commerce recommender, Azure Personalizer (retired). |

**Status:** *bake-off* = in the quick-tier bake-off, tuned with the same budget as every other method; *held back* = heavier, waiting for its turn through the same gate; *managed* = a hosted service.
