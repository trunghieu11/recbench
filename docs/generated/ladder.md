<!-- generated -->

| Rung | Idea it adds | Methods |
|---|---|---|
| 0 — Baselines | "no learning" reference points | [Random](random.md), [MostPopular](most-popular.md) |
| 1 — Neighbourhood and linear models | items that co-occur are related | [ItemKNN](itemknn.md), [SLIM (ElasticNet)](slim.md), [RP3beta](rp3beta.md), [EASE^R](ease.md), [SANSA](sansa.md) |
| 2 — Matrix factorisation | users and items as learned vectors | [iALS](ials.md), [BPR-MF](bpr-mf.md), [PureSVD](puresvd.md), [MultVAE](multvae.md), [RecVAE](recvae.md), [SimpleX](simplex.md), [DirectAU](directau.md) |
| 3 — Graph methods | neighbours of neighbours carry signal | [LightGCN](lightgcn.md), [GF-CF](gfcf.md), [UltraGCN](ultragcn.md), [XSimGCL](xsimgcl.md), [Turbo-CF](turbocf.md) |
| 4 — Sequential models | the *order* of a user's history matters | [GRU4Rec](gru4rec.md), [SASRec](sasrec.md), [V-SKNN](vsknn.md), [BERT4Rec](bert4rec.md), [S3-Rec](s3rec.md), [TIGER-lite](tiger-lite.md), [HSTU](hstu.md) |
| 5 — CTR-style rankers and re-rankers | score one (user, item) pair at a time with rich features | [LightGBM re-ranker](lgbm-rerank.md), [DIN](din.md), [DCN-V2](dcnv2.md), [DCN-V2 re-ranker](dcnv2-rerank.md) |
| 6 — Content-based methods | understand items from their text and images | [Text hash tower](text-hash-tower.md), [Multimodal tower](multimodal-tower.md), [Text-embedding kNN](text-knn.md) |
| 7 — Managed services | rent a recommender instead of building one | [Recombee](recombee.md), [other services](managed-services.md) |
