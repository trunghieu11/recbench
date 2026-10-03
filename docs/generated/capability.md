<!-- generated -->

| Method | Rung | Family | Tasks | Uses order | New items | Needs content | Fidelity | Ranked |
|---|---|---|---|---|---|---|---|---|
| [MostPopular](most-popular.md) | 0 | Baseline | topn, sequential, similar_items | no | yes | no | faithful | yes |
| [Random](random.md) | 0 | Baseline | topn, sequential | no | yes | no | faithful | yes |
| [EASE^R](ease.md) | 1 | Linear autoencoder (item-item) | topn, sequential, similar_items | yes | no | no | faithful | yes |
| [ItemKNN](itemknn.md) | 1 | Neighbourhood collaborative filtering | topn, sequential, similar_items | yes | no | no | faithful | yes |
| [RP3beta](rp3beta.md) | 1 | Graph random walk (item-item) | topn, sequential, similar_items | yes | no | no | faithful | yes |
| [SANSA](sansa.md) | 1 | Linear autoencoder (sparse approximate EASE) | topn, sequential, similar_items | yes | no | no | faithful | yes |
| [SLIM (ElasticNet)](slim.md) | 1 | Sparse linear model (item-item) | topn, sequential, similar_items | yes | no | no | faithful | yes |
| [BPR-MF](bpr-mf.md) | 2 | Matrix factorisation (pairwise loss) | topn, sequential, similar_items | no | no | no | faithful | yes |
| [DirectAU](directau.md) | 2 | Matrix factorisation (alignment and uniformity) | topn, sequential, similar_items | no | no | no | faithful | yes |
| [iALS](ials.md) | 2 | Matrix factorisation (weighted least squares) | topn, sequential, similar_items | no | no | no | faithful | yes |
| [MultVAE](multvae.md) | 2 | Variational autoencoder | topn, sequential, similar_items | yes | no | no | faithful | yes |
| [PureSVD](puresvd.md) | 2 | Matrix factorisation (truncated SVD) | topn, sequential, similar_items | yes | no | no | faithful | yes |
| [RecVAE](recvae.md) | 2 | Variational autoencoder | topn, sequential, similar_items | yes | no | no | faithful | yes |
| [SimpleX](simplex.md) | 2 | Matrix factorisation (cosine contrastive loss) | topn, sequential, similar_items | yes | no | no | faithful | yes |
| [GF-CF](gfcf.md) | 3 | Graph filter (training-free) | topn, sequential, similar_items | yes | no | no | faithful | yes |
| [LightGCN](lightgcn.md) | 3 | Graph collaborative filtering | topn, sequential, similar_items | no | no | no | faithful | yes |
| [Turbo-CF](turbocf.md) | 3 | Graph filter (training-free, polynomial) | topn, sequential, similar_items | yes | no | no | faithful | yes |
| [UltraGCN](ultragcn.md) | 3 | Graph-weighted matrix factorisation | topn, sequential, similar_items | no | no | no | faithful | yes |
| [XSimGCL](xsimgcl.md) | 3 | Graph contrastive learning | topn, sequential, similar_items | no | no | no | faithful | yes |
| [BERT4Rec](bert4rec.md) | 4 | Sequential (bidirectional masked modelling) | topn, sequential, session, similar_items | yes | no | no | faithful | yes |
| [GRU4Rec](gru4rec.md) | 4 | Recurrent neural network (sequential) | topn, sequential, session | yes | no | no | faithful | yes |
| [HSTU](hstu.md) | 4 | Generative sequential transducer | topn, sequential, session, similar_items | yes | no | no | simplified | yes |
| [S3-Rec](s3rec.md) | 4 | Sequential (self-supervised pre-training) | topn, sequential, session, similar_items | yes | no | yes | faithful | yes |
| [SASRec](sasrec.md) | 4 | Sequential (causal self-attention) | topn, sequential, session, similar_items | yes | no | no | faithful | yes |
| [TIGER-lite](tiger-lite.md) | 4 | Generative retrieval (semantic IDs), simplified | topn, sequential | yes | no | no | simplified | no |
| [V-SKNN](vsknn.md) | 4 | Session-based nearest neighbours | topn, sequential, session | yes | no | no | simplified | yes |
| [DCN-V2](dcnv2.md) | 5 | CTR ranker with feature crosses | topn, ctr | no | no | no | faithful | yes |
| [DIN](din.md) | 5 | CTR ranker with target attention | topn, sequential, ctr | yes | no | no | faithful | yes |
| [Multimodal tower](multimodal-tower.md) | 6 | Content-based two-tower with images | topn, sequential, similar_items | yes | yes | yes | simplified | yes |
| [Text hash tower](text-hash-tower.md) | 6 | Content-based two-tower (stand-in for LLM text encoders) | topn, sequential, similar_items | yes | yes | yes | simplified | yes |
| [Recombee](recombee.md) | 7 | Managed recommendation service (SaaS) | topn | no | yes | no | faithful | yes |
