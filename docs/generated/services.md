<!-- generated -->

| Service | Status (2026-10-02) | Why not benchmarked live | Docs |
|---|---|---|---|
| Amazon Personalize | Available (not on AWS's maintenance list as of 2026-10-02). | Real-time campaigns bill per provisioned hour, which risks the <$50/month budget; requests need AWS SigV4 signing (boto3). | [link](https://aws.amazon.com/personalize/) |
| Vertex AI Search for commerce (AI Commerce Search) | Available; retail catalogs only. | Requires a product catalog and user-event import plus model training time; prediction is billed per request. | [link](https://cloud.google.com/retail/docs/overview) |
| Azure AI Personalizer | Retired on 2026-10-01; no new resources since 2023-09-20. | Retired. It was a contextual bandit (choose one action per context), not a collaborative-filtering recommender. | [link](https://learn.microsoft.com/en-us/azure/ai-services/personalizer/) |
