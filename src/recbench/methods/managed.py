"""Managed recommenders. Live calls happen only when credentials are already set."""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

from recbench.protocol import Explanation, MethodSpec, Recommender, Task, Unsupported
from recbench.registry import register_method


class ManagedMethod(Recommender):
    env_keys: tuple[str, ...] = ()

    def credentials(self) -> bool:
        return all(os.environ.get(key) for key in self.env_keys)

    def fit(self, store: Any, config: dict[str, Any]) -> None:
        if not config.get("managed_services", False):
            raise Unsupported(f"{self.spec.name} is not on the full-tier board")
        if not self.credentials():
            raise Unsupported(f"{self.spec.name} has no credentials")
        self.store = store
        self._fitted = True

    def score_candidates(self, store: Any, candidates: Any = None):
        raise Unsupported(f"{self.spec.name} is scored only on a smoke campaign, not the shared file")

    def save(self, path: str) -> None:
        return None

    def load(self, path: str) -> None:
        self._fitted = True


@register_method
class AmazonPersonalize(ManagedMethod):
    spec = MethodSpec(
        name="personalize",
        tasks={Task.topn},
        feedback={"implicit", "explicit"},
        managed=True,
        can_score_candidates=False,
        cost_band="high",
        upstream="Amazon Personalize campaign",
    )
    env_keys = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "RECBENCH_PERSONALIZE_CAMPAIGN_ARN")

    def recommend(self, user_id: str, k: int = 10) -> list[dict[str, Any]]:
        if not self.credentials():
            raise Unsupported("Personalize credentials are not set")
        # The campaign call is intentionally thin: one GetRecommendations request.
        payload = json.dumps(
            {
                "campaignArn": os.environ["RECBENCH_PERSONALIZE_CAMPAIGN_ARN"],
                "userId": user_id,
                "numResults": k,
            }
        ).encode()
        request = urllib.request.Request(
            os.environ.get(
                "RECBENCH_PERSONALIZE_URL",
                "https://personalize-runtime.us-east-1.amazonaws.com/recommendations",
            ),
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode())
        items = body.get("itemList") or body.get("items") or []
        return [
            {
                "item_id": str(item.get("itemId") or item.get("item_id")),
                "score": float(item.get("score") or 0),
                "explanation": Explanation("local", "Amazon Personalize campaign score", []).as_dict(),
            }
            for item in items[:k]
        ]


@register_method
class RecombeeMethod(ManagedMethod):
    spec = MethodSpec(
        name="recombee",
        tasks={Task.topn},
        feedback={"implicit", "explicit"},
        managed=True,
        can_score_candidates=False,
        cost_band="high",
        upstream="Recombee recommendation API",
    )
    env_keys = ("RECBENCH_RECOMBEE_DATABASE_ID", "RECBENCH_RECOMBEE_TOKEN")

    def recommend(self, user_id: str, k: int = 10) -> list[dict[str, Any]]:
        if not self.credentials():
            raise Unsupported("Recombee credentials are not set")
        database = os.environ["RECBENCH_RECOMBEE_DATABASE_ID"]
        token = os.environ["RECBENCH_RECOMBEE_TOKEN"]
        url = f"https://rapi.recombee.com/recombee/db/{database}/recomms/users/{user_id}/items/?count={k}"
        request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode())
        recs = body.get("recomms") or []
        return [
            {
                "item_id": str(item.get("id")),
                "score": 0.0,
                "explanation": Explanation("local", "Recombee recommendation", []).as_dict(),
            }
            for item in recs[:k]
        ]


@register_method
class VertexCommerce(ManagedMethod):
    spec = MethodSpec(
        name="vertex_commerce",
        tasks={Task.topn},
        feedback={"implicit"},
        managed=True,
        can_score_candidates=False,
        cost_band="high",
        upstream="Vertex AI Search for commerce. A live call is optional and is not a v1 gate.",
    )
    env_keys = ("RECBENCH_VERTEX_PROJECT", "RECBENCH_VERTEX_PLACEMENT")

    def fit(self, store: Any, config: dict[str, Any]) -> None:
        # Optional even when managed services are on. Missing credentials skip the live call.
        if not self.credentials():
            raise Unsupported("Vertex credentials are not set; skipping the live smoke")
        super().fit(store, config)

    def recommend(self, user_id: str, k: int = 10) -> list[dict[str, Any]]:
        if not self.credentials():
            raise Unsupported("Vertex credentials are not set")
        project = os.environ["RECBENCH_VERTEX_PROJECT"]
        placement = os.environ["RECBENCH_VERTEX_PLACEMENT"]
        url = (
            "https://retail.googleapis.com/v2/projects/"
            f"{project}/locations/global/catalogs/default_catalog/placements/{placement}:predict"
        )
        payload = json.dumps({"userEvent": {"visitorId": user_id}, "pageSize": k}).encode()
        request = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode())
        results = body.get("results") or []
        return [
            {
                "item_id": str(item.get("id")),
                "score": 0.0,
                "explanation": Explanation("local", "Vertex commerce prediction", []).as_dict(),
            }
            for item in results[:k]
        ]
