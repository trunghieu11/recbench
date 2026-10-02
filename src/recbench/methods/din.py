"""DIN via RecBole's sequential DIN implementation."""

from __future__ import annotations

from recbench.methods.recbole_backend import RecBoleMethod
from recbench.protocol import MethodSpec, Task
from recbench.registry import register_method


@register_method
class DIN(RecBoleMethod):
    model_name = "DIN"
    spec = MethodSpec(
        name="din",
        tasks={Task.topn, Task.ctr, Task.rating, Task.sequential},
        feedback={"implicit", "explicit"},
        requires_side_features=True,
        cost_band="medium",
        upstream="RecBole 1.2 DIN.calculate_loss on the train split",
    )

    def score_candidates(self, store, candidates=None):
        frame = super().score_candidates(store, candidates)
        return frame
