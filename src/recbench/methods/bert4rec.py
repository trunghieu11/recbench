"""BERT4Rec via RecBole's model class and calculate_loss."""

from __future__ import annotations

from recbench.methods.recbole_backend import RecBoleMethod
from recbench.protocol import MethodSpec, Task
from recbench.registry import register_method


@register_method
class BERT4Rec(RecBoleMethod):
    model_name = "BERT4Rec"
    spec = MethodSpec(
        name="bert4rec",
        tasks={Task.topn, Task.sequential, Task.session},
        feedback={"implicit", "explicit"},
        cost_band="medium",
        upstream="RecBole 1.2 BERT4Rec.calculate_loss; histories stop at the cutoff",
    )
