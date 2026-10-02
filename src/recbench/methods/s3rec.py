"""S3-Rec pretrain via RecBole, including the attribute objectives."""

from __future__ import annotations

from recbench.methods.recbole_backend import RecBoleMethod
from recbench.protocol import MethodSpec, Task
from recbench.registry import register_method


@register_method
class S3Rec(RecBoleMethod):
    model_name = "S3Rec"
    needs_item_feature = True
    spec = MethodSpec(
        name="s3rec",
        tasks={Task.topn, Task.sequential, Task.session},
        feedback={"implicit", "explicit"},
        requires_side_features=True,
        cost_band="medium",
        upstream="RecBole 1.2 S3Rec pretrain (MIP/MAP/AAP/SP)",
    )
