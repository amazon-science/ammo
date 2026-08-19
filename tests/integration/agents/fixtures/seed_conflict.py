# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# vllm/model_executor/layers/fused_moe.py (mock)
import torch

def fused_moe_dispatch(hidden_states, w1, w2, topk_weights, topk_ids):
<<<<<<< op-012-gating
    # Op-012: Priority dispatch with grouped expert gating
    from vllm._custom_ops import grouped_expert_gate
    gate_out = grouped_expert_gate(hidden_states, topk_ids)
    return torch.ops.vllm.fused_moe(gate_out, w1, w2, topk_weights, topk_ids)
=======
    # Op-007: Fused expert + gate kernel
    from vllm._custom_ops import fused_expert_gate
    return fused_expert_gate(hidden_states, w1, w2, topk_weights, topk_ids)
>>>>>>> op-007-fused
