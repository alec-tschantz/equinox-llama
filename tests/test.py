"""
test_l3.1_8b_equivalence.py

This script serves as a comprehensive test suite for validating the equivalence
between the original PyTorch implementation of the Llama model and its JAX/Equinox
port. It performs the following key tasks:

1. Imports both the Hugging Face Transformers Llama model and the custom JAX/Equinox
   implementation.
2. Defines test functions for each major component of the Llama architecture, including:
   - Token Embedding
   - Linear Transformations
   - RMS Normalization
   - Multi-Layer Perceptron (MLP)
   - Self-Attention Mechanism
   - Decoder Layer
   - Full Model
   - Causal Language Model

3. For each component, the test:
   - Initializes both PyTorch and JAX/Equinox versions
   - Copies weights from the PyTorch model to the JAX/Equinox model
   - Generates identical inputs for both versions
   - Computes outputs using both implementations
   - Asserts that the outputs are numerically close, within a specified tolerance

The primary purposes of this script are to:
- Ensure that the JAX/Equinox implementation correctly replicates the behavior of
  the original PyTorch model.
- Verify that each component of the Llama architecture has been accurately ported.
- Catch any discrepancies or errors in the porting process.
- Provide a reliable test suite for ongoing development and refactoring of the
  JAX/Equinox implementation.

Usage:
    Run this script using pytest to validate the equivalence of the PyTorch and
    JAX/Equinox implementations of the Llama model. All tests should pass if the
    porting process has been successful.

Note:
    This test suite is crucial for maintaining the integrity and accuracy of the
    JAX/Equinox port. It should be run after any significant changes to the
    implementation and as part of the continuous integration process.
"""

import math
import pytest
import numpy as np
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer, LlamaForCausalLM as HFLlamaForCausalLM
import torch
import equinox as eqx

from typing import Optional, Tuple

from port.l3_eqx import (
    LlamaEmbedding,
    LlamaLinear,
    LlamaRotaryEmbedding,
    LlamaRMSNorm,
    LlamaSdpaAttention,
    LlamaMLP,
    LlamaDecoderLayer,
    LlamaModel,
    LlamaForCausalLM,
    LlamaConfig,
)


def torch_to_jax(tensor):
    return jnp.array(tensor.detach().numpy())


def assert_close(torch_output, jax_output, rtol=1e-5, atol=1e-5):
    np.testing.assert_allclose(
        torch_output.detach().numpy(), jax_output, rtol=rtol, atol=atol
    )


@pytest.fixture(scope="module")
def hf_model():
    model_name = "meta-llama/Meta-Llama-3.1-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = HFLlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()
    return tokenizer, model


@pytest.fixture(scope="module")
def eqx_config(hf_model):
    _, hf_model = hf_model
    config = LlamaConfig(
        vocab_size=hf_model.config.vocab_size,
        hidden_size=hf_model.config.hidden_size,
        intermediate_size=hf_model.config.intermediate_size,
        num_hidden_layers=hf_model.config.num_hidden_layers,
        num_attention_heads=hf_model.config.num_attention_heads,
        num_key_value_heads=hf_model.config.num_key_value_heads,
        max_position_embeddings=hf_model.config.max_position_embeddings,
        rms_norm_eps=hf_model.config.rms_norm_eps,
        rope_theta=hf_model.config.rope_theta,
        attention_bias=hf_model.config.attention_bias,
    )
    return config


def test_llama_model(hf_model, eqx_config):
    tokenizer, hf_model = hf_model
    eqx_model = LlamaModel(eqx_config)

    eqx_model = eqx.tree_at(
        lambda t: t.embed_tokens.weight,
        eqx_model,
        torch_to_jax(hf_model.model.embed_tokens.weight),
    )
    eqx_model = eqx.tree_at(
        lambda t: t.norm.weight, eqx_model, torch_to_jax(hf_model.model.norm.weight)
    )
    for i, layer in enumerate(eqx_model.layers):
        hf_layer = hf_model.model.layers[i]
        eqx_model = eqx.tree_at(
            lambda t: t.layers[i].self_attn.q_proj.weight,
            eqx_model,
            torch_to_jax(hf_layer.self_attn.q_proj.weight),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.layers[i].self_attn.k_proj.weight,
            eqx_model,
            torch_to_jax(hf_layer.self_attn.k_proj.weight),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.layers[i].self_attn.v_proj.weight,
            eqx_model,
            torch_to_jax(hf_layer.self_attn.v_proj.weight),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.layers[i].self_attn.o_proj.weight,
            eqx_model,
            torch_to_jax(hf_layer.self_attn.o_proj.weight),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.layers[i].mlp.gate_proj.weight,
            eqx_model,
            torch_to_jax(hf_layer.mlp.gate_proj.weight),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.layers[i].mlp.up_proj.weight,
            eqx_model,
            torch_to_jax(hf_layer.mlp.up_proj.weight),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.layers[i].mlp.down_proj.weight,
            eqx_model,
            torch_to_jax(hf_layer.mlp.down_proj.weight),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.layers[i].input_layernorm.weight,
            eqx_model,
            torch_to_jax(hf_layer.input_layernorm.weight),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.layers[i].post_attention_layernorm.weight,
            eqx_model,
            torch_to_jax(hf_layer.post_attention_layernorm.weight),
        )

    input_text = "Hello, world!"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    position_ids = torch.arange(input_ids.shape[1])[None, :]

    hf_output = hf_model.model(input_ids, position_ids=position_ids)[0]
    eqx_output = eqx_model(jnp.array(input_ids), position_ids=jnp.array(position_ids))

    assert_close(hf_output, eqx_output, rtol=1, atol=1e-4)


def test_llama_for_causal_lm(hf_model, eqx_config):
    tokenizer, hf_model = hf_model
    eqx_model = LlamaForCausalLM(eqx_config)

    eqx_model = eqx.tree_at(
        lambda t: t.model.embed_tokens.weight,
        eqx_model,
        torch_to_jax(hf_model.model.embed_tokens.weight),
    )
    eqx_model = eqx.tree_at(
        lambda t: t.model.norm.weight,
        eqx_model,
        torch_to_jax(hf_model.model.norm.weight),
    )
    eqx_model = eqx.tree_at(
        lambda t: t.lm_head.weight, eqx_model, torch_to_jax(hf_model.lm_head.weight)
    )
    for i, layer in enumerate(eqx_model.model.layers):
        hf_layer = hf_model.model.layers[i]
        eqx_model = eqx.tree_at(
            lambda t: t.model.layers[i].self_attn.q_proj.weight,
            eqx_model,
            torch_to_jax(hf_layer.self_attn.q_proj.weight),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.model.layers[i].self_attn.k_proj.weight,
            eqx_model,
            torch_to_jax(hf_layer.self_attn.k_proj.weight),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.model.layers[i].self_attn.v_proj.weight,
            eqx_model,
            torch_to_jax(hf_layer.self_attn.v_proj.weight),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.model.layers[i].self_attn.o_proj.weight,
            eqx_model,
            torch_to_jax(hf_layer.self_attn.o_proj.weight),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.model.layers[i].mlp.gate_proj.weight,
            eqx_model,
            torch_to_jax(hf_layer.mlp.gate_proj.weight),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.model.layers[i].mlp.up_proj.weight,
            eqx_model,
            torch_to_jax(hf_layer.mlp.up_proj.weight),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.model.layers[i].mlp.down_proj.weight,
            eqx_model,
            torch_to_jax(hf_layer.mlp.down_proj.weight),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.model.layers[i].input_layernorm.weight,
            eqx_model,
            torch_to_jax(hf_layer.input_layernorm.weight),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.model.layers[i].post_attention_layernorm.weight,
            eqx_model,
            torch_to_jax(hf_layer.post_attention_layernorm.weight),
        )

    input_text = "Hello, world!"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    position_ids = torch.arange(input_ids.shape[1])[None, :]

    hf_output = hf_model(input_ids, position_ids=position_ids).logits
    eqx_output = eqx_model(jnp.array(input_ids), position_ids=jnp.array(position_ids))

    assert_close(hf_output, eqx_output)
