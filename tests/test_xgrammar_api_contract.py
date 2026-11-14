"""Contract checks for the required XGrammar integration surface."""

from __future__ import annotations

import inspect

import pytest


@pytest.mark.runtime
def test_xgrammar_compiler_matcher_and_numpy_bitmask_contract() -> None:
    """Keep the Phase-1 construction path honest across XGrammar upgrades."""
    import numpy
    import xgrammar as xgr

    assert "vocab_size" in inspect.signature(xgr.TokenizerInfo.from_huggingface).parameters
    compile_parameters = inspect.signature(xgr.GrammarCompiler.compile_json_schema).parameters
    assert {"strict_mode", "any_whitespace", "separators", "max_whitespace_cnt"} <= set(compile_parameters)
    assert "debug_print" in inspect.signature(xgr.GrammarMatcher.fill_next_token_bitmask).parameters

    # The model's logits dimension, rather than the tokenizer's number of
    # entries, controls the mask shape.  The 65-entry toy model leaves padded
    # entries just like a 248320-wide lm_head.
    tokenizer = xgr.TokenizerInfo(
        ["<eos>", "{", "}", '"', "ok", ":", "true", "false", ",", " "],
        xgr.VocabType.RAW,
        vocab_size=65,
        stop_token_ids=[0],
    )
    compiler = xgr.GrammarCompiler(tokenizer)
    compiled = compiler.compile_json_schema(
        {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
        },
        strict_mode=True,
    )
    matcher = xgr.GrammarMatcher(compiled)
    bitmask = numpy.zeros(xgr.get_bitmask_shape(1, 65), dtype=numpy.int32)

    assert matcher.fill_next_token_bitmask(bitmask) is True
    assert bitmask.shape == (1, 3)
    assert bitmask.dtype == numpy.int32
    assert matcher.accept_token(1) is not False
