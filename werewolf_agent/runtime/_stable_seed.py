# -*- coding: utf-8 -*-
"""Deterministic stable-seed helper.
    作者: Mike
    创建日期: 2025-01-15
    修改日期: 2026-07-05
    使用示例: 内部模块，无对外接口
P-U3: Single source of truth shared by both
``runtime.sheriff_policy`` and ``runtime.nodes._shared``.
Lives in a leaf module (no ``__init__.py`` siblings) so importers
do not trigger the
``sheriff_policy → nodes.__init__ → day → sheriff_policy`` circular
import that would occur if ``sheriff_policy`` reached for
``nodes._shared`` directly.  ``nodes._shared`` re-exports the
function for backward compatibility with the rest of the codebase.
"""
from __future__ import annotations

import hashlib


def _stable_seed(*parts: object) -> int:
    """Map a tuple of parts to a deterministic 32-bit seed.

    Used for any place the engine needs pseudo-random but
    reproducible output (speech order on no-sheriff days, candidate
    tie-breakers, etc.).  SHA-256 first 8 bytes, masked to
    unsigned 32-bit.
    """
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & 0xFFFFFFFF


__all__ = ["_stable_seed"]
