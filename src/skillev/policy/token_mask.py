from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final, Protocol

import torch

TOKEN_MASK_FORMAT: Final = "xgrammar-bitmask@1"


def bitmask_width(vocab_size: int) -> int:
    return (vocab_size + 31) // 32


@dataclass(frozen=True, slots=True)
class TokenMask:
    constraint_hash: str
    vocab_size: int
    bitmask: torch.Tensor

    def __post_init__(self) -> None:
        if not self.constraint_hash:
            raise ValueError("token mask requires the recorded constraint hash")
        if type(self.vocab_size) is not int or self.vocab_size <= 0:
            raise ValueError("token mask vocabulary size must be a positive integer")
        mask = self.bitmask
        if (
            mask.dtype != torch.int32
            or mask.device.type != "cpu"
            or mask.ndim != 2
            or mask.shape[1] != bitmask_width(self.vocab_size)
            or not mask.is_contiguous()
        ):
            raise ValueError("token mask must be a contiguous int32 CPU [K, ceil(V/32)] bitmask")

    @property
    def rows(self) -> int:
        return int(self.bitmask.shape[0])

    @classmethod
    def from_allowed_ids(
        cls, *, constraint_hash: str, vocab_size: int, rows: Sequence[Iterable[int]]
    ) -> TokenMask:
        packed = torch.zeros((len(rows), bitmask_width(vocab_size)), dtype=torch.int64)
        for index, allowed in enumerate(rows):
            for token in allowed:
                if type(token) is not int or not 0 <= token < vocab_size:
                    raise ValueError("allowed token id outside the vocabulary")
                packed[index, token // 32] |= 1 << (token % 32)
        signed = torch.where(packed >= 2**31, packed - 2**32, packed).to(torch.int32)
        return cls(constraint_hash, vocab_size, signed.contiguous())

    def allowed(self, start: int, stop: int, device: torch.device | str) -> torch.Tensor:
        if not 0 <= start <= stop <= self.rows:
            raise ValueError("token mask row range out of bounds")
        rows = self.bitmask[start:stop].to(device)
        shifts = torch.arange(32, device=rows.device, dtype=torch.int32)
        bits = (rows.unsqueeze(-1) >> shifts) & 1
        return bits.reshape(stop - start, -1)[:, : self.vocab_size].bool()

    def require_targets(self, targets: Sequence[int]) -> None:
        if len(targets) != self.rows:
            raise ValueError("token mask rows must equal the number of scored action tokens")
        for index, token in enumerate(targets):
            if not 0 <= token < self.vocab_size:
                raise ValueError("scored token id outside the mask vocabulary")
            word = int(self.bitmask[index, token // 32])
            if not (word >> (token % 32)) & 1:
                raise ValueError(
                    f"scored action token {token} at position {index} is outside the grammar mask"
                )

    def forced_rows(self) -> tuple[bool, ...]:
        counts = self.allowed(0, self.rows, "cpu").sum(-1)
        return tuple(bool(value == 1) for value in counts.tolist())


class ActionMaskProvider(Protocol):
    def replay_action_mask(
        self, *, constraint_hash: str, action_token_ids: Sequence[int]
    ) -> TokenMask: ...
