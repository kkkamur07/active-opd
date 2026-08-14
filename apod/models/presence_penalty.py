"""Presence penalty as an incremental logits processor.

vLLM's built-in penalty path rebuilds its token mask from scratch every decode
step: ``v1/sample/ops/penalties.py`` calls ``make_tensor_with_pad`` over the
entire generated history of every sequence, on CPU, from Python lists, then
copies it to the GPU. That is O(batch x tokens_so_far) per step -- vLLM says so
itself: "NOTE(nick): The penalties implementation is currently quite
inefficient". Measured here it caps generation at ~2,300 tok/s at 10k tokens and
~575 tok/s at 40k, and the cost per token depends only on trace length, not
batch size, so batching cannot outrun it.

Presence penalty does not need that work. Unlike frequency penalty it is pure
set membership: vLLM computes ``logits -= presence_penalty * output_mask``,
where the mask records only *whether* a token appeared, never how often
(``model_executor/layers/utils.py``). A growing set can be kept resident on the
GPU and updated with the one new token per sequence each step -- O(batch), with
identical logits out.

The equivalence is exact because of where this lands. Returning False from
``is_argmax_invariant`` puts it in ``non_argmax_invariant``, applied at
``v1/sample/sampler.py:399``, immediately before the built-in penalty at line
402 and before temperature scaling either way. Same raw logits, same
subtraction.

The penalty rides in ``SamplingParams.extra_args`` because leaving the native
``presence_penalty`` at 0.0 is what makes ``no_penalties`` True and skips the
expensive path. ``build_llm`` and ``build_sampling_params`` set both ends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.v1.sample.logits_processor import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

EXTRA_ARGS_KEY = "apod_presence_penalty"


class IncrementalPresencePenalty(LogitsProcessor):
    """One penalty value for the whole batch, which is what this project uses."""

    def __init__(
        self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool
    ) -> None:
        self.device = device
        self.max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.penalty = 0.0
        # index -> [live output token list, how many of them we have scattered]
        self.reqs: dict[int, list] = {}
        # Allocated on the first apply(), where the real padded vocab width is
        # visible; get_vocab_size() can disagree with it and guessing would
        # corrupt the mask.
        self.seen: torch.Tensor | None = None

    def is_argmax_invariant(self) -> bool:
        """Penalising a seen token can change which token is the argmax."""

        return False

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        if batch_update is None:
            return  # batch unchanged; apply() still picks up new tokens

        for index in batch_update.removed:
            self._clear(index)

        for index, params, _prompt, output_tok_ids in batch_update.added:
            self._clear(index)  # whatever was here must not leak into the new request
            penalty = float((params.extra_args or {}).get(EXTRA_ARGS_KEY) or 0.0)
            if not penalty:
                continue
            if self.penalty and penalty != self.penalty:
                raise ValueError(
                    f"mixed presence penalties in one batch ({self.penalty} and "
                    f"{penalty}); this processor assumes a single value"
                )
            self.penalty = penalty
            # A reference to the running output list, not a copy -- BatchUpdate
            # guarantees this, and it is how we see new tokens each step.
            self.reqs[index] = [output_tok_ids, 0]

        for a, b, direction in batch_update.moved:
            self._move(a, b, direction)

    def _clear(self, index: int) -> None:
        self.reqs.pop(index, None)
        if self.seen is not None:
            self.seen[index] = False

    def _move(self, a: int, b: int, direction: MoveDirectionality) -> None:
        a_req = self.reqs.pop(a, None)
        b_req = self.reqs.pop(b, None)
        if a_req is not None:
            self.reqs[b] = a_req
        if self.seen is None:
            if direction == MoveDirectionality.SWAP and b_req is not None:
                self.reqs[a] = b_req
            return
        if direction == MoveDirectionality.SWAP:
            if b_req is not None:
                self.reqs[a] = b_req
            a_row = self.seen[a].clone()  # or the second write reads an overwritten row
            self.seen[a] = self.seen[b]
            self.seen[b] = a_row
        else:
            self.seen[b] = self.seen[a]
            self.seen[a] = False

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.reqs:
            return logits

        num_reqs, vocab_width = logits.shape
        if self.seen is None:
            self.seen = torch.zeros(
                (self.max_num_seqs, vocab_width), dtype=torch.bool, device=self.device
            )

        # Scatter everything generated since the last step: one token per
        # sequence in steady state. A resumed request re-absorbs its whole
        # history, which is right because _clear wiped its row.
        rows: list[int] = []
        cols: list[int] = []
        for index, entry in self.reqs.items():
            if index >= num_reqs:
                continue
            tokens, absorbed = entry
            if len(tokens) > absorbed:
                fresh = tokens[absorbed:]
                rows.extend([index] * len(fresh))
                cols.extend(fresh)
                entry[1] = len(tokens)

        if rows:
            self.seen[
                torch.tensor(rows, dtype=torch.long, device=self.device),
                torch.tensor(cols, dtype=torch.long, device=self.device),
            ] = True

        # Untracked rows stay all-False, so they are unaffected.
        logits.sub_(self.seen[:num_reqs] * self.penalty)
        return logits
