import pytest

torch = pytest.importorskip("torch")

from aopd.losses.opd import response_start_from_lengths, response_token_mask

PAD, EOS = 0, 9


def _selected(labels, mask):
    return [token for token, keep in zip(labels[0].tolist(), mask[0].tolist()) if keep]


def test_right_padded_shifted_batch_selects_every_response_token_once():
    ids = torch.tensor([[101, 102, 103, 201, 202, 203, EOS, PAD, PAD]])
    attention = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0, 0]])
    labels = ids[:, 1:]

    start = response_start_from_lengths(attention, torch.tensor([3]))
    mask = response_token_mask(
        attention[:, 1:], start - 1, input_ids=labels, eos_token_id=EOS, pad_token_id=PAD
    )

    assert _selected(labels, mask) == [201, 202, 203, EOS]


def test_left_padded_batch_is_masked_correctly():
    """HF requires left padding for batched decoder-only generation.

    The old mask offset from index 0 by a prompt *count*, which under left
    padding silently kept prompt tokens and dropped response tokens.
    """

    ids = torch.tensor([[PAD, PAD, 101, 102, 103, 201, 202, 203, EOS]])
    attention = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1, 1]])
    labels = ids[:, 1:]

    start = response_start_from_lengths(attention, torch.tensor([3]))
    mask = response_token_mask(
        attention[:, 1:], start - 1, input_ids=labels, eos_token_id=EOS, pad_token_id=PAD
    )

    assert _selected(labels, mask) == [201, 202, 203, EOS]


def test_ragged_right_padded_batch_masks_each_row_independently():
    ids = torch.tensor(
        [
            [101, 102, 201, 202, 203, EOS],
            [101, 102, 103, 301, EOS, PAD],
        ]
    )
    attention = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0]])
    labels = ids[:, 1:]

    start = response_start_from_lengths(attention, torch.tensor([2, 3]))
    mask = response_token_mask(
        attention[:, 1:], start - 1, input_ids=labels, eos_token_id=EOS, pad_token_id=PAD
    )

    rows = [
        [token for token, keep in zip(labels[row].tolist(), mask[row].tolist()) if keep]
        for row in range(2)
    ]
    assert rows == [[201, 202, 203, EOS], [301, EOS]]


def test_eos_survives_when_pad_and_eos_share_an_id():
    """Tokenizers without a pad token alias it to EOS.

    Masking on that id would delete the real EOS -- the token the student most
    needs to learn to emit.
    """

    ids = torch.tensor([[101, 102, 201, 202, EOS]])
    attention = torch.tensor([[1, 1, 1, 1, 1]])
    labels = ids[:, 1:]

    start = response_start_from_lengths(attention, torch.tensor([2]))
    mask = response_token_mask(
        attention[:, 1:], start - 1, input_ids=labels, eos_token_id=EOS, pad_token_id=EOS
    )

    assert _selected(labels, mask) == [201, 202, EOS]
