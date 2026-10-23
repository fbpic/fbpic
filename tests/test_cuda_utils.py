import pytest

from fbpic.utils.cuda import cuda_tpb_bpg_1d, cuda_tpb_bpg_2d


def test_1d_grid_does_not_add_empty_block_for_exact_multiple():
    assert cuda_tpb_bpg_1d(512, TPB=256) == (2, 256)


def test_1d_grid_rounds_up_partial_block():
    assert cuda_tpb_bpg_1d(513, TPB=256) == (3, 256)


def test_2d_grid_uses_ceiling_division_per_dimension():
    assert cuda_tpb_bpg_2d(16, 33, 8, 16) == ((2, 3), (8, 16))


def test_zero_work_keeps_one_block_per_dimension_for_valid_launch():
    assert cuda_tpb_bpg_1d(0, TPB=256) == (1, 256)
    assert cuda_tpb_bpg_2d(0, 33, 8, 16) == ((1, 3), (8, 16))
    assert cuda_tpb_bpg_2d(16, 0, 8, 16) == ((2, 1), (8, 16))


@pytest.mark.parametrize("work", [-1, -100])
def test_1d_grid_rejects_negative_work(work):
    with pytest.raises(ValueError):
        cuda_tpb_bpg_1d(work, TPB=256)


@pytest.mark.parametrize("work", [(-1, 33), (16, -1)])
def test_2d_grid_rejects_negative_work(work):
    with pytest.raises(ValueError):
        cuda_tpb_bpg_2d(*work, 8, 16)


@pytest.mark.parametrize("block", [0, -1])
def test_1d_grid_rejects_nonpositive_block(block):
    with pytest.raises(ValueError):
        cuda_tpb_bpg_1d(1, TPB=block)


@pytest.mark.parametrize("blocks", [(0, 16), (8, 0), (-1, 16), (8, -1)])
def test_2d_grid_rejects_nonpositive_block(blocks):
    with pytest.raises(ValueError):
        cuda_tpb_bpg_2d(1, 1, *blocks)
