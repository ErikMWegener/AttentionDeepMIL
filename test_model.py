"""Tests for model.py count_positive_instances method."""

import torch

from model import Attention, GatedAttention


def _make_dummy_bag(num_instances=10):
    """Create a dummy bag of 28x28 single-channel images."""
    return torch.randn(1, num_instances, 1, 28, 28)


class TestCountPositiveInstances:
    """Tests for Attention.count_positive_instances and GatedAttention.count_positive_instances."""

    def test_attention_returns_count_and_weights(self):
        model = Attention()
        model.eval()
        bag = _make_dummy_bag(8)
        with torch.no_grad():
            count, A = model.count_positive_instances(bag)
        assert isinstance(count, int)
        assert count >= 0
        assert count <= 8
        assert A.shape == (1, 8)

    def test_gated_attention_returns_count_and_weights(self):
        model = GatedAttention()
        model.eval()
        bag = _make_dummy_bag(8)
        with torch.no_grad():
            count, A = model.count_positive_instances(bag)
        assert isinstance(count, int)
        assert count >= 0
        assert count <= 8
        assert A.shape == (1, 8)

    def test_custom_threshold(self):
        model = Attention()
        model.eval()
        bag = _make_dummy_bag(10)
        with torch.no_grad():
            # With threshold=0, all instances should be counted
            count_all, _ = model.count_positive_instances(bag, threshold=0.0)
            # With threshold=1, no instances should be counted (softmax values < 1)
            count_none, _ = model.count_positive_instances(bag, threshold=1.0)
        assert count_all == 10
        assert count_none == 0

    def test_single_instance_bag(self):
        model = Attention()
        model.eval()
        bag = _make_dummy_bag(1)
        with torch.no_grad():
            count, A = model.count_positive_instances(bag)
        # Single instance gets all attention (softmax=1.0), threshold=1/1=1.0
        # Since 1.0 is not > 1.0, count should be 0 with default threshold
        assert count == 0
        assert A.shape == (1, 1)

    def test_attention_weights_sum_to_one(self):
        model = GatedAttention()
        model.eval()
        bag = _make_dummy_bag(5)
        with torch.no_grad():
            _, A = model.count_positive_instances(bag)
        assert abs(A.sum().item() - 1.0) < 1e-5


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
