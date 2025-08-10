"""
Unit tests for losses module.

This module contains comprehensive tests for the losses module,
covering NoneReduce context manager, reduce_loss function, and
LabelSmoothingCrossEntropy class.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from cmn_ai.losses import LabelSmoothingCrossEntropy, NoneReduce, reduce_loss


class TestNoneReduce:
    """Test cases for NoneReduce context manager."""

    def setup_method(self):
        """Set up test data for each test method."""
        self.batch_size = 4
        self.num_classes = 3
        self.outputs = torch.randn(self.batch_size, self.num_classes)
        self.targets = torch.randint(0, self.num_classes, (self.batch_size,))

    def test_context_manager_with_pytorch_loss(self):
        """Test NoneReduce with PyTorch loss functions."""
        loss_func = nn.CrossEntropyLoss()

        # Test normal reduction
        normal_loss = loss_func(self.outputs, self.targets)
        assert normal_loss.dim() == 0  # Scalar tensor

        # Test with NoneReduce context
        with NoneReduce(loss_func) as no_reduce_loss_func:
            unreduced_loss = no_reduce_loss_func(self.outputs, self.targets)
            assert unreduced_loss.shape == (self.batch_size,)
            assert unreduced_loss.dim() == 1

        # Test that original reduction is restored
        restored_loss = loss_func(self.outputs, self.targets)
        assert restored_loss.dim() == 0

    def test_context_manager_with_custom_loss(self):
        """Test NoneReduce with custom loss functions."""

        def custom_loss(outputs, targets, reduction="mean"):
            loss = F.cross_entropy(outputs, targets, reduction="none")
            if reduction == "mean":
                return loss.mean()
            elif reduction == "sum":
                return loss.sum()
            else:
                return loss

        # Test normal behavior
        normal_loss = custom_loss(self.outputs, self.targets)
        assert normal_loss.dim() == 0

        # Test with NoneReduce context
        with NoneReduce(custom_loss) as no_reduce_loss_func:
            unreduced_loss = no_reduce_loss_func(self.outputs, self.targets)
            assert unreduced_loss.shape == (self.batch_size,)

        # Test that function still works normally after context
        restored_loss = custom_loss(self.outputs, self.targets)
        assert restored_loss.dim() == 0

    def test_context_manager_preserves_original_reduction(self):
        """Test that NoneReduce preserves the original reduction setting."""
        loss_func = nn.CrossEntropyLoss(reduction="sum")
        original_reduction = loss_func.reduction

        with NoneReduce(loss_func) as no_reduce_loss_func:
            unreduced_loss = no_reduce_loss_func(self.outputs, self.targets)
            assert unreduced_loss.shape == (self.batch_size,)

        assert loss_func.reduction == original_reduction

    def test_context_manager_exception_handling(self):
        """Test that NoneReduce properly restores reduction even with exceptions."""
        loss_func = nn.CrossEntropyLoss(reduction="mean")
        original_reduction = loss_func.reduction

        with pytest.raises(ValueError):
            with NoneReduce(loss_func):
                raise ValueError("Test exception")

        assert loss_func.reduction == original_reduction

    def test_context_manager_multiple_entries(self):
        """Test NoneReduce with multiple context entries."""
        loss_func = nn.CrossEntropyLoss()

        with NoneReduce(loss_func) as no_reduce_1:
            with NoneReduce(loss_func) as no_reduce_2:
                loss_1 = no_reduce_1(self.outputs, self.targets)
                loss_2 = no_reduce_2(self.outputs, self.targets)
                assert loss_1.shape == (self.batch_size,)
                assert loss_2.shape == (self.batch_size,)

        # Should be restored to original state
        normal_loss = loss_func(self.outputs, self.targets)
        assert normal_loss.dim() == 0


class TestReduceLoss:
    """Test cases for reduce_loss function."""

    def setup_method(self):
        """Set up test data for each test method."""
        self.test_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        self.batch_tensor = torch.randn(10, 5)

    def test_reduce_loss_mean(self):
        """Test reduce_loss with mean reduction."""
        result = reduce_loss(self.test_tensor, reduction="mean")
        expected = self.test_tensor.mean()
        torch.testing.assert_close(result, expected)

    def test_reduce_loss_sum(self):
        """Test reduce_loss with sum reduction."""
        result = reduce_loss(self.test_tensor, reduction="sum")
        expected = self.test_tensor.sum()
        torch.testing.assert_close(result, expected)

    def test_reduce_loss_none(self):
        """Test reduce_loss with no reduction."""
        result = reduce_loss(self.test_tensor, reduction=None)
        torch.testing.assert_close(result, self.test_tensor)

    def test_reduce_loss_default_none(self):
        """Test reduce_loss with default None reduction."""
        result = reduce_loss(self.test_tensor)
        torch.testing.assert_close(result, self.test_tensor)

    def test_reduce_loss_batch_tensor(self):
        """Test reduce_loss with batch tensor."""
        result_mean = reduce_loss(self.batch_tensor, reduction="mean")
        result_sum = reduce_loss(self.batch_tensor, reduction="sum")

        assert result_mean.dim() == 0
        assert result_sum.dim() == 0
        torch.testing.assert_close(result_mean, self.batch_tensor.mean())
        torch.testing.assert_close(result_sum, self.batch_tensor.sum())

    def test_reduce_loss_single_element(self):
        """Test reduce_loss with single element tensor."""
        single_tensor = torch.tensor([42.0])

        result_mean = reduce_loss(single_tensor, reduction="mean")
        result_sum = reduce_loss(single_tensor, reduction="sum")
        result_none = reduce_loss(single_tensor, reduction=None)

        torch.testing.assert_close(result_mean, torch.tensor(42.0))
        torch.testing.assert_close(result_sum, torch.tensor(42.0))
        torch.testing.assert_close(result_none, single_tensor)

    def test_reduce_loss_empty_tensor(self):
        """Test reduce_loss with empty tensor."""
        empty_tensor = torch.tensor([])

        # Empty tensor operations may behave differently in different PyTorch versions
        # Test that no reduction works
        result = reduce_loss(empty_tensor, reduction=None)
        torch.testing.assert_close(result, empty_tensor)

        # Test that sum works (returns 0 for empty tensors)
        result_sum = reduce_loss(empty_tensor, reduction="sum")
        assert torch.isfinite(result_sum)

        # Mean of empty tensor is NaN, which is expected behavior
        result_mean = reduce_loss(empty_tensor, reduction="mean")
        assert torch.isnan(result_mean)

    def test_reduce_loss_different_dtypes(self):
        """Test reduce_loss with different data types."""
        float_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        double_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        result_float = reduce_loss(float_tensor, reduction="mean")
        result_double = reduce_loss(double_tensor, reduction="mean")

        assert result_float.dtype == torch.float32
        assert result_double.dtype == torch.float64


class TestLabelSmoothingCrossEntropy:
    """Test cases for LabelSmoothingCrossEntropy class."""

    def setup_method(self):
        """Set up test data for each test method."""
        self.batch_size = 4
        self.num_classes = 3
        self.outputs = torch.randn(self.batch_size, self.num_classes)
        self.targets = torch.randint(0, self.num_classes, (self.batch_size,))

        # Create deterministic outputs for testing
        self.det_outputs = torch.tensor(
            [
                [1.0, 0.0, 0.0],  # Strong prediction for class 0
                [0.0, 1.0, 0.0],  # Strong prediction for class 1
                [0.0, 0.0, 1.0],  # Strong prediction for class 2
                [0.5, 0.3, 0.2],  # Uncertain prediction
            ]
        )
        self.det_targets = torch.tensor([0, 1, 2, 0])

    def test_init_default_parameters(self):
        """Test LabelSmoothingCrossEntropy initialization with default parameters."""
        loss_func = LabelSmoothingCrossEntropy()

        assert loss_func.eps == 0.1
        assert loss_func.reduction == "mean"

    def test_init_custom_parameters(self):
        """Test LabelSmoothingCrossEntropy initialization with custom parameters."""
        loss_func = LabelSmoothingCrossEntropy(eps=0.05, reduction="sum")

        assert loss_func.eps == 0.05
        assert loss_func.reduction == "sum"

    def test_forward_basic_functionality(self):
        """Test basic forward pass functionality."""
        loss_func = LabelSmoothingCrossEntropy(eps=0.1)
        loss = loss_func(self.outputs, self.targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar tensor
        # Note: requires_grad depends on input tensors, not the loss function itself

    def test_forward_no_smoothing(self):
        """Test forward pass with no smoothing (eps=0)."""
        loss_func = LabelSmoothingCrossEntropy(eps=0.0)
        loss = loss_func(self.det_outputs, self.det_targets)

        # Should be equivalent to standard cross-entropy
        expected_loss = F.cross_entropy(self.det_outputs, self.det_targets)
        torch.testing.assert_close(loss, expected_loss, rtol=1e-5, atol=1e-6)

    def test_forward_maximum_smoothing(self):
        """Test forward pass with maximum smoothing (eps=1)."""
        loss_func = LabelSmoothingCrossEntropy(eps=1.0)
        loss = loss_func(self.det_outputs, self.det_targets)

        # With eps=1, should be equivalent to KL divergence with uniform distribution
        log_probs = F.log_softmax(self.det_outputs, dim=-1)
        uniform_loss = -log_probs.sum(dim=-1) / self.num_classes
        expected_loss = uniform_loss.mean()

        torch.testing.assert_close(loss, expected_loss, rtol=1e-5, atol=1e-6)

    def test_forward_different_reductions(self):
        """Test forward pass with different reduction methods."""
        # Test mean reduction
        loss_func_mean = LabelSmoothingCrossEntropy(eps=0.1, reduction="mean")
        loss_mean = loss_func_mean(self.outputs, self.targets)
        assert loss_mean.dim() == 0

        # Test sum reduction
        loss_func_sum = LabelSmoothingCrossEntropy(eps=0.1, reduction="sum")
        loss_sum = loss_func_sum(self.outputs, self.targets)
        assert loss_sum.dim() == 0

        # Test none reduction
        loss_func_none = LabelSmoothingCrossEntropy(eps=0.1, reduction="none")
        loss_none = loss_func_none(self.outputs, self.targets)
        assert loss_none.shape == (self.batch_size,)

    def test_forward_manual_calculation(self):
        """Test forward pass with manual calculation verification."""
        eps = 0.1
        loss_func = LabelSmoothingCrossEntropy(eps=eps)

        # Manual calculation
        log_probs = F.log_softmax(self.det_outputs, dim=-1)
        nll_loss = F.nll_loss(log_probs, self.det_targets, reduction="mean")
        uniform_loss = -log_probs.sum(dim=-1) / self.num_classes
        uniform_loss = uniform_loss.mean()

        expected_loss = (1 - eps) * nll_loss + eps * uniform_loss

        # Forward pass
        actual_loss = loss_func(self.det_outputs, self.det_targets)

        torch.testing.assert_close(
            actual_loss, expected_loss, rtol=1e-5, atol=1e-6
        )

    def test_forward_single_sample(self):
        """Test forward pass with single sample."""
        single_output = torch.randn(1, self.num_classes)
        single_target = torch.randint(0, self.num_classes, (1,))

        loss_func = LabelSmoothingCrossEntropy(eps=0.1)
        loss = loss_func(single_output, single_target)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

    def test_forward_large_batch(self):
        """Test forward pass with large batch size."""
        large_batch_size = 100
        large_outputs = torch.randn(large_batch_size, self.num_classes)
        large_targets = torch.randint(0, self.num_classes, (large_batch_size,))

        loss_func = LabelSmoothingCrossEntropy(eps=0.1)
        loss = loss_func(large_outputs, large_targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

    def test_forward_different_dtypes(self):
        """Test forward pass with different data types."""
        loss_func = LabelSmoothingCrossEntropy(eps=0.1)

        # Float32
        outputs_f32 = self.outputs.float()
        targets_f32 = self.targets.long()
        loss_f32 = loss_func(outputs_f32, targets_f32)
        assert loss_f32.dtype == torch.float32

        # Float64
        outputs_f64 = self.outputs.double()
        targets_f64 = self.targets.long()
        loss_f64 = loss_func(outputs_f64, targets_f64)
        assert loss_f64.dtype == torch.float64

    def test_forward_device_compatibility(self):
        """Test forward pass with different devices."""
        if torch.cuda.is_available():
            loss_func = LabelSmoothingCrossEntropy(eps=0.1)

            # Move to GPU
            outputs_gpu = self.outputs.cuda()
            targets_gpu = self.targets.cuda()
            loss_gpu = loss_func(outputs_gpu, targets_gpu)

            assert loss_gpu.device.type == "cuda"

            # Move back to CPU
            outputs_cpu = self.outputs.cpu()
            targets_cpu = self.targets.cpu()
            loss_cpu = loss_func(outputs_cpu, targets_cpu)

            assert loss_cpu.device.type == "cpu"

    def test_forward_gradient_flow(self):
        """Test that gradients flow properly through the loss function."""
        loss_func = LabelSmoothingCrossEntropy(eps=0.1)

        outputs = self.outputs.clone().requires_grad_(True)
        loss = loss_func(outputs, self.targets)

        loss.backward()

        assert outputs.grad is not None
        assert outputs.grad.shape == outputs.shape

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test valid eps values
        for eps in [0.0, 0.1, 0.5, 1.0]:
            loss_func = LabelSmoothingCrossEntropy(eps=eps)
            assert loss_func.eps == eps

        # Test valid reduction values
        for reduction in ["mean", "sum", "none"]:
            loss_func = LabelSmoothingCrossEntropy(reduction=reduction)
            assert loss_func.reduction == reduction

    def test_forward_with_autograd(self):
        """Test forward pass in autograd context."""
        loss_func = LabelSmoothingCrossEntropy(eps=0.1)

        with torch.enable_grad():
            outputs = self.outputs.clone().requires_grad_(True)
            loss = loss_func(outputs, self.targets)

            assert loss.requires_grad
            loss.backward()
            assert outputs.grad is not None

    def test_forward_edge_cases(self):
        """Test forward pass with edge cases."""
        loss_func = LabelSmoothingCrossEntropy(eps=0.1)

        # Test with very small outputs
        small_outputs = torch.tensor([[0.001, 0.001, 0.001]])
        small_targets = torch.tensor([0])
        loss_small = loss_func(small_outputs, small_targets)
        assert torch.isfinite(loss_small)

        # Test with very large outputs
        large_outputs = torch.tensor([[1000.0, 1000.0, 1000.0]])
        large_targets = torch.tensor([0])
        loss_large = loss_func(large_outputs, large_targets)
        assert torch.isfinite(loss_large)

    def test_forward_consistency(self):
        """Test that forward pass is consistent for same inputs."""
        loss_func = LabelSmoothingCrossEntropy(eps=0.1)

        loss1 = loss_func(self.outputs, self.targets)
        loss2 = loss_func(self.outputs, self.targets)

        torch.testing.assert_close(loss1, loss2)

    def test_module_inheritance(self):
        """Test that LabelSmoothingCrossEntropy properly inherits from nn.Module."""
        loss_func = LabelSmoothingCrossEntropy()

        assert isinstance(loss_func, nn.Module)
        assert hasattr(loss_func, "forward")
        assert hasattr(loss_func, "training")

    def test_serialization(self):
        """Test that LabelSmoothingCrossEntropy can be serialized."""
        loss_func = LabelSmoothingCrossEntropy(eps=0.15, reduction="sum")

        # Save and load the entire module
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False) as f:
            torch.save(loss_func, f.name)
            loaded_loss_func = torch.load(f.name, weights_only=False)

        os.unlink(f.name)

        # Check that parameters are preserved
        assert loaded_loss_func.eps == 0.15
        assert loaded_loss_func.reduction == "sum"

        # Test that the loaded function works the same
        original_loss = loss_func(self.outputs, self.targets)
        loaded_loss = loaded_loss_func(self.outputs, self.targets)
        torch.testing.assert_close(original_loss, loaded_loss)
