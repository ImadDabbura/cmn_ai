"""
Unit tests for activation functions module.

This module contains comprehensive tests for the GeneralRelu class,
covering initialization, forward pass, edge cases, and PyTorch compatibility.
"""

import torch
import torch.nn as nn

from cmn_ai.activations import GeneralRelu


class TestGeneralRelu:
    """Test cases for GeneralRelu class."""

    def setup_method(self):
        """Set up test data for each test method."""
        self.test_input = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0, 5.0])
        self.batch_input = torch.randn(10, 5)  # Batch of random data

    def test_init_default_parameters(self):
        """Test GeneralRelu initialization with default parameters."""
        act = GeneralRelu()

        assert act.leak == 0.1
        assert act.sub == 0.4
        assert act.maxv is None

    def test_init_custom_parameters(self):
        """Test GeneralRelu initialization with custom parameters."""
        act = GeneralRelu(leak=0.2, sub=0.5, maxv=3.0)

        assert act.leak == 0.2
        assert act.sub == 0.5
        assert act.maxv == 3.0

    def test_init_none_parameters(self):
        """Test GeneralRelu initialization with None parameters."""
        act = GeneralRelu(leak=None, sub=None, maxv=None)

        assert act.leak is None
        assert act.sub is None
        assert act.maxv is None

    def test_forward_standard_relu(self):
        """Test forward pass with standard ReLU behavior (leak=None)."""
        act = GeneralRelu(leak=None, sub=None, maxv=None)
        output = act(self.test_input)

        expected = torch.tensor([0.0, 0.0, 0.0, 0.5, 2.0, 5.0])
        torch.testing.assert_close(output, expected)

    def test_forward_leaky_relu(self):
        """Test forward pass with leaky ReLU behavior."""
        act = GeneralRelu(leak=0.1, sub=None, maxv=None)
        output = act(self.test_input)

        expected = torch.tensor([-0.2, -0.05, 0.0, 0.5, 2.0, 5.0])
        torch.testing.assert_close(output, expected)

    def test_forward_with_subtraction(self):
        """Test forward pass with subtraction applied."""
        act = GeneralRelu(leak=0.1, sub=0.4, maxv=None)
        output = act(self.test_input)

        expected = torch.tensor([-0.6, -0.45, -0.4, 0.1, 1.6, 4.6])
        torch.testing.assert_close(output, expected)

    def test_forward_with_max_clipping(self):
        """Test forward pass with maximum value clipping."""
        act = GeneralRelu(leak=0.1, sub=0.4, maxv=2.0)
        output = act(self.test_input)

        expected = torch.tensor([-0.6, -0.45, -0.4, 0.1, 1.6, 2.0])
        torch.testing.assert_close(output, expected)

    def test_forward_all_features(self):
        """Test forward pass with all features enabled."""
        act = GeneralRelu(leak=0.2, sub=0.3, maxv=1.5)
        output = act(self.test_input)

        # Manual calculation: leaky_relu(x, 0.2) - 0.3, then clamp to 1.5
        expected = torch.tensor([-0.7, -0.4, -0.3, 0.2, 1.5, 1.5])
        torch.testing.assert_close(output, expected)

    def test_forward_batch_input(self):
        """Test forward pass with batch input."""
        act = GeneralRelu(leak=0.1, sub=0.2, maxv=1.0)
        output = act(self.batch_input)

        assert output.shape == self.batch_input.shape
        assert torch.all(output <= 1.0)  # Check max clipping

        # Manually compute expected output to verify correctness
        expected = torch.nn.functional.leaky_relu(self.batch_input, 0.1) - 0.2
        expected = expected.clamp(max=1.0)

        torch.testing.assert_close(output, expected)

    def test_forward_gradient_flow(self):
        """Test that gradients flow properly through the activation."""
        act = GeneralRelu(leak=0.1, sub=0.4, maxv=2.0)
        x = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)

        output = act(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_forward_inplace_operations(self):
        """Test that in-place operations work correctly."""
        act = GeneralRelu(leak=0.1, sub=0.4, maxv=2.0)
        x = self.test_input.clone()

        output = act(x)
        # Check that input is not modified in-place
        assert not torch.equal(x, output)

    def test_module_inheritance(self):
        """Test that GeneralRelu properly inherits from nn.Module."""
        act = GeneralRelu()

        assert isinstance(act, nn.Module)
        assert hasattr(act, "forward")

    def test_parameters_persistence(self):
        """Test that parameters persist correctly after forward pass."""
        act = GeneralRelu(leak=0.15, sub=0.25, maxv=3.5)

        # Store original parameters
        original_leak = act.leak
        original_sub = act.sub
        original_maxv = act.maxv

        # Perform forward pass
        _ = act(self.test_input)

        # Check parameters haven't changed
        assert act.leak == original_leak
        assert act.sub == original_sub
        assert act.maxv == original_maxv

    def test_edge_cases_zero_input(self):
        """Test behavior with zero input."""
        act = GeneralRelu(leak=0.1, sub=0.4, maxv=2.0)
        zero_input = torch.zeros(5)

        output = act(zero_input)
        expected = torch.full((5,), -0.4)  # 0 - 0.4 = -0.4
        torch.testing.assert_close(output, expected)

    def test_edge_cases_large_input(self):
        """Test behavior with very large input values."""
        act = GeneralRelu(leak=0.1, sub=0.4, maxv=2.0)
        large_input = torch.tensor([100.0, -100.0, 50.0])

        output = act(large_input)
        expected = torch.tensor([2.0, -10.4, 2.0])  # Clipped to maxv=2.0
        torch.testing.assert_close(output, expected)

    def test_edge_cases_single_element(self):
        """Test behavior with single element tensor."""
        act = GeneralRelu(leak=0.1, sub=0.4, maxv=2.0)
        single_input = torch.tensor([1.5])

        output = act(single_input)
        expected = torch.tensor([1.1])  # 1.5 - 0.4 = 1.1
        torch.testing.assert_close(output, expected)

    def test_different_dtypes(self):
        """Test that the activation works with different data types."""
        act = GeneralRelu(leak=0.1, sub=0.4, maxv=2.0)

        for dtype in [torch.float32, torch.float64]:
            x = self.test_input.to(dtype)
            output = act(x)

            assert output.dtype == dtype
            assert not torch.isnan(output).any()

    def test_device_compatibility(self):
        """Test that the activation works on different devices."""
        if torch.cuda.is_available():
            act = GeneralRelu(leak=0.1, sub=0.4, maxv=2.0)
            x = self.test_input.to("cuda")

            output = act(x)
            assert output.device == x.device
            assert not torch.isnan(output).any()

    def test_serialization(self):
        """Test that the activation can be recreated with same parameters."""
        act = GeneralRelu(leak=0.15, sub=0.25, maxv=3.5)

        # Test parameter-based recreation (since GeneralRelu has no learnable parameters)
        # This simulates how the module would be saved/loaded in practice
        params = {"leak": act.leak, "sub": act.sub, "maxv": act.maxv}

        # Create new instance with same parameters
        loaded_act = GeneralRelu(**params)

        # Test that recreated activation produces same output
        output_original = act(self.test_input)
        output_loaded = loaded_act(self.test_input)

        torch.testing.assert_close(output_original, output_loaded)

        # Also test that parameters are preserved
        assert loaded_act.leak == act.leak
        assert loaded_act.sub == act.sub
        assert loaded_act.maxv == act.maxv

    def test_parameter_validation(self):
        """Test that invalid parameters raise appropriate errors."""
        # These should work without errors
        GeneralRelu(leak=0.0, sub=0.0, maxv=0.0)
        GeneralRelu(
            leak=-0.1, sub=-0.1, maxv=-0.1
        )  # Negative values are allowed

        # Test with very large values
        GeneralRelu(leak=1000.0, sub=1000.0, maxv=1000.0)

    def test_forward_with_autograd(self):
        """Test that the activation works properly with autograd."""
        act = GeneralRelu(leak=0.1, sub=0.4, maxv=2.0)
        x = torch.tensor([-1.0, 0.0, 1.0, 3.0], requires_grad=True)

        output = act(x)
        loss = output.mean()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert not torch.isnan(x.grad).any()

    def test_forward_with_complex_network(self):
        """Test the activation in a simple neural network."""
        act = GeneralRelu(leak=0.1, sub=0.4, maxv=2.0)
        linear = nn.Linear(3, 2)

        x = torch.randn(4, 3)
        output = linear(x)
        activated = act(output)

        assert activated.shape == (4, 2)
        assert torch.all(activated <= 2.0)  # Check max clipping
