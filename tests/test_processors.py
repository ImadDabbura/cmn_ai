"""
Unit tests for the NumericalizeProcessor class.
"""

from typing import List

import pytest

from cmn_ai.utils.processors import NumericalizeProcessor


class TestNumericalizeProcessor:
    """Test cases for NumericalizeProcessor."""

    def test_init_default_parameters(self) -> None:
        """Test initialization with default parameters."""
        processor = NumericalizeProcessor()

        assert processor.vocab is None
        assert processor.max_vocab == 60000
        assert processor.min_freq == 2
        assert processor.reserved_tokens is None
        assert processor.unk_token == "<unk>"
        assert not processor.is_fitted

    def test_init_custom_parameters(self) -> None:
        """Test initialization with custom parameters."""
        vocab = ["<unk>", "hello", "world"]
        reserved_tokens = ["<pad>", "<eos>"]

        processor = NumericalizeProcessor(
            vocab=vocab,
            max_vocab=1000,
            min_freq=5,
            reserved_tokens=reserved_tokens,
            unk_token="<unknown>",
        )

        assert processor.vocab == vocab
        assert processor.max_vocab == 1000
        assert processor.min_freq == 5
        assert processor.reserved_tokens == reserved_tokens
        assert processor.unk_token == "<unknown>"
        assert processor.is_fitted  # Should be fitted with predefined vocab

    def test_build_vocabulary_from_data(self) -> None:
        """Test building vocabulary from input data."""
        processor = NumericalizeProcessor(min_freq=1)
        tokens = [
            ["hello", "world"],
            ["hello", "there"],
            ["world", "is", "beautiful"],
        ]

        # Process tokens to build vocabulary
        processor(tokens)

        assert processor.is_fitted
        assert len(processor) > 0
        assert "<unk>" in processor.vocab
        assert "hello" in processor.vocab
        assert "world" in processor.vocab
        assert "there" in processor.vocab
        assert "is" in processor.vocab
        assert "beautiful" in processor.vocab

    def test_build_vocabulary_with_frequency_threshold(self) -> None:
        """Test vocabulary building respects frequency threshold."""
        processor = NumericalizeProcessor(min_freq=2)
        tokens = [
            ["hello", "world"],  # hello: 2, world: 2
            ["hello", "there"],  # there: 1
            ["world", "unique"],  # unique: 1
        ]

        processor(tokens)

        assert "hello" in processor.vocab  # freq=2 >= min_freq=2
        assert "world" in processor.vocab  # freq=2 >= min_freq=2
        assert "there" not in processor.vocab  # freq=1 < min_freq=2
        assert "unique" not in processor.vocab  # freq=1 < min_freq=2

    def test_build_vocabulary_with_reserved_tokens(self) -> None:
        """Test vocabulary building includes reserved tokens."""
        reserved_tokens = ["<pad>", "<eos>", "<sos>"]
        processor = NumericalizeProcessor(
            min_freq=1, reserved_tokens=reserved_tokens
        )
        tokens = [["hello", "world"]]

        processor(tokens)

        for token in reserved_tokens:
            assert token in processor.vocab
        assert "hello" in processor.vocab
        assert "world" in processor.vocab

    def test_process_single_token(self) -> None:
        """Test processing a single token."""
        processor = NumericalizeProcessor(min_freq=1)
        tokens = [["hello", "world"]]
        processor.fit(tokens)

        result = processor.process("hello")
        assert isinstance(result, int)
        assert result == processor.get_index("hello")

    def test_process_token_list(self) -> None:
        """Test processing a list of tokens."""
        processor = NumericalizeProcessor(min_freq=1)
        tokens = [["hello", "world"]]
        processor.fit(tokens)

        result = processor.process(["hello", "world"])
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == processor.get_index("hello")
        assert result[1] == processor.get_index("world")

    def test_process_unknown_token(self) -> None:
        """Test processing unknown tokens returns unk_idx."""
        processor = NumericalizeProcessor(min_freq=1)
        tokens = [["hello", "world"]]
        processor.fit(tokens)

        result = processor.process("unknown_token")
        assert result == processor.unk_idx

    def test_deprocess_single_index(self) -> None:
        """Test deprocessing a single index."""
        processor = NumericalizeProcessor(min_freq=1)
        tokens = [["hello", "world"]]
        processor.fit(tokens)

        hello_idx = processor.get_index("hello")
        result = processor.deprocess(hello_idx)
        assert result == "hello"

    def test_deprocess_index_list(self) -> None:
        """Test deprocessing a list of indices."""
        processor = NumericalizeProcessor(min_freq=1)
        tokens = [["hello", "world"]]
        processor.fit(tokens)

        indices = [processor.get_index("hello"), processor.get_index("world")]
        result = processor.deprocess(indices)
        assert result == ["hello", "world"]

    def test_deprocess_unknown_index(self) -> None:
        """Test deprocessing unknown indices returns unk_token."""
        processor = NumericalizeProcessor(min_freq=1)
        tokens = [["hello", "world"]]
        processor.fit(tokens)

        # Use an index that doesn't exist
        result = processor.deprocess(999)
        assert result == processor.unk_token

    def test_getitem_single_token(self) -> None:
        """Test bracket notation with single token."""
        processor = NumericalizeProcessor(min_freq=1)
        tokens = [["hello", "world"]]
        processor.fit(tokens)

        result = processor["hello"]
        assert result == processor.get_index("hello")

    def test_getitem_token_list(self) -> None:
        """Test bracket notation with token list."""
        processor = NumericalizeProcessor(min_freq=1)
        tokens = [["hello", "world"]]
        processor.fit(tokens)

        result = processor[["hello", "world"]]
        assert result == [
            processor.get_index("hello"),
            processor.get_index("world"),
        ]

    def test_fit_method(self) -> None:
        """Test the fit method."""
        processor = NumericalizeProcessor(min_freq=1)
        tokens = [["hello", "world"]]

        # Fit without processing
        result = processor.fit(tokens)

        assert result is processor  # Should return self for chaining
        assert processor.is_fitted
        assert "hello" in processor.vocab
        assert "world" in processor.vocab

    def test_get_token(self) -> None:
        """Test getting token by index."""
        processor = NumericalizeProcessor(min_freq=1)
        tokens = [["hello", "world"]]
        processor.fit(tokens)

        hello_idx = processor.get_index("hello")
        result = processor.get_token(hello_idx)
        assert result == "hello"

    def test_get_index(self) -> None:
        """Test getting index by token."""
        processor = NumericalizeProcessor(min_freq=1)
        tokens = [["hello", "world"]]
        processor.fit(tokens)

        result = processor.get_index("hello")
        assert isinstance(result, int)
        assert result >= 0

    def test_len_method(self) -> None:
        """Test the len method."""
        processor = NumericalizeProcessor(min_freq=1)
        tokens = [["hello", "world"]]
        processor.fit(tokens)

        assert len(processor) == len(processor.vocab)

    def test_len_unfitted(self) -> None:
        """Test len method when not fitted."""
        processor = NumericalizeProcessor()
        assert len(processor) == 0

    def test_properties(self) -> None:
        """Test processor properties."""
        processor = NumericalizeProcessor(min_freq=1)
        tokens = [["hello", "world"]]
        processor.fit(tokens)

        # Test token_to_idx property
        assert isinstance(processor.token_to_idx, dict)
        assert "hello" in processor.token_to_idx
        assert "world" in processor.token_to_idx

        # Test idx_to_token property
        assert isinstance(processor.idx_to_token, dict)
        assert processor.token_to_idx["hello"] in processor.idx_to_token

        # Test unk_idx property
        assert processor.unk_idx == processor.token_to_idx[processor.unk_token]

        # Test unk alias
        assert processor.unk == processor.unk_idx

    def test_error_unfitted_processor(self) -> None:
        """Test error handling for unfitted processor."""
        processor = NumericalizeProcessor()

        with pytest.raises(
            ValueError, match="Processor must be fitted before processing"
        ):
            processor.process("hello")

        with pytest.raises(
            ValueError, match="Processor must be fitted before deprocessing"
        ):
            processor.deprocess([1, 2, 3])

        with pytest.raises(
            ValueError, match="Processor must be fitted before getting tokens"
        ):
            processor.get_token(1)

        with pytest.raises(
            ValueError, match="Processor must be fitted before getting indices"
        ):
            processor.get_index("hello")

        with pytest.raises(
            ValueError,
            match="Processor must be fitted before accessing mappings",
        ):
            _ = processor.token_to_idx

        with pytest.raises(
            ValueError,
            match="Processor must be fitted before accessing mappings",
        ):
            _ = processor.idx_to_token

        with pytest.raises(
            ValueError,
            match="Processor must be fitted before accessing unk_idx",
        ):
            _ = processor.unk_idx

    def test_mixed_input_types(self) -> None:
        """Test processing mixed input types."""
        processor = NumericalizeProcessor(min_freq=1)
        tokens = [
            "single_token",  # Single string
            ["list", "of", "tokens"],  # List of strings
            ("tuple", "of", "tokens"),  # Tuple of strings
        ]

        processor.fit(tokens)

        # Test processing mixed types
        result1 = processor.process("single_token")
        result2 = processor.process(["list", "of", "tokens"])
        result3 = processor.process(("tuple", "of", "tokens"))

        assert isinstance(result1, int)
        assert isinstance(result2, list)
        assert isinstance(result3, list)
        assert all(isinstance(x, int) for x in result2)
        assert all(isinstance(x, int) for x in result3)

    def test_vocabulary_ordering(self) -> None:
        """Test that vocabulary is properly ordered."""
        processor = NumericalizeProcessor(min_freq=1)
        tokens = [["world", "hello", "there"]]
        processor.fit(tokens)

        # Vocabulary should be sorted
        vocab = processor.vocab
        assert vocab[0] == "<unk>"  # unk_token should be first
        assert vocab == sorted(vocab)  # Rest should be sorted

    def test_max_vocab_limit(self) -> None:
        """Test that max_vocab limit is respected."""
        processor = NumericalizeProcessor(max_vocab=3, min_freq=1)
        tokens = [["a", "b", "c", "d", "e"]]  # 5 unique tokens

        processor.fit(tokens)

        # Should respect max_vocab limit exactly
        assert len(processor) == processor.max_vocab
        assert len(processor.vocab) == 3

        # Should include unk_token and most frequent tokens
        assert "<unk>" in processor.vocab

    def test_max_vocab_with_reserved_tokens(self) -> None:
        """Test max_vocab limit with reserved tokens."""
        reserved_tokens = ["<pad>", "<eos>"]
        processor = NumericalizeProcessor(
            max_vocab=4, min_freq=1, reserved_tokens=reserved_tokens
        )
        tokens = [["a", "b", "c", "d", "e"]]  # 5 unique tokens

        processor.fit(tokens)

        # Should respect max_vocab limit exactly
        assert len(processor) == processor.max_vocab
        assert len(processor.vocab) == 4

        # Should include unk_token, reserved tokens, and some data tokens
        assert "<unk>" in processor.vocab
        assert "<pad>" in processor.vocab
        assert "<eos>" in processor.vocab

        # Should have only 1 slot left for data tokens (4 - 3 reserved = 1)
        data_tokens = [
            token
            for token in processor.vocab
            if token not in ["<unk>", "<pad>", "<eos>"]
        ]
        assert len(data_tokens) == 1

    def test_max_vocab_smaller_than_reserved(self) -> None:
        """Test max_vocab smaller than number of reserved tokens."""
        reserved_tokens = [
            "<pad>",
            "<eos>",
            "<sos>",
            "<cls>",
        ]  # 4 reserved tokens
        processor = NumericalizeProcessor(
            max_vocab=3,
            min_freq=1,
            reserved_tokens=reserved_tokens,  # max_vocab=3
        )
        tokens = [["a", "b", "c"]]  # 3 data tokens

        processor.fit(tokens)

        # Should still respect max_vocab limit exactly
        assert len(processor) == processor.max_vocab
        assert len(processor.vocab) == 3

        # Should prioritize unk_token (always included)
        assert "<unk>" in processor.vocab

        # Should include at most 2 additional tokens (either reserved or data tokens)
        non_unk_tokens = [
            token for token in processor.vocab if token != "<unk>"
        ]
        assert len(non_unk_tokens) == 2

    def test_max_vocab_edge_case_one_slot(self) -> None:
        """Test max_vocab=1 (only unk_token should fit)."""
        processor = NumericalizeProcessor(
            max_vocab=1, min_freq=1, reserved_tokens=["<pad>", "<eos>"]
        )
        tokens = [["a", "b", "c"]]

        processor.fit(tokens)

        # Should respect max_vocab=1 limit
        assert len(processor) == 1
        assert len(processor.vocab) == 1

        # Should only contain unk_token
        assert processor.vocab == ["<unk>"]

    def test_empty_input(self) -> None:
        """Test handling of empty input."""
        processor = NumericalizeProcessor(min_freq=1)
        tokens = []

        processor.fit(tokens)

        # Should only contain unk_token
        assert len(processor) == 1
        assert processor.vocab == ["<unk>"]

    def test_single_token_input(self) -> None:
        """Test handling of single token input."""
        processor = NumericalizeProcessor(min_freq=1)
        tokens = ["hello"]

        processor.fit(tokens)

        assert "hello" in processor.vocab
        assert len(processor) == 2  # unk_token + hello

    def test_duplicate_tokens(self) -> None:
        """Test handling of duplicate tokens."""
        processor = NumericalizeProcessor(min_freq=1)
        tokens = [["hello", "hello", "world", "world", "world"]]

        processor.fit(tokens)

        # Should only have unique tokens
        assert len(processor.vocab) == len(set(processor.vocab))
        assert "hello" in processor.vocab
        assert "world" in processor.vocab

    def test_backward_compatibility(self) -> None:
        """Test backward compatibility with old interface."""
        processor = NumericalizeProcessor(min_freq=1)
        tokens = [["hello", "world"]]
        processor.fit(tokens)

        # Test that unk property still works
        assert processor.unk == processor.unk_idx
        assert processor.unk == processor.token_to_idx["<unk>"]

    def test_call_method_chain(self) -> None:
        """Test that __call__ method can be chained."""
        processor = NumericalizeProcessor(min_freq=1)
        tokens = [["hello", "world"], ["hello", "there"]]

        # First call should fit and process
        result1 = processor(tokens)

        # Second call should just process (already fitted)
        result2 = processor(tokens)

        assert processor.is_fitted
        assert result1 == result2  # Should give same results

    def test_predefined_vocabulary_processing(self) -> None:
        """Test processing with predefined vocabulary."""
        vocab = ["<unk>", "<pad>", "hello", "world"]
        processor = NumericalizeProcessor(vocab=vocab)

        # Should be fitted immediately
        assert processor.is_fitted
        assert processor.vocab == vocab

        # Test processing
        result = processor.process(["hello", "world"])
        assert result == [2, 3]  # Indices for hello and world

        # Test unknown token
        result = processor.process(["unknown"])
        assert result == [0]  # unk_idx

    def test_predefined_vocabulary_immediate_use(self) -> None:
        """Test that processor with predefined vocabulary can be used immediately."""
        vocab = ["<unk>", "hello", "world"]
        processor = NumericalizeProcessor(vocab=vocab)

        # Should be able to use immediately without calling fit() or __call__()
        assert processor.is_fitted
        assert processor.get_index("hello") == 1
        assert processor.get_token(1) == "hello"
        assert processor["world"] == 2
        assert processor.deprocess([1, 2]) == ["hello", "world"]

        # Test that unk_idx works correctly
        assert processor.unk_idx == 0  # "<unk>" is at index 0

    def test_predefined_vocabulary_without_unk_token(self) -> None:
        """Test predefined vocabulary that doesn't include unk_token."""
        vocab = ["hello", "world"]  # Missing unk_token

        # This should work fine - the processor will use the provided vocab
        processor = NumericalizeProcessor(vocab=vocab)
        assert processor.is_fitted
        assert processor.vocab == vocab

        # When processing unknown tokens, it should return index 0 as fallback
        result = processor.process("unknown_token")
        assert result == 0  # Should return index 0 as fallback

        # The unk_idx should also return 0 as fallback
        assert processor.unk_idx == 0
