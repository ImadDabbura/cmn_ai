"""Tests for text data utilities."""

from cmn_ai.text.data import TextList


def test_text_list_from_files_reads_text_files(tmp_path):
    """Test TextList discovers and reads text files."""
    text_path = tmp_path / "sample.txt"
    ignored_path = tmp_path / "sample.md"
    text_path.write_text("Hello\nworld", encoding="utf8")
    ignored_path.write_text("ignored", encoding="utf8")

    texts = TextList.from_files(tmp_path, recurse=False)

    assert len(texts) == 1
    assert texts.path == tmp_path
    assert texts[0] == "Hello\nworld"


def test_text_list_applies_transform_after_loading(tmp_path):
    """Test TextList applies transforms to loaded text."""
    text_path = tmp_path / "sample.txt"
    text_path.write_text("  Hello  ", encoding="utf8")

    texts = TextList.from_files(
        tmp_path,
        recurse=False,
        tfms=lambda text: text.strip().lower(),
    )

    assert texts[0] == "hello"
