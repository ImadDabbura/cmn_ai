"""Tests for vision data utilities."""

from PIL import Image

from cmn_ai.vision.data import ImageList


def test_image_list_from_files_loads_supported_images(tmp_path):
    """Test ImageList discovers image files and loads PIL images."""
    image_path = tmp_path / "sample.png"
    ignored_path = tmp_path / "sample.txt"
    Image.new("RGB", (3, 2), color=(255, 0, 0)).save(image_path)
    ignored_path.write_text("not an image")

    images = ImageList.from_files(tmp_path, extensions=".png", recurse=False)

    assert len(images) == 1
    assert images.path == tmp_path
    loaded = images[0]
    try:
        assert loaded.size == (3, 2)
        assert loaded.mode == "RGB"
    finally:
        loaded.close()


def test_image_list_applies_transform_after_loading(tmp_path):
    """Test ImageList applies transforms to loaded images."""
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (4, 5)).save(image_path)

    images = ImageList.from_files(
        tmp_path,
        extensions=".png",
        recurse=False,
        tfms=lambda image: image.size,
    )

    assert images[0] == (4, 5)
