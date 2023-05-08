import mimetypes
from pathlib import Path
from typing import Iterable

import PIL

from ..utils.data import ItemList, get_files

IMAGE_EXTENSIONS = [
    k for k, v in mimetypes.types_map.items() if v.startswith("image/")
]
