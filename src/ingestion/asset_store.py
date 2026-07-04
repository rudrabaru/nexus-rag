import shutil
from pathlib import Path


class LocalAssetStore:
    """
    Storage abstraction layer for visual assets (images, charts, etc).
    Currently implemented as local filesystem storage, but can easily be swapped
    for GCS/S3 later.
    """

    def __init__(self, base_dir: str = "assets"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, doc_id: str, data: bytes, filename: str) -> str:
        """
        Saves data and returns the asset reference path.
        """
        doc_dir = self.base_dir / doc_id
        doc_dir.mkdir(parents=True, exist_ok=True)

        file_path = doc_dir / filename
        with open(file_path, "wb") as f:
            f.write(data)

        return str(file_path)

    def get_path(self, asset_ref: str) -> Path:
        """Returns the absolute Path for an asset_ref."""
        return Path(asset_ref).absolute()

    def delete_doc_assets(self, doc_id: str):
        """Deletes all assets associated with a document."""
        doc_dir = self.base_dir / doc_id
        if doc_dir.exists():
            shutil.rmtree(doc_dir)
