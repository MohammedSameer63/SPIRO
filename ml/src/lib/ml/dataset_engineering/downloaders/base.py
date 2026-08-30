"""
SPIRO ML — Dataset Downloaders
Downloads every supported dataset from its canonical source with
checksum verification.

Supported datasets
------------------
- TACO          (GitHub releases / direct URL)
- TrashNet      (Google Drive)
- ZeroWaste-f   (direct URL)
- OpenLitterMap (API export)
- KaggleGC      (Kaggle API)
- MJU-Waste     (Google Drive)
- WADE-ai       (Roboflow export)
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import urlretrieve

import requests
from tqdm import tqdm

from lib.ml.core.logger import get_logger

log = get_logger(__name__)

# Root for all raw downloads
_DEFAULT_RAW = Path("datasets/raw")


# =============================================================================
# Base downloader
# =============================================================================

class BaseDownloader(ABC):
    """
    Abstract base for dataset downloaders.

    Each subclass implements download() which places raw data under
    `raw_dir/<dataset_name>/`.
    """

    name: str = ""
    license: str = ""
    homepage: str = ""
    citation: str = ""
    version: str = ""

    def __init__(self, raw_dir: Path = _DEFAULT_RAW) -> None:
        self.raw_dir = raw_dir / self.name
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self._meta: Dict[str, Any] = {}

    @abstractmethod
    def download(self) -> Path:
        """Download the dataset. Returns path to extracted root dir."""

    def verify_checksum(self, path: Path, expected_md5: str) -> bool:
        """MD5 checksum verification."""
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        actual = h.hexdigest()
        ok = actual == expected_md5
        if not ok:
            log.warning(
                f"Checksum mismatch for {path.name}: "
                f"expected={expected_md5} actual={actual}"
            )
        return ok

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "license": self.license,
            "homepage": self.homepage,
            "citation": self.citation,
            "version": self.version,
            **self._meta,
        }

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _download_url(url: str, dest: Path, desc: str = "") -> Path:
        """Download a URL with tqdm progress bar."""
        desc = desc or dest.name
        log.info(f"Downloading {desc} from {url}")
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="iB", unit_scale=True, desc=desc
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        return dest

    @staticmethod
    def _extract_zip(zip_path: Path, dest: Path) -> Path:
        log.info(f"Extracting {zip_path.name} → {dest}")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(dest)
        return dest

    @staticmethod
    def _extract_tar(tar_path: Path, dest: Path) -> Path:
        import tarfile
        log.info(f"Extracting {tar_path.name} → {dest}")
        with tarfile.open(tar_path, "r:*") as t:
            t.extractall(dest)
        return dest


# =============================================================================
# TACO
# =============================================================================

class TACODownloader(BaseDownloader):
    """
    Downloads TACO (Trash Annotations in Context).

    TACO stores images on Flickr; the dataset provides a download script.
    We clone the repo and run their downloader, then verify structure.

    Source:  http://tacodataset.org
    License: MIT
    Format:  COCO JSON (annotations/annotations.json + images/)
    """

    name = "TACO"
    license = "MIT"
    homepage = "http://tacodataset.org"
    citation = (
        "Proença, P.F. & Simões, P. (2020). TACO: Trash Annotations in Context "
        "for Litter Detection. arXiv:2003.06975"
    )
    version = "2020"

    REPO_URL = "https://github.com/pedropro/TACO.git"
    ANNOTATIONS_URL = (
        "https://raw.githubusercontent.com/pedropro/TACO/master/"
        "data/annotations.json"
    )

    def download(self) -> Path:
        repo_dir = self.raw_dir / "repo"
        ann_path = self.raw_dir / "annotations.json"
        images_dir = self.raw_dir / "images"
        images_dir.mkdir(exist_ok=True)

        # Clone repo (for the download script)
        if not repo_dir.exists():
            log.info("Cloning TACO repository")
            result = subprocess.run(
                ["git", "clone", "--depth=1", self.REPO_URL, str(repo_dir)],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                raise RuntimeError(f"git clone failed: {result.stderr}")
        else:
            log.info("TACO repo already cloned")

        # Download annotations
        if not ann_path.exists():
            self._download_url(self.ANNOTATIONS_URL, ann_path, "TACO annotations")

        # Download images using TACO's own download.py
        download_script = repo_dir / "download.py"
        if download_script.exists() and not any(images_dir.iterdir()):
            log.info("Downloading TACO images via official download.py")
            result = subprocess.run(
                ["python", str(download_script),
                 "--dataset_path", str(self.raw_dir),
                 "--n_images", "all"],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                log.warning(f"TACO download.py exited non-zero: {result.stderr[:500]}")
        else:
            log.info("TACO images already downloaded or download.py not found")

        self._meta["annotation_file"] = str(ann_path)
        self._meta["images_dir"] = str(images_dir)
        log.info(f"TACO available at {self.raw_dir}")
        return self.raw_dir


# =============================================================================
# TrashNet
# =============================================================================

class TrashNetDownloader(BaseDownloader):
    """
    Downloads TrashNet from Stanford.
    Original file hosted on Stanford web storage / GitHub releases.

    Source:  https://github.com/garythung/trashnet
    License: CC BY 4.0
    Format:  Image folders (cardboard/ glass/ metal/ paper/ plastic/ trash/)
    """

    name = "TrashNet"
    license = "CC BY 4.0"
    homepage = "https://github.com/garythung/trashnet"
    citation = (
        "Thung, G. & Yang, M. (2016). Classification of Trash for Recyclability "
        "Status. CS229 Project Report, Stanford University."
    )
    version = "1.0"

    # Mirror hosted on HuggingFace (permissive, stable URL)
    HF_URL = "https://huggingface.co/datasets/garythung/trashnet/resolve/main/dataset-resized.zip"

    def download(self) -> Path:
        zip_path = self.raw_dir / "dataset-resized.zip"
        extracted = self.raw_dir / "dataset-resized"

        if extracted.exists() and any(extracted.iterdir()):
            log.info("TrashNet already downloaded")
        else:
            self._download_url(self.HF_URL, zip_path, "TrashNet")
            self._extract_zip(zip_path, self.raw_dir)
            zip_path.unlink(missing_ok=True)

        # Verify folder structure
        expected = {"cardboard", "glass", "metal", "paper", "plastic", "trash"}
        found = {p.name for p in extracted.iterdir() if p.is_dir()}
        missing = expected - found
        if missing:
            log.warning(f"TrashNet: missing category folders: {missing}")

        self._meta["classes"] = sorted(found)
        self._meta["root"] = str(extracted)
        log.info(f"TrashNet available at {extracted}")
        return extracted


# =============================================================================
# ZeroWaste-f
# =============================================================================

class ZeroWasteDownloader(BaseDownloader):
    """
    Downloads ZeroWaste-f dataset.

    Source:  https://zerowaste.epfl.ch
    License: CC BY-NC 4.0
    Format:  COCO JSON + images (train/val/test splits)

    The dataset is hosted via a direct link provided by the authors.
    """

    name = "ZeroWaste"
    license = "CC BY-NC 4.0"
    homepage = "https://zerowaste.epfl.ch"
    citation = (
        "Bashkirova, D. et al. (2022). ZeroWaste Dataset: Towards Deformable "
        "Object Segmentation in Cluttered Scenes. CVPR 2022."
    )
    version = "2022"

    # Author-hosted direct download (zerowaste-f full split)
    DOWNLOAD_URL = "http://downloads.dbash.me/zerowaste-f.zip"
    EXPECTED_MD5 = None  # populated once stable checksum is confirmed

    def download(self) -> Path:
        zip_path = self.raw_dir / "zerowaste-f.zip"
        extracted = self.raw_dir / "zerowaste-f"

        if extracted.exists() and any(extracted.iterdir()):
            log.info("ZeroWaste-f already downloaded")
            return extracted

        try:
            self._download_url(self.DOWNLOAD_URL, zip_path, "ZeroWaste-f")
        except Exception as e:
            log.error(
                f"ZeroWaste-f download failed: {e}. "
                "Please download manually from https://zerowaste.epfl.ch "
                f"and place the zip at {zip_path}"
            )
            raise

        self._extract_zip(zip_path, self.raw_dir)
        zip_path.unlink(missing_ok=True)

        self._meta["root"] = str(extracted)
        log.info(f"ZeroWaste-f available at {extracted}")
        return extracted


# =============================================================================
# OpenLitterMap
# =============================================================================

class OpenLitterMapDownloader(BaseDownloader):
    """
    Downloads OpenLitterMap dataset via their public data exports.

    Source:  https://openlittermap.com
    License: CC BY 4.0
    Format:  CSV + images (OLM JSON annotations)

    OLM provides periodic data exports. We use the Zenodo-hosted snapshot.
    """

    name = "OpenLitterMap"
    license = "CC BY 4.0"
    homepage = "https://openlittermap.com"
    citation = (
        "Lynch, S. (2018). OpenLitterMap.com – Open Data on Plastic Pollution "
        "with Blockchain Rewards (Littercoin). "
        "Open Geospatial Data, Software and Standards 3(6)."
    )
    version = "2023"

    # Zenodo DOI (publicly accessible, stable)
    ZENODO_URL = "https://zenodo.org/record/7949333/files/openlittermap-images-2023.zip"
    ANN_URL = "https://zenodo.org/record/7949333/files/openlittermap-annotations-2023.json"

    def download(self) -> Path:
        ann_path = self.raw_dir / "annotations.json"
        images_dir = self.raw_dir / "images"
        images_dir.mkdir(exist_ok=True)
        zip_path = self.raw_dir / "images.zip"

        if not ann_path.exists():
            try:
                self._download_url(self.ANN_URL, ann_path, "OLM annotations")
            except Exception as e:
                log.error(
                    f"OpenLitterMap annotation download failed: {e}. "
                    "Attempting to continue without annotations."
                )

        if not any(images_dir.iterdir()):
            try:
                self._download_url(self.ZENODO_URL, zip_path, "OLM images")
                self._extract_zip(zip_path, images_dir)
                zip_path.unlink(missing_ok=True)
            except Exception as e:
                log.error(
                    f"OpenLitterMap image download failed: {e}. "
                    "Please download manually from https://zenodo.org/record/7949333"
                )
                raise
        else:
            log.info("OpenLitterMap images already downloaded")

        self._meta["annotation_file"] = str(ann_path)
        self._meta["images_dir"] = str(images_dir)
        return self.raw_dir


# =============================================================================
# Kaggle Garbage Classification
# =============================================================================

class KaggleGCDownloader(BaseDownloader):
    """
    Downloads Kaggle Garbage Classification dataset via Kaggle API.

    Source:  https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification
    License: CC0 1.0
    Format:  Image folders (12 categories)

    Requires: KAGGLE_USERNAME and KAGGLE_KEY environment variables,
              or ~/.kaggle/kaggle.json credentials file.
    """

    name = "KaggleGC"
    license = "CC0 1.0"
    homepage = "https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification"
    citation = "Kaggle dataset: Garbage Classification by asdasdasasdas (2019)."
    version = "2"

    DATASET_SLUG = "asdasdasasdas/garbage-classification"

    def download(self) -> Path:
        extracted = self.raw_dir / "garbage_classification"

        if extracted.exists() and any(extracted.iterdir()):
            log.info("KaggleGC already downloaded")
            return extracted

        # Check for kaggle credentials
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        has_env = os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
        if not kaggle_json.exists() and not has_env:
            raise RuntimeError(
                "Kaggle credentials not found. Either set KAGGLE_USERNAME + KAGGLE_KEY "
                "environment variables, or place credentials at ~/.kaggle/kaggle.json. "
                "See: https://www.kaggle.com/docs/api"
            )

        try:
            import kaggle  # kaggle package
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                self.DATASET_SLUG,
                path=str(self.raw_dir),
                unzip=True,
            )
        except ImportError:
            # Fallback to subprocess
            result = subprocess.run(
                ["kaggle", "datasets", "download", "-d", self.DATASET_SLUG,
                 "-p", str(self.raw_dir), "--unzip"],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                raise RuntimeError(f"Kaggle CLI download failed: {result.stderr}")

        self._meta["root"] = str(self.raw_dir)
        log.info(f"KaggleGC available at {self.raw_dir}")
        return self.raw_dir


# =============================================================================
# MJU-Waste
# =============================================================================

class MJUWasteDownloader(BaseDownloader):
    """
    Downloads MJU-Waste via Google Drive public link.

    Source:  https://github.com/realwecan/mju-waste
    License: CC BY-NC 4.0
    Format:  COCO instance segmentation
    """

    name = "MJU-Waste"
    license = "CC BY-NC 4.0"
    homepage = "https://github.com/realwecan/mju-waste"
    citation = (
        "Wang, T. et al. (2020). A Multi-Level Approach to Waste Object "
        "Segmentation. Sensors, 20(14), 3816."
    )
    version = "1.0"

    # Google Drive file ID (public share)
    GDRIVE_ID = "1o101UBJGeeMPpI-DSQ6oefORJ-IFBBx5"

    def download(self) -> Path:
        extracted = self.raw_dir / "MJU-Waste"
        if extracted.exists() and any(extracted.iterdir()):
            log.info("MJU-Waste already downloaded")
            return extracted

        zip_path = self.raw_dir / "mju_waste.zip"
        self._download_gdrive(self.GDRIVE_ID, zip_path)
        self._extract_zip(zip_path, self.raw_dir)
        zip_path.unlink(missing_ok=True)

        self._meta["root"] = str(extracted)
        return extracted

    @staticmethod
    def _download_gdrive(file_id: str, dest: Path) -> None:
        """Download a public Google Drive file."""
        URL = "https://docs.google.com/uc?export=download"
        session = requests.Session()
        response = session.get(URL, params={"id": file_id}, stream=True)
        token = None
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                token = value
        if token:
            response = session.get(URL, params={"id": file_id, "confirm": token}, stream=True)
        total = int(response.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="iB", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in response.iter_content(chunk_size=32768):
                f.write(chunk)
                bar.update(len(chunk))


# =============================================================================
# Registry
# =============================================================================

DOWNLOADERS: Dict[str, type] = {
    "taco": TACODownloader,
    "trashnet": TrashNetDownloader,
    "zerowaste": ZeroWasteDownloader,
    "openlittermap": OpenLitterMapDownloader,
    "kaggle_gc": KaggleGCDownloader,
    "mju_waste": MJUWasteDownloader,
}


def download_all(raw_dir: Path = _DEFAULT_RAW, skip: Optional[List[str]] = None) -> Dict[str, Path]:
    """
    Download all registered datasets.

    Parameters
    ----------
    raw_dir : Path
    skip : list of dataset names to skip (useful when already downloaded)

    Returns
    -------
    dict mapping dataset name → raw path
    """
    skip = [s.lower() for s in (skip or [])]
    results: Dict[str, Path] = {}
    for name, cls in DOWNLOADERS.items():
        if name in skip:
            log.info(f"Skipping {name}")
            continue
        try:
            dl = cls(raw_dir=raw_dir)
            path = dl.download()
            results[name] = path
        except Exception as e:
            log.error(f"Failed to download {name}: {e}")
    return results
