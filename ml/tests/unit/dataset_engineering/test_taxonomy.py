"""Tests for TaxonomyLoader."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

# Reset singleton before each test module
import lib.ml.dataset_engineering.taxonomy as _tax_module


@pytest.fixture(autouse=True)
def reset_taxonomy_singleton():
    """Reset singleton so tests can load different taxonomy files."""
    _tax_module.TaxonomyLoader._instance = None
    yield
    _tax_module.TaxonomyLoader._instance = None


class TestTaxonomyLoader:
    def test_load_full_taxonomy(self):
        """Load the real 109-class taxonomy from configs/."""
        from lib.ml.dataset_engineering.taxonomy import TaxonomyLoader
        tax = TaxonomyLoader(Path("configs/taxonomy/spiro_taxonomy.yaml"))
        assert tax.num_classes == 109

    def test_id_to_name(self):
        from lib.ml.dataset_engineering.taxonomy import TaxonomyLoader
        tax = TaxonomyLoader(Path("configs/taxonomy/spiro_taxonomy.yaml"))
        assert tax.id_to_name(0) == "plastic_bottle"
        assert tax.id_to_name(53) == "cigarette_butt"

    def test_name_to_id(self):
        from lib.ml.dataset_engineering.taxonomy import TaxonomyLoader
        tax = TaxonomyLoader(Path("configs/taxonomy/spiro_taxonomy.yaml"))
        assert tax.name_to_id("plastic_bottle") == 0
        assert tax.name_to_id("unknown_litter") == 106

    def test_invalid_id_raises(self):
        from lib.ml.dataset_engineering.taxonomy import TaxonomyLoader
        tax = TaxonomyLoader(Path("configs/taxonomy/spiro_taxonomy.yaml"))
        with pytest.raises(KeyError):
            tax.id_to_name(9999)

    def test_invalid_name_raises(self):
        from lib.ml.dataset_engineering.taxonomy import TaxonomyLoader
        tax = TaxonomyLoader(Path("configs/taxonomy/spiro_taxonomy.yaml"))
        with pytest.raises(KeyError):
            tax.name_to_id("nonexistent_class")

    def test_group_ids(self):
        from lib.ml.dataset_engineering.taxonomy import TaxonomyLoader
        tax = TaxonomyLoader(Path("configs/taxonomy/spiro_taxonomy.yaml"))
        plastic_ids = tax.group_ids("plastic")
        assert 0 in plastic_ids   # plastic_bottle
        assert 4 in plastic_ids   # plastic_straw

    def test_all_names_length(self):
        from lib.ml.dataset_engineering.taxonomy import TaxonomyLoader
        tax = TaxonomyLoader(Path("configs/taxonomy/spiro_taxonomy.yaml"))
        assert len(tax.all_names()) == 109

    def test_all_ids_contiguous(self):
        from lib.ml.dataset_engineering.taxonomy import TaxonomyLoader
        tax = TaxonomyLoader(Path("configs/taxonomy/spiro_taxonomy.yaml"))
        assert tax.all_ids() == list(range(109))

    def test_validate_id(self):
        from lib.ml.dataset_engineering.taxonomy import TaxonomyLoader
        tax = TaxonomyLoader(Path("configs/taxonomy/spiro_taxonomy.yaml"))
        assert tax.validate_id(0) is True
        assert tax.validate_id(108) is True
        assert tax.validate_id(109) is False
        assert tax.validate_id(-1) is False

    def test_singleton_returns_same_instance(self):
        from lib.ml.dataset_engineering.taxonomy import TaxonomyLoader
        t1 = TaxonomyLoader(Path("configs/taxonomy/spiro_taxonomy.yaml"))
        t2 = TaxonomyLoader()
        assert t1 is t2

    def test_id_to_group(self):
        from lib.ml.dataset_engineering.taxonomy import TaxonomyLoader
        tax = TaxonomyLoader(Path("configs/taxonomy/spiro_taxonomy.yaml"))
        assert tax.id_to_group(0) == "plastic"
        assert tax.id_to_group(22) == "glass"
        assert tax.id_to_group(53) == "cigarette"
