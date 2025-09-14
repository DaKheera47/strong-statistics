"""Test suite for import routing system using pytest."""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import (
    get_enabled_import_type,
    detect_format_mismatch,
    get_processor_function,
    validate_file_constraints,
    load_config,
)


class TestConfiguration:
    """Test configuration loading and validation."""

    def test_config_loads(self):
        """Test that configuration loads without errors."""
        config = load_config()
        assert config is not None
        assert "import_types" in config
        assert "validation" in config

    def test_enabled_import_type(self):
        """Test getting the enabled import type."""
        enabled_type = get_enabled_import_type()
        assert enabled_type is not None
        assert enabled_type in ["strong", "hevy"]

    def test_config_has_required_fields(self):
        """Test that config has all required fields."""
        config = load_config()

        for import_type in ["strong", "hevy"]:
            assert import_type in config["import_types"]
            type_config = config["import_types"][import_type]

            assert "enabled" in type_config
            assert "header_validation" in type_config
            assert "processor_module" in type_config
            assert "processor_function" in type_config


class TestFormatDetection:
    """Test CSV format detection logic."""

    @pytest.fixture
    def strong_header(self):
        return "Date,Workout Name,Duration,Exercise Name,Set Order,Weight,Reps,Distance,Seconds,RPE"

    @pytest.fixture
    def hevy_header(self):
        return '"title","start_time","end_time","description","exercise_title","superset_id","exercise_notes","set_index","set_type","weight_kg","reps","distance_km","duration_seconds","rpe"'

    def test_detect_hevy_mismatch_when_strong_enabled(self, hevy_header):
        """Test detecting hevy format when strong is configured."""
        mismatch = detect_format_mismatch(hevy_header, "strong")
        assert mismatch == "hevy"

    def test_detect_strong_mismatch_when_hevy_enabled(self, strong_header):
        """Test detecting strong format when hevy is configured."""
        mismatch = detect_format_mismatch(strong_header, "hevy")
        assert mismatch == "strong"

    def test_no_mismatch_with_correct_format(self, strong_header):
        """Test no mismatch when format matches configuration."""
        mismatch = detect_format_mismatch(strong_header, "strong")
        assert mismatch is None

    def test_unknown_format_returns_none(self):
        """Test that completely unknown formats return None."""
        unknown_header = "Unknown,Headers,Here"
        mismatch = detect_format_mismatch(unknown_header, "strong")
        assert mismatch is None


class TestProcessorLoading:
    """Test dynamic processor loading."""

    def test_load_strong_processor(self):
        """Test loading strong processor function."""
        processor = get_processor_function("strong")
        assert processor is not None
        assert processor.__name__ == "process_strong_csv"
        assert callable(processor)

    def test_load_hevy_processor(self):
        """Test loading hevy processor function."""
        processor = get_processor_function("hevy")
        assert processor is not None
        assert processor.__name__ == "process_hevy_csv"
        assert callable(processor)

    def test_invalid_import_type_raises_error(self):
        """Test that invalid import type raises appropriate error."""
        with pytest.raises(ValueError, match="Unknown import type"):
            get_processor_function("invalid_type")


class TestFileValidation:
    """Test file constraint validation."""

    def test_valid_csv_content_type(self):
        """Test that valid CSV content types are accepted."""
        test_data = b"Date,Exercise\n2024-01-01,Bench Press"

        # Should not raise any exception
        validate_file_constraints(test_data, "text/csv")
        validate_file_constraints(test_data, "application/vnd.ms-excel")
        validate_file_constraints(test_data, "application/octet-stream")

    def test_invalid_content_type_raises_error(self):
        """Test that invalid content types raise ValueError."""
        test_data = b"some data"

        with pytest.raises(ValueError, match="Unsupported content type"):
            validate_file_constraints(test_data, "application/json")

    def test_file_size_limit_enforced(self):
        """Test that file size limits are enforced."""
        # Create data larger than 10MB (default limit)
        large_data = b"x" * (11 * 1024 * 1024)

        with pytest.raises(ValueError, match="File too large"):
            validate_file_constraints(large_data, "text/csv")

    def test_empty_file_handled(self):
        """Test that empty files are handled gracefully."""
        empty_data = b""

        # Should not raise file size error for empty files
        validate_file_constraints(empty_data, "text/csv")


class TestEndToEnd:
    """Test end-to-end scenarios with sample data."""

    @pytest.fixture
    def sample_data_dir(self):
        return Path(__file__).parent.parent / "sample_data"

    def test_strong_csv_processing_setup(self, sample_data_dir):
        """Test that Strong CSV can be set up for processing."""
        strong_csv = sample_data_dir / "strong_sample.csv"

        if strong_csv.exists():
            with open(strong_csv, "rb") as f:
                content = f.read()

            header = content.splitlines()[0].decode()
            enabled_type = get_enabled_import_type()

            # If strong is enabled, should not detect mismatch
            if enabled_type == "strong":
                mismatch = detect_format_mismatch(header, enabled_type)
                assert mismatch is None

                processor = get_processor_function(enabled_type)
                assert processor.__name__ == "process_strong_csv"

    def test_hevy_csv_mismatch_detection(self, sample_data_dir):
        """Test that Hevy CSV mismatch is properly detected."""
        hevy_csv = sample_data_dir / "hevy_sample.csv"

        if hevy_csv.exists():
            with open(hevy_csv, "rb") as f:
                content = f.read()

            header = content.splitlines()[0].decode()
            enabled_type = get_enabled_import_type()

            # Should detect mismatch if strong is enabled but hevy uploaded
            if enabled_type == "strong":
                mismatch = detect_format_mismatch(header, enabled_type)
                assert mismatch == "hevy"


class TestConfigurationSwitching:
    """Test configuration switching scenarios."""

    def test_configuration_determines_processing(self):
        """Test that configuration determines which processor is used."""
        enabled_type = get_enabled_import_type()
        processor = get_processor_function(enabled_type)

        if enabled_type == "strong":
            assert processor.__name__ == "process_strong_csv"
        elif enabled_type == "hevy":
            assert processor.__name__ == "process_hevy_csv"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
