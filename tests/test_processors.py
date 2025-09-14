"""Test suite for CSV processors using pytest with mocking."""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.processors.strong_processor import (
    normalize_df,
    process_strong_csv,
    EXPECTED_COLUMNS,
    NORMALIZED_COLUMNS,
)
from app.processors.hevy_processor import (
    normalize_hevy_df,
    process_hevy_csv,
    HEVY_EXPECTED_COLUMNS,
    HEVY_NORMALIZED_COLUMNS,
)


class TestStrongProcessor:
    """Test Strong CSV processor."""

    @pytest.fixture
    def strong_sample_data(self):
        """Sample Strong CSV data."""
        return {
            "Date": ["2024-01-15 10:00:00", "2024-01-15 10:00:00"],
            "Workout Name": ["Push Day", "Push Day"],
            "Duration": ["45m", "45m"],
            "Exercise Name": ["Bench Press", "Overhead Press"],
            "Set Order": [1, 1],
            "Weight": [135.0, 95.0],
            "Reps": [8, 10],
            "Distance": [None, None],
            "Seconds": [None, None],
            "RPE": [7, 6],
        }

    def test_expected_columns_defined(self):
        """Test that expected columns are properly defined."""
        assert len(EXPECTED_COLUMNS) == 10
        assert "Date" in EXPECTED_COLUMNS
        assert "Exercise Name" in EXPECTED_COLUMNS

    def test_normalize_df_success(self, strong_sample_data):
        """Test successful normalization of Strong data."""
        df = pd.DataFrame(strong_sample_data)
        normalized = normalize_df(df)

        assert len(normalized) == 2
        assert list(normalized.columns) == NORMALIZED_COLUMNS
        assert normalized["exercise"].iloc[0] == "Bench Press"
        assert normalized["weight"].iloc[0] == 135.0

    def test_normalize_df_missing_columns(self):
        """Test that missing columns raise appropriate error."""
        incomplete_data = pd.DataFrame({"Date": ["2024-01-15"], "Weight": [100]})

        with pytest.raises(ValueError, match="Missing columns"):
            normalize_df(incomplete_data)

    def test_date_parsing(self, strong_sample_data):
        """Test that dates are parsed correctly."""
        df = pd.DataFrame(strong_sample_data)
        normalized = normalize_df(df)

        assert normalized["date"].iloc[0] == "2024-01-15T10:00:00"

    @patch("app.processors.strong_processor.get_conn")
    @patch("app.processors.strong_processor.set_meta")
    def test_process_strong_csv_success(self, mock_set_meta, mock_get_conn):
        """Test successful Strong CSV processing."""
        # Mock database connection
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 2
        mock_conn.executemany.return_value = mock_cursor
        mock_conn.__enter__.return_value = mock_conn
        mock_get_conn.return_value = mock_conn

        # Mock CSV reading
        test_csv_path = Path("test.csv")
        sample_data = pd.DataFrame(
            {
                "Date": ["2024-01-15 10:00:00"],
                "Workout Name": ["Test"],
                "Duration": ["30m"],
                "Exercise Name": ["Bench Press"],
                "Set Order": [1],
                "Weight": [135.0],
                "Reps": [8],
                "Distance": [None],
                "Seconds": [None],
                "RPE": [7],
            }
        )

        with patch("pandas.read_csv", return_value=sample_data):
            with patch(
                "app.processors.strong_processor.count_sets", side_effect=[0, 1]
            ):
                result = process_strong_csv(test_csv_path)

        assert result == 1
        mock_set_meta.assert_called_once()


class TestHevyProcessor:
    """Test Hevy CSV processor."""

    @pytest.fixture
    def hevy_sample_data(self):
        """Sample Hevy CSV data."""
        return {
            "title": ["Push", "Push"],
            "start_time": ["14 Sep 2025, 17:41", "14 Sep 2025, 17:41"],
            "end_time": ["14 Sep 2025, 17:42", "14 Sep 2025, 17:43"],
            "description": ["", ""],
            "exercise_title": ["Bench Press (Barbell)", "Overhead Press"],
            "superset_id": [None, None],
            "exercise_notes": ["", ""],
            "set_index": [0, 1],
            "set_type": ["normal", "normal"],
            "weight_kg": [60.0, 40.0],
            "reps": [10, 8],
            "distance_km": [None, None],
            "duration_seconds": [None, None],
            "rpe": [7, 6],
        }

    def test_hevy_expected_columns_defined(self):
        """Test that Hevy expected columns are properly defined."""
        assert len(HEVY_EXPECTED_COLUMNS) == 14
        assert "title" in HEVY_EXPECTED_COLUMNS
        assert "exercise_title" in HEVY_EXPECTED_COLUMNS

    def test_normalize_hevy_df_success(self, hevy_sample_data):
        """Test successful normalization of Hevy data."""
        df = pd.DataFrame(hevy_sample_data)
        normalized = normalize_hevy_df(df)

        assert len(normalized) == 2
        assert list(normalized.columns) == HEVY_NORMALIZED_COLUMNS
        assert normalized["exercise"].iloc[0] == "Bench Press (Barbell)"
        assert normalized["weight"].iloc[0] == 60.0

    def test_hevy_date_parsing(self, hevy_sample_data):
        """Test that Hevy dates are parsed correctly."""
        df = pd.DataFrame(hevy_sample_data)
        normalized = normalize_hevy_df(df)

        assert normalized["date"].iloc[0] == "2025-09-14T17:41:00"

    def test_hevy_duration_calculation(self, hevy_sample_data):
        """Test that duration is calculated from start/end times."""
        df = pd.DataFrame(hevy_sample_data)
        normalized = normalize_hevy_df(df)

        # First workout: 1 minute duration
        assert normalized["duration_min"].iloc[0] == 1.0
        # Second workout: 2 minutes duration
        assert normalized["duration_min"].iloc[1] == 2.0

    def test_normalize_hevy_df_missing_columns(self):
        """Test that missing columns raise appropriate error."""
        incomplete_data = pd.DataFrame({"title": ["Test"], "weight_kg": [100]})

        with pytest.raises(ValueError, match="Missing columns for hevy import"):
            normalize_hevy_df(incomplete_data)

    @patch("app.processors.hevy_processor.get_conn")
    @patch("app.processors.hevy_processor.set_meta")
    def test_process_hevy_csv_success(self, mock_set_meta, mock_get_conn):
        """Test successful Hevy CSV processing."""
        # Mock database connection
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        mock_conn.executemany.return_value = mock_cursor
        mock_conn.__enter__.return_value = mock_conn
        mock_get_conn.return_value = mock_conn

        # Mock CSV reading
        test_csv_path = Path("test.csv")
        sample_data = pd.DataFrame(
            {
                "title": ["Push"],
                "start_time": ["14 Sep 2025, 17:41"],
                "end_time": ["14 Sep 2025, 17:42"],
                "description": [""],
                "exercise_title": ["Bench Press"],
                "superset_id": [None],
                "exercise_notes": [""],
                "set_index": [0],
                "set_type": ["normal"],
                "weight_kg": [60.0],
                "reps": [10],
                "distance_km": [None],
                "duration_seconds": [None],
                "rpe": [7],
            }
        )

        with patch("pandas.read_csv", return_value=sample_data):
            with patch(
                "app.processors.hevy_processor.count_hevy_sets", side_effect=[0, 1]
            ):
                result = process_hevy_csv(test_csv_path)

        assert result == 1
        mock_set_meta.assert_called_once()


class TestProcessorComparison:
    """Test differences between processors."""

    def test_column_mapping_differences(self):
        """Test that processors handle different column structures."""
        # Strong uses simpler column names
        strong_cols = set(EXPECTED_COLUMNS)
        hevy_cols = set(HEVY_EXPECTED_COLUMNS)

        # They should be different
        assert strong_cols != hevy_cols

        # But both should map to same normalized schema
        assert NORMALIZED_COLUMNS == HEVY_NORMALIZED_COLUMNS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
