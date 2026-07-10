"""Tests for data filtering and rating bin logic."""

import pytest
from src.data_pipeline.filter import (
    get_rating_bin,
    is_bullet,
    passes_global_filters,
    filter_to_rating_bin,
    assign_bin_label,
    RATING_BINS,
)


class TestRatingBins:
    def test_get_rating_bin(self):
        assert get_rating_bin(1100) == 1100
        assert get_rating_bin(1199) == 1100
        assert get_rating_bin(1200) == 1200
        assert get_rating_bin(1899) == 1800
        assert get_rating_bin(1099) is None
        assert get_rating_bin(1900) is None
        assert get_rating_bin(2500) is None

    def test_rating_bins_correct_count(self):
        assert len(RATING_BINS) == 8
        assert RATING_BINS == [1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800]

    def test_assign_bin_label(self):
        assert assign_bin_label(1150) == "1100-1199"
        assert assign_bin_label(1500) == "1500-1599"
        assert assign_bin_label(1099) is None
        assert assign_bin_label(1900) is None


class TestBulletFilter:
    def test_is_bullet(self):
        assert is_bullet("bullet") is True
        assert is_bullet("blitz") is False
        assert is_bullet("rapid") is False
        assert is_bullet("classical") is False

    def test_passes_global_filters(self):
        # Passes all filters
        rec = {
            "tc_category": "blitz",
            "clock_remaining": 120.0,
        }
        assert passes_global_filters(rec) is True

    def test_fails_bullet(self):
        rec = {"tc_category": "bullet", "clock_remaining": 120.0}
        assert passes_global_filters(rec) is False

    def test_fails_low_clock(self):
        rec = {"tc_category": "blitz", "clock_remaining": 10.0}
        assert passes_global_filters(rec) is False

    def test_no_clock_passes(self):
        rec = {"tc_category": "blitz", "clock_remaining": None}
        assert passes_global_filters(rec) is True


class TestFilterToRatingBin:
    def test_filter_both_players_in_bin(self):
        records = [
            {"white_rating": 1150, "black_rating": 1180, "side_to_move": "white"},
            {"white_rating": 1150, "black_rating": 1300, "side_to_move": "white"},
            {"white_rating": 1200, "black_rating": 1150, "side_to_move": "black"},
            {"white_rating": 1500, "black_rating": 1500, "side_to_move": "white"},
        ]
        filtered = filter_to_rating_bin(records, 1100, require_both_players=True)
        assert len(filtered) == 1
        assert filtered[0]["white_rating"] == 1150

    def test_filter_side_to_move(self):
        records = [
            {"white_rating": 1150, "black_rating": 1300, "side_to_move": "white"},
            {"white_rating": 1300, "black_rating": 1150, "side_to_move": "black"},
        ]
        filtered = filter_to_rating_bin(records, 1100, require_both_players=False)
        assert len(filtered) == 2

    def test_no_records_in_bin(self):
        records = [
            {"white_rating": 1500, "black_rating": 1500, "side_to_move": "white"},
        ]
        filtered = filter_to_rating_bin(records, 1100, require_both_players=True)
        assert len(filtered) == 0

    def test_missing_ratings(self):
        records = [
            {"white_rating": None, "black_rating": 1150, "side_to_move": "white"},
            {"white_rating": 1150, "black_rating": None, "side_to_move": "white"},
        ]
        filtered = filter_to_rating_bin(records, 1100)
        assert len(filtered) == 0
