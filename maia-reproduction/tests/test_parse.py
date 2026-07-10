"""Tests for PGN parsing utilities."""

import chess.pgn
import io
import pytest

from src.data_pipeline.parse import (
    parse_time_control,
    parse_clock_comment,
    parse_eval_comment,
    parse_game,
)


class TestTimeControl:
    def test_parse_standard_tc(self):
        result = parse_time_control({"TimeControl": "300+3"})
        assert result is not None
        assert result["initial"] == 300
        assert result["increment"] == 3
        # 300 + 40*3 = 420s -> blitz (180-479)
        assert result["category"] == "blitz"

    def test_parse_blitz(self):
        result = parse_time_control({"TimeControl": "180+0"})
        assert result is not None
        assert result["category"] == "blitz"

    def test_parse_bullet(self):
        result = parse_time_control({"TimeControl": "60+0"})
        assert result is not None
        assert result["category"] == "bullet"

    def test_parse_classical(self):
        result = parse_time_control({"TimeControl": "1800+0"})
        assert result is not None
        assert result["category"] == "classical"

    def test_missing_tc(self):
        result = parse_time_control({})
        assert result is None

    def test_none_tc(self):
        result = parse_time_control({"TimeControl": "-"})
        assert result is None


class TestClockComment:
    def test_parse_full_time(self):
        assert parse_clock_comment("[%clk 1:30:00]") == 5400.0

    def test_parse_partial_time(self):
        assert parse_clock_comment("[%clk 0:05:30]") == 330.0

    def test_no_comment(self):
        assert parse_clock_comment("") is None

    def test_no_clock(self):
        assert parse_clock_comment("some other comment") is None


class TestEvalComment:
    def test_parse_centipawn(self):
        assert parse_eval_comment("[%eval 0.50]") == 0.50
        assert parse_eval_comment("[%eval -1.25]") == -1.25

    def test_parse_mate(self):
        # Mate in 3 for side to move
        val = parse_eval_comment("[%eval #3]")
        assert val is not None and val > 9000

    def test_no_eval(self):
        assert parse_eval_comment("") is None

    def test_mixed_comment(self):
        assert parse_eval_comment("[%clk 0:30:00] [%eval 0.75]") == 0.75


class TestParseGame:
    def test_parse_initial_position(self):
        pgn = """
[Event "Test"]
[Site "?"]
[Date "2020.01.01"]
[Round "1"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]
[WhiteElo "1500"]
[BlackElo "1500"]
[TimeControl "300+0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0
"""
        game = chess.pgn.read_game(io.StringIO(pgn))
        assert game is not None
        records = parse_game(game, min_clock=0, skip_first_n_ply=0)
        assert len(records) >= 6
        assert records[0]["move_uci"] == "e2e4"
        assert records[0]["white_rating"] == 1500
        assert records[0]["black_rating"] == 1500
        assert records[0]["tc_category"] == "blitz"

    def test_skip_first_ply(self):
        pgn = """
[Event "Test"]
[WhiteElo "1500"]
[BlackElo "1500"]
[TimeControl "300+0"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 1-0
"""
        game = chess.pgn.read_game(io.StringIO(pgn))
        assert game is not None
        records = parse_game(game, min_clock=0, skip_first_n_ply=2)
        assert len(records) >= 2  # Nf3 and Nc6
        assert all(r["ply"] > 2 for r in records)

    def test_missing_ratings(self):
        pgn = """
[Event "Test"]
[WhiteElo ""]
[BlackElo ""]
[TimeControl "300+0"]
[Result "1-0"]

1. e4 e5 1-0
"""
        game = chess.pgn.read_game(io.StringIO(pgn))
        assert game is not None
        records = parse_game(game)
        # Empty ratings become None, records are still parsed
        # Rating filtering happens in a later pipeline stage
        assert len(records) > 0
        assert records[0]["white_rating"] is None
        assert records[0]["black_rating"] is None

    def test_clock_filtering(self):
        pgn = """
[Event "Test"]
[WhiteElo "1500"]
[BlackElo "1500"]
[TimeControl "300+0"]
[Result "1-0"]

1. e4 {[%clk 0:00:10]} e5 1-0
"""
        game = chess.pgn.read_game(io.StringIO(pgn))
        assert game is not None
        records = parse_game(game, min_clock=30.0)
        # e4 had only 10s clock, should be filtered out
        assert len(records) == 1  # only e5
