"""Kiểm tra đọc cấu hình chấm của evaluate_practice.py."""

import json

import pytest

import evaluate_practice as ep


def test_doc_eval_config_khi_co_file(tmp_path):
    (tmp_path / "video_1").mkdir()
    (tmp_path / "video_1" / "eval_config.json").write_text(json.dumps({"benchmark": "ABC", "split": "train"}))
    assert ep._load_eval_config(tmp_path)["benchmark"] == "ABC"


def test_eval_config_mac_dinh_khi_co_nhan(tmp_path):
    (tmp_path / "video_1" / "gt").mkdir(parents=True)
    (tmp_path / "video_1" / "gt" / "gt.txt").write_text("")
    assert ep._load_eval_config(tmp_path) == ep.DEFAULT_EVAL_CONFIG


def test_eval_config_bao_loi_khi_thieu_nhan(tmp_path):
    (tmp_path / "video_1").mkdir()
    with pytest.raises(FileNotFoundError):
        ep._load_eval_config(tmp_path)
