# © 2026 TimeWealthy Limited — DeepGuard
"""Tests for MetadataAnalyzer."""

import pytest
from unittest.mock import patch, MagicMock

from app.analyzers.metadata import MetadataAnalyzer


class TestMetadataAnalyzer:

    def test_analyze_no_ffprobe_returns_gracefully(self, mock_video_path):
        """ffprobe がない → エラーなし、スコアは何らかの値を返す"""
        analyzer = MetadataAnalyzer()
        with patch("subprocess.run", side_effect=FileNotFoundError("ffprobe not found")):
            result = analyzer.analyze(mock_video_path)
        # When ffprobe is unavailable, we get empty metadata and fall back gracefully
        assert 0.0 <= result.score <= 100.0

    def test_analyze_with_empty_metadata(self, mock_video_path):
        """空のメタデータ → エラーなし、スコア正常"""
        analyzer = MetadataAnalyzer()
        with patch.object(analyzer, "_extract_metadata", return_value={}):
            result = analyzer.analyze(mock_video_path)
        assert 0.0 <= result.score <= 100.0

    def test_ai_tool_signature_detected(self, mock_video_path):
        """AI生成ツールのシグネチャ → codec_signature findingが存在すること"""
        analyzer = MetadataAnalyzer()
        fake_metadata = {
            "format": {
                "tags": {"encoder": "Sora Video Generation v2", "title": "test"}
            },
            "streams": [],
        }
        with patch.object(analyzer, "_extract_metadata", return_value=fake_metadata):
            result = analyzer.analyze(mock_video_path)

        # Most important: codec_signature finding must exist
        sig_findings = [f for f in result.findings if f.type == "codec_signature"]
        assert len(sig_findings) > 0
        assert sig_findings[0].confidence >= 90.0
        # Score should be elevated (above 50%)
        assert result.score >= 50.0

    def test_clean_camera_metadata_low_score(self, mock_video_path):
        """正規カメラのメタデータ → 低スコア"""
        analyzer = MetadataAnalyzer()
        fake_metadata = {
            "format": {
                "tags": {
                    "encoder": "Lavf58.45.100",
                    "creation_time": "2026-01-01T12:00:00Z",
                    "make": "Canon",
                    "model": "EOS R5",
                }
            },
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "bit_rate": "8000000",
                    "r_frame_rate": "30/1",
                    "avg_frame_rate": "30000/1001",  # Different from r_frame_rate
                }
            ],
        }
        with patch.object(analyzer, "_extract_metadata", return_value=fake_metadata):
            result = analyzer.analyze(mock_video_path)

        assert result.score < 60.0

    def test_low_bitrate_flagged(self, mock_video_path):
        """低ビットレート → low_bitrate フラグ"""
        analyzer = MetadataAnalyzer()
        fake_metadata = {
            "format": {"tags": {}},
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "bit_rate": "50000",  # Very low: 50 kbps
                    "r_frame_rate": "30/1",
                    "avg_frame_rate": "30/1",
                }
            ],
        }
        with patch.object(analyzer, "_extract_metadata", return_value=fake_metadata):
            result = analyzer.analyze(mock_video_path)

        low_br_findings = [f for f in result.findings if f.type == "low_bitrate"]
        assert len(low_br_findings) > 0

    def test_minimal_metadata_flagged(self, mock_video_path):
        """メタデータが最小限 → minimal_metadata フラグ"""
        analyzer = MetadataAnalyzer()
        fake_metadata = {
            "format": {"tags": {}},  # No tags at all
            "streams": [],
        }
        with patch.object(analyzer, "_extract_metadata", return_value=fake_metadata):
            result = analyzer.analyze(mock_video_path)

        minimal_findings = [f for f in result.findings if f.type == "minimal_metadata"]
        assert len(minimal_findings) > 0

    def test_score_always_0_to_100(self, mock_video_path):
        """スコアは常に0-100"""
        analyzer = MetadataAnalyzer()
        for metadata in [
            {},
            {"format": {"tags": {}}, "streams": []},
            {"format": {"tags": {"encoder": "sora"}}, "streams": []},
        ]:
            with patch.object(analyzer, "_extract_metadata", return_value=metadata):
                result = analyzer.analyze(mock_video_path)
            assert 0.0 <= result.score <= 100.0

    def test_findings_have_required_fields(self, mock_video_path):
        """Finding に必須フィールドがある"""
        analyzer = MetadataAnalyzer()
        fake_metadata = {
            "format": {"tags": {"encoder": "pika 1.0"}},
            "streams": [],
        }
        with patch.object(analyzer, "_extract_metadata", return_value=fake_metadata):
            result = analyzer.analyze(mock_video_path)

        for finding in result.findings:
            assert hasattr(finding, "type")
            assert hasattr(finding, "confidence")
            assert hasattr(finding, "description")
