"""PaddleOCR-VL adapter narrow seam: routing, layout mapping, subprocess contract."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from wenmai.components.paddleocr.adapter import (
    build_text_from_ocr_payload,
    choose_pdf_route,
    measure_pdf_chars_per_page,
    parse_scanned_pdf,
)
from wenmai.config import PaddleOCR, PdfLoad, Settings
from wenmai.storage.images import ImageStore

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "paddleocr"


@pytest.fixture
def paddleocr_config() -> PaddleOCR:
    return PaddleOCR(
        python="/fake/paddleocr-env/bin/python",
        script="/fake/scripts/paddleocr_parse_pdf.py",
        mlx_python="/fake/paddleocr-env/bin/python",
        mlx_model="mlx-community/PaddleOCR-VL-1.6-5bit",
        mlx_fallback_model="PaddlePaddle/PaddleOCR-VL-1.6",
        server_port=8111,
        server_url="http://127.0.0.1:8111/",
        vl_rec_api_model_name="mlx-community/PaddleOCR-VL-1.6-5bit",
        idle_timeout_seconds=60,
    )


@pytest.fixture
def pdf_load_config() -> PdfLoad:
    return PdfLoad(mode="auto", chars_per_page_threshold=100)


def _fixture_runner(fixture_name: str):
    fixture_path = FIXTURES / fixture_name

    def runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        output_idx = cmd.index("-o")
        output_path = Path(cmd[output_idx + 1])
        output_path.write_text(fixture_path.read_text(encoding="utf-8"), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    return runner


def _write_text_pdf(path: Path, text: str) -> Path:
    pdf = canvas.Canvas(str(path), pagesize=letter)
    pdf.drawString(72, 720, text)
    pdf.save()
    return path


def _write_image_only_pdf(path: Path) -> Path:
    image_path = path.with_suffix(".png")
    Image.new("RGB", (80, 80), color=(20, 80, 140)).save(image_path)

    pdf = canvas.Canvas(str(path), pagesize=letter)
    pdf.drawImage(str(image_path), 72, 580, width=160, height=160)
    pdf.save()
    return path


def test_measure_pdf_chars_per_page_counts_extractable_text(tmp_path: Path) -> None:
    source = _write_text_pdf(
        tmp_path / "dense.pdf",
        "湄洲岛是妈祖信仰的发源地，祖庙是信俗活动的中心场所。" * 8,
    )
    assert measure_pdf_chars_per_page(source) >= 100


def test_measure_pdf_chars_per_page_is_zero_for_image_only_pdf(tmp_path: Path) -> None:
    source = _write_image_only_pdf(tmp_path / "scan.pdf")
    assert measure_pdf_chars_per_page(source) == 0.0


def test_choose_pdf_route_auto_prefers_markitdown_for_dense_pdf(
    tmp_path: Path, pdf_load_config: PdfLoad
) -> None:
    source = _write_text_pdf(
        tmp_path / "dense.pdf",
        "湄洲岛是妈祖信仰的发源地，祖庙是信俗活动的中心场所。" * 8,
    )
    assert choose_pdf_route(source, pdf_load_config) == "markitdown"


def test_choose_pdf_route_auto_prefers_ocr_for_image_only_pdf(
    tmp_path: Path, pdf_load_config: PdfLoad
) -> None:
    source = _write_image_only_pdf(tmp_path / "scan.pdf")
    assert choose_pdf_route(source, pdf_load_config) == "paddleocr-vl"


def test_choose_pdf_route_override_forces_markitdown(
    tmp_path: Path, pdf_load_config: PdfLoad
) -> None:
    source = _write_image_only_pdf(tmp_path / "scan.pdf")
    assert choose_pdf_route(source, pdf_load_config, override_mode="markitdown") == "markitdown"


def test_choose_pdf_route_override_forces_ocr(
    tmp_path: Path, pdf_load_config: PdfLoad
) -> None:
    source = _write_text_pdf(tmp_path / "dense.pdf", "妈祖祖庙简介。" * 20)
    assert choose_pdf_route(source, pdf_load_config, override_mode="ocr") == "paddleocr-vl"


def test_build_text_from_ocr_payload_maps_layout_labels(
    test_settings: Settings, tmp_path: Path
) -> None:
    payload = json.loads((FIXTURES / "sample_layout.json").read_text(encoding="utf-8"))
    image_store = ImageStore(test_settings)
    text = build_text_from_ocr_payload(
        payload,
        image_store,
        document_id="doc-33",
        source_path=str(tmp_path / "scan.pdf"),
    )

    assert "湄洲岛是妈祖信仰的发源地。" in text
    assert "图表：历年祭典参与人数呈上升趋势。" in text
    assert "印章：湄洲妈祖祖庙" in text
    assert "[IMAGE:" in text
    assert "header_image" not in text
    assert "footer_image" not in text


def test_parse_scanned_pdf_raises_on_nonzero_exit_without_markitdown_fallback(
    paddleocr_config: PaddleOCR, tmp_path: Path
) -> None:
    pdf = _write_image_only_pdf(tmp_path / "scan.pdf")

    def failing_runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            cmd,
            returncode=1,
            stdout="",
            stderr="layout model failed to load",
        )

    class _NoopManager:
        def ensure_running(self) -> None:
            return None

        def touch(self) -> None:
            return None

    with pytest.raises(RuntimeError, match="PaddleOCR-VL parsing failed") as exc_info:
        parse_scanned_pdf(
            pdf,
            config=paddleocr_config,
            runner=failing_runner,
            server_manager=_NoopManager(),
        )

    message = str(exc_info.value)
    assert "exit code 1" in message
    assert "layout model failed" in message
    assert "MarkItDown" not in message


def test_parse_scanned_pdf_invokes_subprocess_with_env_strip_and_vl_args(
    paddleocr_config: PaddleOCR, tmp_path: Path
) -> None:
    pdf = _write_image_only_pdf(tmp_path / "scan.pdf")
    captured: list[list[str]] = []

    def capture_runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        captured.append(cmd)
        output_idx = cmd.index("-o")
        output_path = Path(cmd[output_idx + 1])
        output_path.write_text(
            (FIXTURES / "sample_layout.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    class _NoopManager:
        def ensure_running(self) -> None:
            return None

        def touch(self) -> None:
            return None

    payload = parse_scanned_pdf(
        pdf,
        config=paddleocr_config,
        runner=capture_runner,
        server_manager=_NoopManager(),
    )

    assert payload["pages"]
    assert len(captured) == 1
    cmd = captured[0]
    assert cmd[:5] == ["env", "-u", "PYTHONHOME", "-u", "PYTHONPATH"]
    assert cmd[5] == paddleocr_config.python
    assert cmd[6] == paddleocr_config.script
    assert cmd[7] == str(pdf)
    assert cmd[cmd.index("-o") + 1].endswith(".json")
    assert cmd[cmd.index("--vl-rec-backend") + 1] == "mlx-vlm-server"
    assert cmd[cmd.index("--vl-rec-server-url") + 1] == paddleocr_config.server_url
    assert (
        cmd[cmd.index("--vl-rec-api-model-name") + 1]
        == paddleocr_config.vl_rec_api_model_name
    )


def test_ingesting_scanned_pdf_records_paddleocr_trace_and_placeholder(
    test_settings: Settings, tmp_path: Path
) -> None:
    from fastapi.testclient import TestClient

    from wenmai.app import create_app
    from wenmai.ingestion import loaders as loaders_module

    pdf = _write_image_only_pdf(tmp_path / "scan.pdf")
    original_parse = loaders_module.parse_scanned_pdf

    def fake_parse(pdf_path, **kwargs):
        return json.loads((FIXTURES / "sample_layout.json").read_text(encoding="utf-8"))

    loaders_module.parse_scanned_pdf = fake_parse
    try:
        client = TestClient(create_app(test_settings))
        response = client.post("/ingest", json={"source_path": str(pdf)})
        assert response.status_code == 200
        body = response.json()
        assert body["chunk_count"] >= 1

        trace_path = Path(test_settings.paths.traces)
        trace = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[-1])
        load_stage = next(stage for stage in trace["stages"] if stage["name"] == "load")
        assert load_stage["method"] == "paddleocr-vl"
        assert load_stage["provider"] == "mlx-vlm-server"
    finally:
        loaders_module.parse_scanned_pdf = original_parse
