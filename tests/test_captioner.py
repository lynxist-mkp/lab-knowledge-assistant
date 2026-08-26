"""Ingestion seam: captioner replaces image placeholders or keeps them on vision failure."""

from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from wenmai.app import create_app
from wenmai.config import Settings
from wenmai.factories import vector_store as vector_store_factory


def _write_pdf_with_embedded_image(path: Path) -> Path:
    image_path = path.with_suffix(".png")
    Image.new("RGB", (40, 40), color=(180, 40, 40)).save(image_path)

    pdf = canvas.Canvas(str(path), pagesize=letter)
    pdf.drawString(72, 720, "Fuzhou shipyard heritage photo below.")
    pdf.drawString(
        72,
        700,
        "The Minpai shipbuilding tradition spans centuries along the Fujian coast. "
        "Artifacts from the yard document maritime trade and naval engineering.",
    )
    pdf.drawImage(str(image_path), 72, 580, width=120, height=120)
    pdf.save()
    return path


def _captioner_settings(test_settings: Settings) -> Settings:
    test_settings.transform.stages = ["refiner", "enricher", "captioner"]
    return test_settings


def test_captioner_replaces_placeholder_with_fake_vision_description(
    test_settings: Settings, tmp_path: Path
) -> None:
    settings = _captioner_settings(test_settings)
    source = _write_pdf_with_embedded_image(tmp_path / "shipyard.pdf")
    client = TestClient(create_app(settings))

    response = client.post("/ingest", json={"source_path": str(source)})

    assert response.status_code == 200
    body = response.json()
    store = vector_store_factory.create(settings)
    chunks = store.get_by_document_id(body["document_id"])
    combined = "\n".join(chunk.text for chunk in chunks)
    assert "[IMAGE:" not in combined
    assert "图片占位说明" in combined

    trace_path = Path(settings.paths.traces)
    trace = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[-1])
    captioner_stage = next(stage for stage in trace["stages"] if stage["name"] == "captioner")
    assert captioner_stage["method"] == "vision"
    assert "captioned" in captioner_stage["output_summary"]


def test_captioner_keeps_placeholder_when_vision_fails(
    test_settings: Settings, tmp_path: Path
) -> None:
    settings = _captioner_settings(test_settings)
    settings.fakes["vision"] = "error"
    source = _write_pdf_with_embedded_image(tmp_path / "shipyard.pdf")
    client = TestClient(create_app(settings))

    response = client.post("/ingest", json={"source_path": str(source)})

    assert response.status_code == 200
    body = response.json()
    store = vector_store_factory.create(settings)
    chunks = store.get_by_document_id(body["document_id"])
    combined = "\n".join(chunk.text for chunk in chunks)
    assert re.search(r"\[IMAGE: [a-f0-9]+\]", combined)

    trace_path = Path(settings.paths.traces)
    trace = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[-1])
    captioner_stage = next(stage for stage in trace["stages"] if stage["name"] == "captioner")
    assert captioner_stage.get("error")
