"""Ingestion seam: PDF with embedded image → disk, index row, chunk placeholder."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from wenmai.app import create_app
from wenmai.config import Settings
from wenmai.factories import vector_store as vector_store_factory


def _write_pdf_with_embedded_image(path: Path) -> Path:
    """Build a one-page PDF: ASCII text plus a small embedded raster image."""
    image_path = path.with_suffix(".png")
    Image.new("RGB", (40, 40), color=(180, 40, 40)).save(image_path)

    pdf = canvas.Canvas(str(path), pagesize=letter)
    pdf.drawString(72, 720, "Fuzhou shipyard heritage photo below.")
    pdf.drawImage(str(image_path), 72, 580, width=120, height=120)
    pdf.save()
    return path


def test_ingesting_pdf_extracts_image_writes_index_and_chunk_placeholder(
    test_settings: Settings, tmp_path: Path
) -> None:
    source = _write_pdf_with_embedded_image(tmp_path / "shipyard.pdf")
    client = TestClient(create_app(test_settings))

    response = client.post("/ingest", json={"source_path": str(source)})

    assert response.status_code == 200
    body = response.json()
    assert body["chunk_count"] >= 1

    store = vector_store_factory.create(test_settings)
    chunks = store.get_by_document_id(body["document_id"])
    combined = "\n".join(chunk.text for chunk in chunks)
    placeholder_match = re.search(r"\[IMAGE: ([a-f0-9]+)\]", combined)
    assert placeholder_match is not None, combined

    image_id = placeholder_match.group(1)
    collection = test_settings.product.collection
    images_root = Path(test_settings.paths.images)
    image_files = list((images_root / collection).glob(f"{image_id}.*"))
    assert len(image_files) == 1
    assert image_files[0].stat().st_size > 0

    conn = sqlite3.connect(test_settings.paths.image_index)
    try:
        row = conn.execute(
            """
            SELECT image_id, document_id, source_path, page, file_path
            FROM images
            WHERE image_id = ?
            """,
            (image_id,),
        ).fetchone()
    finally:
        conn.close()

    assert row is not None
    assert row[0] == image_id
    assert row[1] == body["document_id"]
    assert row[2] == str(source)
    assert row[3] == 1
    assert Path(row[4]).is_file()
