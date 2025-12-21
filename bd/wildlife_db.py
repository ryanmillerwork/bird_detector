from __future__ import annotations

"""
Helpers for inserting detections into the `wildlife` Postgres table.

Schema reference (provided by user):
    CREATE TABLE IF NOT EXISTS wildlife (
        id SERIAL PRIMARY KEY,
        detected_at TIMESTAMPTZ NOT NULL,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        image_path TEXT NOT NULL,
        detector_model TEXT,
        detector_label TEXT,
        detector_confidence REAL,
        classifier_model TEXT,
        classifier_label TEXT,
        classifier_confidence REAL,
        bbox_x1 INTEGER,
        bbox_y1 INTEGER,
        bbox_x2 INTEGER,
        bbox_y2 INTEGER,
        video_source TEXT,
        reviewed BOOLEAN DEFAULT FALSE,
        corrected_label TEXT
    );
"""

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping


class UnsafeIdentifierError(ValueError):
    pass


def _quote_ident(name: str) -> str:
    """
    Very small identifier sanitizer for table names.
    Allows [A-Za-z0-9_] only.
    """
    n = str(name).strip()
    if not n:
        raise UnsafeIdentifierError("Empty identifier")
    if not all(c.isalnum() or c == "_" for c in n):
        raise UnsafeIdentifierError(f"Unsafe identifier: {name!r}")
    return f'"{n}"'


def wildlife_table_from_env() -> str:
    return os.environ.get("WILDLIFE_TABLE", "wildlife").strip() or "wildlife"


@dataclass(frozen=True)
class WildlifeRow:
    detected_at: datetime
    image_path: str
    detector_model: str | None
    detector_label: str | None
    detector_confidence: float | None
    classifier_model: str | None
    classifier_label: str | None
    classifier_confidence: float | None
    bbox_x1: int | None
    bbox_y1: int | None
    bbox_x2: int | None
    bbox_y2: int | None
    video_source: str | None


def create_wildlife_table(conn, *, table: str | None = None) -> None:
    t = _quote_ident(table or wildlife_table_from_env())
    sql = f"""
    CREATE TABLE IF NOT EXISTS {t} (
        id SERIAL PRIMARY KEY,
        detected_at TIMESTAMPTZ NOT NULL,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        image_path TEXT NOT NULL,

        detector_model TEXT,
        detector_label TEXT,
        detector_confidence REAL,

        classifier_model TEXT,
        classifier_label TEXT,
        classifier_confidence REAL,

        bbox_x1 INTEGER,
        bbox_y1 INTEGER,
        bbox_x2 INTEGER,
        bbox_y2 INTEGER,

        video_source TEXT,

        reviewed BOOLEAN DEFAULT FALSE,
        corrected_label TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_wildlife_detected_at ON {t}(detected_at);
    CREATE INDEX IF NOT EXISTS idx_wildlife_classifier_label ON {t}(classifier_label);
    CREATE INDEX IF NOT EXISTS idx_wildlife_reviewed ON {t}(reviewed);
    """
    cur = conn.cursor()
    cur.execute(sql)


def insert_wildlife(conn, row: WildlifeRow, *, table: str | None = None) -> None:
    t = _quote_ident(table or wildlife_table_from_env())
    sql = f"""
        INSERT INTO {t} (
            detected_at,
            image_path,
            detector_model,
            detector_label,
            detector_confidence,
            classifier_model,
            classifier_label,
            classifier_confidence,
            bbox_x1,
            bbox_y1,
            bbox_x2,
            bbox_y2,
            video_source
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        );
    """
    cur = conn.cursor()
    cur.execute(
        sql,
        (
            row.detected_at,
            row.image_path,
            row.detector_model,
            row.detector_label,
            row.detector_confidence,
            row.classifier_model,
            row.classifier_label,
            row.classifier_confidence,
            row.bbox_x1,
            row.bbox_y1,
            row.bbox_x2,
            row.bbox_y2,
            row.video_source,
        ),
    )


def row_from_detection(
    *,
    detected_at: datetime,
    image_path: str,
    det: Mapping[str, Any],
    detector_model: str | None,
    classifier_model: str | None,
    video_source: str | None,
) -> WildlifeRow:
    bbox = det.get("bbox") or (None, None, None, None)
    try:
        x1, y1, x2, y2 = bbox
    except Exception:
        x1 = y1 = x2 = y2 = None

    detector_label = det.get("class")
    detector_confidence = det.get("confidence")

    classifier_label = det.get("species")
    classifier_confidence = det.get("species_confidence")

    return WildlifeRow(
        detected_at=detected_at,
        image_path=str(image_path),
        detector_model=detector_model,
        detector_label=str(detector_label) if detector_label is not None else None,
        detector_confidence=float(detector_confidence) if detector_confidence is not None else None,
        classifier_model=classifier_model,
        classifier_label=str(classifier_label) if classifier_label else None,
        classifier_confidence=float(classifier_confidence) if classifier_confidence is not None else None,
        bbox_x1=int(x1) if x1 is not None else None,
        bbox_y1=int(y1) if y1 is not None else None,
        bbox_x2=int(x2) if x2 is not None else None,
        bbox_y2=int(y2) if y2 is not None else None,
        video_source=video_source,
    )





