"""
Background ingestion worker — processes PDF upload jobs from the Redis queue.

Run via:  rq worker --url redis://redis:6379
"""

import os
import time
import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from qdrant_client.models import Filter, FieldCondition, MatchValue

from config import (
    qdrant_client,
    QDRANT_COLLECTION,
    MAX_RETRIES,
    get_vector_store,
)
from database import SessionLocal
from models import IngestionJob, JobStatus
from redis_client import get_job_queue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_ingestion(job_id: str, file_path: str, filename: str):
    """
    Main ingestion task — extracted from the original /upload endpoint.

    Idempotent: deletes any existing vectors for this document_id before
    inserting, so retries never produce duplicates.
    """
    db = SessionLocal()
    try:
        # ── 1. Mark job as PROCESSING ─────────────────────────────
        job = db.query(IngestionJob).filter(IngestionJob.id == job_id).first()
        if not job:
            logger.error(f"Job {job_id} not found in database")
            return

        job.status = JobStatus.PROCESSING
        db.commit()
        logger.info(f"[{job_id}] Processing {filename}...")

        # ── 2. OCR conversion via Docling (same as original code) ─
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )

        doc_converted = converter.convert(file_path)
        md_content = doc_converted.document.export_to_markdown()

        # ── 3. Chunk with RecursiveCharacterTextSplitter ──────────
        langchain_doc = Document(
            page_content=md_content,
            metadata={"source": filename, "document_id": job_id},
        )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
        )
        splits = text_splitter.split_documents([langchain_doc])

        # Ensure every chunk carries the document_id
        for doc in splits:
            doc.metadata["document_id"] = job_id

        # ── 4. Idempotency: delete existing vectors for this job ──
        try:
            qdrant_client.delete(
                collection_name=QDRANT_COLLECTION,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.document_id",
                            match=MatchValue(value=job_id),
                        )
                    ]
                ),
            )
            logger.info(f"[{job_id}] Cleared any previous vectors")
        except Exception:
            # Collection may be empty / first run — safe to ignore
            pass

        # ── 5. Add documents to Qdrant ────────────────────────────
        vector_store = get_vector_store()
        vector_store.add_documents(splits)
        logger.info(f"[{job_id}] Added {len(splits)} chunks to Qdrant")

        # ── 6. Cleanup temp file ──────────────────────────────────
        if os.path.exists(file_path):
            os.remove(file_path)

        # ── 7. Mark job COMPLETED ─────────────────────────────────
        job.status = JobStatus.COMPLETED
        db.commit()
        logger.info(f"[{job_id}] ✅ Ingestion complete")

    except Exception as e:
        db.rollback()
        logger.error(f"[{job_id}] Ingestion failed: {e}")

        # Retry logic
        job = db.query(IngestionJob).filter(IngestionJob.id == job_id).first()
        if job:
            job.retry_count += 1
            if job.retry_count < MAX_RETRIES:
                job.status = JobStatus.PENDING
                db.commit()

                # Re-enqueue with exponential backoff
                delay = 2 ** job.retry_count  # 2s, 4s, 8s
                logger.info(
                    f"[{job_id}] Retry {job.retry_count}/{MAX_RETRIES} "
                    f"in {delay}s"
                )
                time.sleep(delay)
                queue = get_job_queue()
                queue.enqueue(
                    process_ingestion, job_id, file_path, filename
                )
            else:
                job.status = JobStatus.FAILED
                job.error_message = str(e)[:500]
                db.commit()
                logger.error(f"[{job_id}] ❌ Max retries exceeded")
    finally:
        db.close()
