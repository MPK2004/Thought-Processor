import os
import time

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
from logger import get_logger

log = get_logger("worker")


def process_ingestion(job_id: str, file_path: str, filename: str):
    """
    Main ingestion task. Idempotent: deletes existing vectors for this
    document_id before inserting, so retries never produce duplicates.
    """
    db = SessionLocal()
    try:
        job = db.query(IngestionJob).filter(IngestionJob.id == job_id).first()
        if not job:
            log.error(f"Job {job_id} not found in database")
            return

        job.status = JobStatus.PROCESSING
        db.commit()
        log.info(f"[{job_id}] PENDING -> PROCESSING | file={filename}")

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
        log.info(f"[{job_id}] OCR complete, markdown_length={len(md_content)}")

        langchain_doc = Document(
            page_content=md_content,
            metadata={"source": filename, "document_id": job_id},
        )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
        )
        splits = text_splitter.split_documents([langchain_doc])

        for doc in splits:
            doc.metadata["document_id"] = job_id

        log.info(f"[{job_id}] Chunked into {len(splits)} chunks")

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
            log.info(f"[{job_id}] Cleared previous vectors")
        except Exception:
            pass

        vector_store = get_vector_store()
        vector_store.add_documents(splits)
        log.info(f"[{job_id}] Added {len(splits)} chunks to Qdrant")

        if os.path.exists(file_path):
            os.remove(file_path)

        job.status = JobStatus.COMPLETED
        db.commit()
        log.info(f"[{job_id}] PROCESSING -> COMPLETED")

    except Exception as e:
        db.rollback()
        log.error(f"[{job_id}] Ingestion failed: {e}")

        job = db.query(IngestionJob).filter(IngestionJob.id == job_id).first()
        if job:
            job.retry_count += 1
            if job.retry_count < MAX_RETRIES:
                job.status = JobStatus.PENDING
                db.commit()

                delay = 2 ** job.retry_count
                log.info(
                    f"[{job_id}] FAILED -> PENDING (retry "
                    f"{job.retry_count}/{MAX_RETRIES} in {delay}s)"
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
                log.error(
                    f"[{job_id}] PROCESSING -> FAILED | "
                    f"max retries exceeded"
                )
    finally:
        db.close()
