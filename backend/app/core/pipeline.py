"""Orchestrates the full document processing pipeline and updates
processing_status at each real stage, so the frontend's progress UI
reflects actual backend state (per spec section 8: no fake progress).
"""
from sqlalchemy.orm import Session

from app.models.models import Document, DocumentPage, DocumentBlock, Chunk, ChunkParent
from app.nlp import extraction, statistics, ner, pos, keywords
from app.nlp.spacy_model import parse as spacy_parse
from app.retrieval import chunking, embeddings, vector_store, lexical
from app.core.config import get_settings

settings = get_settings()

# in-memory cache of BM25 indexes per document, rebuilt on process start
_bm25_cache: dict[str, object] = {}


def get_bm25(document_id: str, chunk_texts: list[str]):
    if document_id not in _bm25_cache:
        _bm25_cache[document_id] = lexical.build_bm25(chunk_texts)
    return _bm25_cache[document_id]


def invalidate_bm25(document_id: str):
    _bm25_cache.pop(document_id, None)


def process_document(db: Session, document: Document):
    try:
        # 1. Extract (layout-aware: blocks, headings, tables, bounding boxes)
        document.processing_status = "extracting"
        db.commit()
        extracted = extraction.extract_document(document.storage_path, document.file_type)
        pages = extracted.pages

        for page_num, text in pages:
            db.add(DocumentPage(document_id=document.id, page_number=page_num, text=text))
        for i, b in enumerate(extracted.blocks):
            db.add(DocumentBlock(
                document_id=document.id, block_index=i, page_number=b.page, kind=b.kind, level=b.level,
                text=b.text, bbox=b.bbox, font_size=b.font_size, section_path=b.section_path,
                table_json=b.table, char_start=b.char_start, char_end=b.char_end,
            ))
        document.title = extracted.title
        document.toc_json = extracted.toc
        document.structure_json = {
            "page_sizes": {str(k): v for k, v in extracted.page_sizes.items()},
            "ocr_pages": extracted.ocr_pages,
            "notes": extracted.notes,
            "block_counts": {k: sum(1 for b in extracted.blocks if b.kind == k) for k in ("heading", "paragraph", "table")},
        }
        db.commit()

        full_text = "\n\n".join(text for _, text in pages)
        document.page_count = len(pages)

        # 2. NLP analysis
        document.processing_status = "analyzing"
        db.commit()

        stats = statistics.compute_statistics(full_text)
        document.word_count = stats["word_count"]
        document.sentence_count = stats["sentence_count"]
        document.paragraph_count = stats["paragraph_count"]
        document.stats_json = stats
        # NER and POS both run over the same spaCy pipeline — parse once and
        # share the Doc between them instead of parsing the document twice.
        spacy_doc = spacy_parse(full_text)
        document.entities_json = ner.extract_entities_from_doc(spacy_doc)
        document.pos_json = pos.compute_pos_distribution_from_doc(spacy_doc)

        document.keywords_json = {
            "tfidf": keywords.extract_keywords_tfidf(full_text),
            "textrank": keywords.extract_keywords_textrank(full_text),
            "keybert": keywords.extract_keywords_keybert(full_text),
        }

        db.commit()

        # 3. Chunk: structure-aware children + section-level parents
        parent_dicts, chunk_dicts = chunking.build_chunks(extracted.blocks, extracted.title or document.name)
        if not chunk_dicts:
            raise ValueError("Document produced no usable text chunks.")

        parent_rows = []
        for i, p in enumerate(parent_dicts):
            row = ChunkParent(
                document_id=document.id, parent_index=i, section_path=p["section_path"],
                page_start=p["page_start"], page_end=p["page_end"], text=p["text"],
            )
            db.add(row)
            parent_rows.append(row)
        db.flush()

        chunk_rows = []
        for i, c in enumerate(chunk_dicts):
            row = Chunk(
                document_id=document.id,
                page_number=c["page"],
                page_end=c["page_end"],
                section=c["section"],
                section_path=c["section_path"],
                chunk_index=c["chunk_index"],
                kind=c["kind"],
                text=c["text"],
                embed_text=c["embed_text"],
                char_start=c["char_start"],
                char_end=c["char_end"],
                spans_json=c["spans"],
                parent_id=parent_rows[c["parent_idx"]].id,
                vector_row=i,
            )
            db.add(row)
            chunk_rows.append(row)
        db.commit()

        # 4. Embeddings
        document.processing_status = "embedding"
        db.commit()
        chunk_texts = [c["embed_text"] for c in chunk_dicts]
        vectors = embeddings.embed_texts(chunk_texts)

        # 5. Vector index
        document.processing_status = "indexing"
        db.commit()
        vector_store.build_index(document.id, vectors)
        get_bm25(document.id, chunk_texts)  # warm the BM25 cache

        document.processing_status = "ready"
        db.commit()

    except Exception as exc:
        document.processing_status = "failed"
        document.error_message = str(exc)
        db.commit()
        raise
