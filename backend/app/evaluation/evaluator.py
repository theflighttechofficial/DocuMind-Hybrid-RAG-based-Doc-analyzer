"""Runs an evaluation dataset against the document's live retrieval + QA
pipeline and computes real metrics. Nothing here is precomputed/faked.
"""
from sqlalchemy.orm import Session

from app.models.models import Document, Chunk, Evaluation
from app.core.pipeline import get_bm25
from app.retrieval.hybrid import retrieve
from app.qa.generator import answer_question
from app.evaluation import metrics as m

_METHODS = ["bm25", "semantic", "hybrid"]


def run_evaluation(db: Session, document_id: str, eval_set: list[dict], compare_rerank: bool = False) -> dict:
    """eval_set: list of {"question": str, "answer": str (ground truth), "page": int (optional)}.

    compare_rerank: when True, also runs hybrid retrieval through the cross-encoder
    reranker under a "hybrid_reranked" row, so reranked vs non-reranked metrics can
    be compared side by side in the same run.
    """
    doc = db.get(Document, document_id)
    if not doc or doc.processing_status != "ready":
        raise ValueError("Document not found or not ready.")

    chunks = db.query(Chunk).filter_by(document_id=document_id).order_by(Chunk.chunk_index).all()
    chunk_texts = [c.text for c in chunks]
    bm25 = get_bm25(document_id, chunk_texts)

    methods = _METHODS + (["hybrid_reranked"] if compare_rerank else [])
    retrieval_results = {method: {"precision": [], "recall": [], "mrr": []} for method in methods}
    qa_results = {"exact_match": [], "f1": [], "semantic_similarity": [], "bertscore_f1": []}
    summarization_results = {"rouge1": [], "rouge2": [], "rougeL": []}

    for item in eval_set:
        question = item["question"]
        relevant_pages = {item["page"]} if item.get("page") else set()

        for method in methods:
            actual_method = "hybrid" if method == "hybrid_reranked" else method
            evidence = retrieve(
                document_id, question, chunks, bm25, method=actual_method, top_k=5,
                rerank=(method == "hybrid_reranked"),
            )
            retrieved_pages = [e["chunk"].page_number for e in evidence]
            if relevant_pages:
                retrieval_results[method]["precision"].append(m.precision_at_k(retrieved_pages, relevant_pages, 5))
                retrieval_results[method]["recall"].append(m.recall_at_k(retrieved_pages, relevant_pages, 5))
                retrieval_results[method]["mrr"].append(m.mean_reciprocal_rank(retrieved_pages, relevant_pages))

        if item.get("answer"):
            evidence = retrieve(document_id, question, chunks, bm25, method="hybrid", top_k=5)
            result = answer_question(question, evidence)
            prediction = result["answer"]
            ground_truth = item["answer"]

            qa_results["exact_match"].append(m.exact_match(prediction, ground_truth))
            qa_results["f1"].append(m.token_f1(prediction, ground_truth))
            qa_results["semantic_similarity"].append(m.semantic_similarity(prediction, ground_truth))
            qa_results["bertscore_f1"].append(m.bertscore_f1(prediction, ground_truth))

            if item.get("is_summary_task"):
                summarization_results["rouge1"].append(m.rouge_n(prediction, ground_truth, 1))
                summarization_results["rouge2"].append(m.rouge_n(prediction, ground_truth, 2))
                summarization_results["rougeL"].append(m.rouge_l(prediction, ground_truth))

            db.add(Evaluation(
                document_id=document_id, question=question, ground_truth=ground_truth,
                generated_answer=prediction, method="hybrid",
                metrics_json={"f1": qa_results["f1"][-1], "exact_match": qa_results["exact_match"][-1]},
            ))

    db.commit()

    def _avg(values):
        return round(sum(values) / len(values), 4) if values else None

    return {
        "sample_size": len(eval_set),
        "retrieval": {
            method: {k: _avg(v) for k, v in stats.items()}
            for method, stats in retrieval_results.items()
        },
        "qa": {k: _avg(v) for k, v in qa_results.items()},
        "summarization": {k: _avg(v) for k, v in summarization_results.items()} if any(summarization_results.values()) else None,
    }
