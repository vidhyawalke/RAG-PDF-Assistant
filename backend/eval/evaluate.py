"""
================================================================================
Automated RAG Evaluation & Latency Benchmark Suite
--------------------------------------------------------------------------------
References & Documentation Sources:
- RAG Evaluation & Benchmarking Survey Methodology: https://arxiv.org/abs/2312.10997
- Python Unit Testing & Automated Metrics Reporting: https://docs.python.org/3/library/unittest.html
- Python JSON File Handling: https://docs.python.org/3/library/json.html
================================================================================
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

from backend.rag_chain import rag_pipeline
from backend.config import settings

# Configure evaluation logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluate")


def create_sample_pdf(output_path: str) -> str:
    """
    Generate a valid sample PDF for automated evaluation tests.
    Source: PDF 1.4 Binary Specification (https://opensource.adobe.com/dc-acrobat-sdk-docs/)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    content_lines = [
        "Junior AI Software Development Engineer (SDE) - Specification",
        "1. Role Overview & Objectives:",
        "The primary role of a Junior AI Software Development Engineer is to assist in turning AI prototypes into production-ready software features.",
        "Engineers must write clean, maintainable code in Python or other core languages, monitor model performance, and debug integration issues.",
        "",
        "2. Key Responsibilities & Data Preparation:",
        "Data Preparation: Clean, transform, and manage datasets for training or RAG (Retrieval-Augmented Generation) pipelines.",
        "Model Integration: Connect LLMs, vector databases (such as ChromaDB or FAISS), or machine learning APIs to backend applications.",
        "Testing & Evaluation: Run evaluation scripts to check model accuracy, latency, and response quality.",
        "",
        "3. Technical Stack & Requirements:",
        "Programming Languages: Solid foundation in Python and SQL; familiarity with JavaScript and Node.js is a plus.",
        "AI/ML Stack: Exposure to frameworks like PyTorch, TensorFlow, Scikit-learn, LangChain, or LlamaIndex.",
        "Cloud & Tools: Experience with Git version control and basic cloud services (AWS, GCP, or Azure)."
    ]

    pdf_content = (
        "%PDF-1.4\n"
        "1 0 obj <</Type /Catalog /Pages 2 0 R>> endobj\n"
        "2 0 obj <</Type /Pages /Kids [3 0 R] /Count 1>> endobj\n"
        "3 0 obj <</Type /Page /Parent 2 0 R /Resources <</Font <</F1 4 0 R>>>> /MediaBox [0 0 612 792] /Contents 5 0 R>> endobj\n"
        "4 0 obj <</Type /Font /Subtype /Type1 /BaseFont /Helvetica>> endobj\n"
        "5 0 obj <</Length 650>> stream\n"
        "BT /F1 12 Tf 40 750 Td 14 TL\n"
    )
    
    for line in content_lines:
        safe_line = line.replace("(", "\\(").replace(")", "\\)")
        pdf_content += f"({safe_line}) '\n"
        
    pdf_content += (
        "ET\n"
        "endstream\n"
        "endobj\n"
        "xref\n"
        "0 6\n"
        "0000000000 65535 f \n"
        "0000000009 00000 n \n"
        "0000000058 00000 n \n"
        "0000000115 00000 n \n"
        "0000000230 00000 n \n"
        "0000000300 00000 n \n"
        "trailer <</Size 6 /Root 1 0 R>>\n"
        "startxref\n"
        "1000\n"
        "%%EOF"
    )

    with open(output_path, "wb") as f:
        f.write(pdf_content.encode("latin1"))

    logger.info(f"Generated sample evaluation PDF: {output_path}")
    return output_path


# Ground Truth Benchmark Test Suite
# Source Pattern: Evaluation Benchmark Dataset Pattern
BENCHMARK_QUESTIONS = [
    {
        "id": 1,
        "question": "What is the primary role of a Junior AI SDE?",
        "expected_keywords": ["assist", "turning AI prototypes", "production-ready"],
        "category": "Role Overview"
    },
    {
        "id": 2,
        "question": "What are the core responsibilities in Data Preparation?",
        "expected_keywords": ["clean", "transform", "manage datasets", "RAG"],
        "category": "Data Preparation"
    },
    {
        "id": 3,
        "question": "Which frameworks and vector databases are mentioned in the ML stack?",
        "expected_keywords": ["ChromaDB", "FAISS", "LangChain", "PyTorch"],
        "category": "Model Integration"
    },
    {
        "id": 4,
        "question": "What evaluation metrics should be checked according to the document?",
        "expected_keywords": ["accuracy", "latency", "response quality"],
        "category": "Testing & Evaluation"
    },
    {
        "id": 5,
        "question": "What programming languages are required?",
        "expected_keywords": ["Python", "SQL", "JavaScript"],
        "category": "Requirements"
    }
]


def run_evaluation():
    """
    Run automated evaluation benchmarking precision, latency, and context quality.
    Source: https://arxiv.org/abs/2312.10997
    """
    logger.info("=== Starting PDF Q&A Bot Evaluation Suite ===")
    
    sample_pdf_path = os.path.join(settings.UPLOAD_DIR, "eval_sample_spec.pdf")
    create_sample_pdf(sample_pdf_path)
    
    ingest_result = rag_pipeline.process_pdf(sample_pdf_path)
    logger.info(f"Ingested {ingest_result['total_chunks']} chunks in {ingest_result['ingestion_time_ms']} ms.")
    
    results = []
    total_latency = 0.0
    passed_count = 0

    print("\n" + "="*85)
    print(f" {'ID':<3} | {'Category':<20} | {'Latency (ms)':<12} | {'Retrieval Score':<15} | {'Status':<8}")
    print("="*85)

    for q in BENCHMARK_QUESTIONS:
        res = rag_pipeline.answer_question(q["question"], top_k=3)
        answer = res["answer"].lower()
        latency = res["execution_time_ms"]
        total_latency += latency
        
        # Calculate retrieval & ground-truth keyword precision
        matched_keywords = [kw for kw in q["expected_keywords"] if kw.lower() in answer or any(kw.lower() in s["content"].lower() for s in res["sources"])]
        retrieval_precision = len(matched_keywords) / len(q["expected_keywords"]) if q["expected_keywords"] else 1.0
        
        status_flag = "PASS" if retrieval_precision >= 0.5 else "WARN"
        if status_flag == "PASS":
            passed_count += 1

        results.append({
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "answer": res["answer"],
            "sources_retrieved": len(res["sources"]),
            "top_page": res["sources"][0]["page"] if res["sources"] else "N/A",
            "latency_ms": latency,
            "matched_keywords": matched_keywords,
            "precision_score": round(retrieval_precision * 100, 1),
            "status": status_flag
        })

        print(f" {q['id']:<3} | {q['category']:<20} | {latency:<12.2f} | {retrieval_precision*100:<14.1f}% | {status_flag:<8}")

    print("="*85)
    avg_latency = round(total_latency / len(BENCHMARK_QUESTIONS), 2)
    pass_rate = round((passed_count / len(BENCHMARK_QUESTIONS)) * 100, 1)

    print(f"\nEvaluation Summary:")
    print(f" Total Questions Tested : {len(BENCHMARK_QUESTIONS)}")
    print(f" Pass Rate              : {pass_rate}% ({passed_count}/{len(BENCHMARK_QUESTIONS)})")
    print(f" Average Latency        : {avg_latency} ms")
    print("="*85 + "\n")

    # Output Markdown and JSON Reports
    eval_dir = Path(__file__).parent
    report_md_path = eval_dir / "EVALUATION_REPORT.md"
    json_path = eval_dir / "eval_results.json"

    with open(json_path, "w") as f:
        json.dump({
            "summary": {
                "total_questions": len(BENCHMARK_QUESTIONS),
                "pass_rate_pct": pass_rate,
                "avg_latency_ms": avg_latency
            },
            "results": results
        }, f, indent=2)

    md_report = f"""# PDF Q&A Bot - Automated Evaluation Benchmark Report

## Overview & Performance Summary
- **Total Test Cases**: {len(BENCHMARK_QUESTIONS)}
- **Pass Rate**: {pass_rate}%
- **Average End-to-End Latency**: {avg_latency} ms
- **Ingestion Time**: {ingest_result['ingestion_time_ms']} ms ({ingest_result['total_chunks']} chunks)

## Detailed Test Results Matrix

| ID | Category | Question | Ret. Precision | Latency (ms) | Status | Top Source |
|---|---|---|---|---|---|---|
"""
    for r in results:
        md_report += f"| {r['id']} | {r['category']} | {r['question']} | {r['precision_score']}% | {r['latency_ms']} | **{r['status']}** | Page {r['top_page']} |\n"

    md_report += "\n## Sample Questions & Model Responses\n\n"
    for r in results:
        md_report += f"### Q{r['id']}: {r['question']}\n"
        md_report += f"- **Status**: `{r['status']}` ({r['precision_score']}% keyword match)\n"
        md_report += f"- **Answer**: {r['answer']}\n"
        md_report += f"- **Latency**: {r['latency_ms']} ms\n\n"

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(md_report)

    logger.info(f"Written benchmark report to: {report_md_path}")
    return results


if __name__ == "__main__":
    run_evaluation()
