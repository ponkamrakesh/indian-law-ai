"""
RAG Evaluation Module using RAGAS + MLflow.
Run evaluations on test queries for precise metrics: faithfulness, answer_relevancy, context_precision, context_recall.
Completely free (RAGAS open source).
Integrates with LangGraph agent.
"""
import mlflow
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from datasets import Dataset
from app.services.langgraph_agent import process_legal_query
from app.config import settings
from typing import List, Dict
import pandas as pd

def get_test_dataset() -> Dataset:
    """Sample test queries for Indian Law RAG evaluation. Expand with real cases."""
    data = {
        "question": [
            "What is the punishment for murder under IPC?",
            "Explain Section 302 IPC in simple terms.",
            "What should I do if someone threatens me with violence in India?",
            "What are the rights of an accused in CrPC?"
        ],
        "ground_truth": [
            "Section 302 IPC prescribes death or life imprisonment for murder.",
            "IPC 302: Whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine.",
            "File a complaint with police under relevant sections like 503, 506 IPC. Seek protection order if needed.",
            "Right to legal aid, right to remain silent, right to fair trial, etc."
        ]
    }
    return Dataset.from_dict(data)

def run_ragas_evaluation(test_queries: List[str] = None):
    """Run full RAGAS evaluation and log to MLflow."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("IndianLawAI_RAG_Evaluation")
    
    llm = ChatGroq(groq_api_key=settings.groq_api_key, model_name=settings.llm_model)
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    
    if test_queries is None:
        test_dataset = get_test_dataset()
    else:
        # For custom queries
        test_dataset = Dataset.from_dict({"question": test_queries})
    
    # Generate answers and contexts using our agent
    results = []
    for q in test_dataset["question"]:
        agent_result = process_legal_query(q)
        results.append({
            "question": q,
            "answer": agent_result["response"],
            "contexts": [d["content"] for d in agent_result["sources"]],
            "ground_truth": test_dataset["ground_truth"][test_dataset["question"].index(q)] if "ground_truth" in test_dataset.column_names else ""
        })
    
    eval_dataset = Dataset.from_list(results)
    
    # Run RAGAS
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    evaluation_result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings
    )
    
    # Log to MLflow
    with mlflow.start_run(run_name=f"ragas_eval_{datetime.now().strftime('%Y%m%d_%H%M')}"):
        for metric_name, score in evaluation_result.items():
            mlflow.log_metric(metric_name, float(score))
        mlflow.log_param("llm_model", settings.llm_model)
        mlflow.log_param("embedding_model", settings.embedding_model)
        mlflow.log_param("top_k", settings.top_k)
        
        # Log detailed results as artifact
        df = evaluation_result.to_pandas()
        df.to_csv("ragas_results.csv", index=False)
        mlflow.log_artifact("ragas_results.csv")
    
    print("RAGAS Evaluation Complete. Metrics:")
    print(evaluation_result)
    return evaluation_result

if __name__ == "__main__":
    from datetime import datetime
    print("Running RAG Evaluation for IndianLawAI...")
    run_ragas_evaluation()