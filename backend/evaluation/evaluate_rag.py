"""
RAG Evaluation Module using RAGAS + MLflow.
"""
import mlflow
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import Dataset
from app.services.langgraph_agent import process_legal_query
from app.config import settings
from typing import List
from datetime import datetime

def get_test_dataset():
    data = {
        "question": [
            "What is the punishment for murder under IPC?",
            "Explain Section 302 IPC in simple terms."
        ],
        "ground_truth": [
            "Section 302 IPC prescribes death or life imprisonment for murder.",
            "IPC 302: Whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine."
        ]
    }
    return Dataset.from_dict(data)

def run_ragas_evaluation():
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("IndianLawAI_RAG_Evaluation")
    
    llm = ChatGroq(groq_api_key=settings.groq_api_key, model_name=settings.llm_model)
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    
    test_dataset = get_test_dataset()
    results = []
    
    for q in test_dataset["question"]:
        agent_result = process_legal_query(q)
        results.append({
            "question": q,
            "answer": agent_result["response"],
            "contexts": [d["content"] for d in agent_result["sources"]],
            "ground_truth": test_dataset["ground_truth"][test_dataset["question"].index(q)]
        })
    
    eval_dataset = Dataset.from_list(results)
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    
    evaluation_result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings
    )
    
    with mlflow.start_run(run_name=f"ragas_eval_{datetime.now().strftime('%Y%m%d_%H%M')}"):
        for metric_name, score in evaluation_result.items():
            mlflow.log_metric(metric_name, float(score))
    
    print("RAGAS Evaluation Complete.")
    print(evaluation_result)
    return evaluation_result

if __name__ == "__main__":
    run_ragas_evaluation()