import asyncio
import os
from typing import Dict, List

import numpy as np
from openai import AzureOpenAI, OpenAI
from transformers import AutoModelForSequenceClassification, AutoTokenizer

nli_model_name = "facebook/bart-large-mnli"
tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)


def chunk_text(text: str, chunk_size: int = 100) -> List[str]:
    words = text.split()
    return [
        " ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)
    ]


async def find_best_chunk(cv_chunks: List[str], criterion: str, azure: bool) -> str:
    prompt = f"Find the best matching chunk in the CV for the following criterion: {criterion}. CV Chunks: {cv_chunks}"
    if azure:
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2023-12-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
    else:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    best_chunk = response.choices[0].message.content.strip()
    return best_chunk


async def evaluate_nli(
    cv_text: str, extracted_info: Dict[str, List[str]]
) -> Dict[str, float]:
    cv_chunks = chunk_text(cv_text)
    scores = {}

    for criterion, evidences in extracted_info.items():
        best_chunk = await find_best_chunk(cv_chunks, criterion)
        entailment_results = []

        for evidence in evidences:
            inputs = tokenizer(f"{best_chunk} </s> {evidence}", return_tensors="pt")
            outputs = nli_model(**inputs)
            logits = outputs.logits.detach().numpy()
            predicted_label = np.argmax(logits, axis=1)[0]

            is_entailment = predicted_label == 2  # Index 2 is entailment in the MNLI model
            entailment_results.append(is_entailment)

        scores[criterion] = entailment_results
    return scores


def calculate_overall_entailment(scores: Dict[str, List[bool]]) -> float:
    total_entailments = sum(
        [sum(results) for results in scores.values()]
    )  # Sum of True events for entailment
    total_checks = sum(
        [len(results) for results in scores.values()]
    )  # Number of all events
    if total_checks == 0:
        return 0.0
    return (total_entailments / total_checks) * 100


def main(cv_text: str, extracted_info: Dict[str, List[str]]):
    """
    :param cv_text: extracted text data from the file
    :param extracted_info: responses from the request made to the visa assessment statement
    :return: NLI scores, overall NLI percentage
    """
    scores = asyncio.run(evaluate_nli(cv_text, extracted_info))
    overall_entailment_percentage = calculate_overall_entailment(scores)
    return scores, overall_entailment_percentage
