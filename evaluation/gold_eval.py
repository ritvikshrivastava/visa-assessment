import json
from typing import Dict

import requests


def load_gold_set(file_path: str):
    with open(file_path, "r") as file:
        return json.load(file)


def assess_cv(cv_text: str):
    response = requests.post(
        "http://localhost:8000/assess_visa",
        files={"cv": ("resume.txt", cv_text, "text/plain")},
        data={"use_azure": False},
    )
    return response.json()


def win_loss_analysis(predictions: Dict[str, str], gold_set: Dict[str, str]):
    win = 0
    loss = 0

    for i, (pred_label, gold_label) in enumerate(
        zip(predictions.values(), gold_set.values())
    ):
        if pred_label == gold_label:
            win += 1
        else:
            loss += 1
    return win, loss


def main():
    gold_set = load_gold_set("gold_set.json")

    predictions = {}
    for i, resume in enumerate(gold_set):
        cv_text = resume["cv_text"]
        result = assess_cv(cv_text)
        pred_label = result["eligibility_label"]
        predictions[f"resume_{i + 1}"] = pred_label

    gold_labels = {
        f"resume_{i + 1}": resume["label"] for i, resume in enumerate(gold_set)
    }

    win, loss = win_loss_analysis(predictions, gold_labels)
    print(f"Win: {win}, Loss: {loss}")


if __name__ == "__main__":
    main()
