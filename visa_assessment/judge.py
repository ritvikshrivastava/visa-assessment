import json, os

from openai import AzureOpenAI, OpenAI

from visa_assessment.eligibility import CRITERIA


async def judge_eligibility(cv_text: str, azure=True):
    if azure:
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2023-12-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
    else:
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are an expert in US immigration law, and you specialize in O-1A visa eligibility assessment.",
            },
            {"role": "user", "content": create_prompt(cv_text)},
        ],
        max_tokens=1024,
        temperature=0.5,
    )

    response_text = response.choices[0].message.content
    results = parse_llm_response(response_text)
    rating = get_rating(results)

    return {"results": results, "rating": rating}


def create_prompt(cv_text: str):
    criteria_details = "\n\n".join(
        [
            f"{key}: {value['description']}\nEvidence Types: {', '.join(value['evidence_types'])}"
            for key, value in CRITERIA.items()
        ]
    )
    criteria_keys = ", ".join(CRITERIA.keys())

    return (
        f"Analyze the following CV text to match against the O-1A visa criteria:\n\n"
        f"CV Text:\n{cv_text}\n\n"
        f"Identify and list the evidence related to each of the following criteria:\n\n{criteria_details}\n\n"
        "Return a JSON response structured as follows:\n"
        "Each key should be one of the following: "
        f"{criteria_keys}\n"
        "The value for each key should be the evidence supporting that criterion.\n"
        "Do not return anything else apart from the JSON."
    )


def parse_llm_response(response_text: str):
    """
    Parse JSON response from model, and remove markup json language
    """
    return json.loads(response_text.replace("```", "").replace("json", ""))


def get_rating(results):
    """
    Basing this judgement on the assumptions and guideline understanding that
    meeting 5+ criterias is high eligibility standards,
    meeting 3-4 criterias is medium eligibility standards, and the rest are low.
    """
    count_non_empty_criteria = sum(
        1
        for key, value in results.items()
        if value and any(item.strip() for item in value)
    )
    if count_non_empty_criteria >= 5:
        return "high"
    elif count_non_empty_criteria >= 3:
        return "medium"
    else:
        return "low"
