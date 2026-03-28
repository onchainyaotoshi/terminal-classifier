from transformers import pipeline


# Natural language labels for better model accuracy, mapped to API response labels
NATURAL_LABELS = [
    "waiting for user input",
    "still processing",
    "task completed",
]

LABEL_MAP = {
    "waiting for user input": "waiting_input",
    "still processing": "processing",
    "task completed": "completed",
}


class TerminalClassifier:
    def __init__(self):
        self.pipe = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
        )

    def classify(self, text: str) -> dict:
        result = self.pipe(text, candidate_labels=NATURAL_LABELS)

        scores = {
            LABEL_MAP[label]: round(score, 4)
            for label, score in zip(result["labels"], result["scores"])
        }

        top_label = LABEL_MAP[result["labels"][0]]
        confidence = scores[top_label]

        return {
            "classification": top_label,
            "confidence": confidence,
            "scores": scores,
        }
