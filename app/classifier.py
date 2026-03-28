from transformers import pipeline


LABELS = ["waiting_input", "processing", "completed"]


class TerminalClassifier:
    def __init__(self):
        self.pipe = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
        )

    def classify(self, text: str) -> dict:
        result = self.pipe(text, candidate_labels=LABELS)

        scores = {
            label: round(score, 4)
            for label, score in zip(result["labels"], result["scores"])
        }

        top_label = result["labels"][0]
        confidence = scores[top_label]

        return {
            "classification": top_label,
            "confidence": confidence,
            "scores": scores,
        }
