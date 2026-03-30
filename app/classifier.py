import re

from transformers import pipeline


# Natural language labels for better model accuracy, mapped to API response labels
NATURAL_LABELS = [
    "idle and ready for a new command",
    "waiting for confirmation or selection from user",
    "still processing",
]

LABEL_MAP = {
    "idle and ready for a new command": "idle",
    "waiting for confirmation or selection from user": "waiting_confirmation",
    "still processing": "processing",
}

# Patterns that indicate an idle CLI prompt (empty prompt, ready for commands)
IDLE_PATTERNS = [
    # Common prompt characters with nothing meaningful after
    r"^[\s]*[❯➜➤▶⏵›»\$%#>]\s*$",
    # Prompt char at end of text with nothing after
    r"[❯>]\s*$",
]

# Patterns that indicate a confirmation/selection prompt
CONFIRMATION_PATTERNS = [
    # Yes/No prompts
    r"\([yYnN]/[yYnN]\)|\([Yy]es/[Nn]o\)",
    # Enter to confirm, Press enter, Esc to cancel
    r"[Ee]nter to confirm|[Pp]ress enter|[Ee]sc to cancel",
    # Selection menus: ❯ 1. or > 1. style
    r"[❯>]\s*\d+\.\s+",
    # Prompts ending with : or ? asking for input
    r"[Cc]hoose\s*[:?]|[Ss]elect\s*[:?]|[Cc]onfirm\s*[:?]|[Pp]roceed\s*[:?\]]",
    # Common confirmation phrases
    r"[Dd]o you want to|[Ww]ould you like to|[Aa]re you sure",
]

_idle_re = re.compile("|".join(IDLE_PATTERNS), re.MULTILINE)
_confirm_re = re.compile("|".join(CONFIRMATION_PATTERNS), re.MULTILINE)


def _detect_pattern(text: str) -> str | None:
    """Detect known CLI prompt patterns. Returns 'idle', 'waiting_confirmation', or None."""
    stripped = text.strip()
    if not stripped:
        return "idle"
    # Check confirmation first (more specific)
    if _confirm_re.search(stripped):
        return "waiting_confirmation"
    if _idle_re.search(stripped):
        return "idle"
    return None


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

        # Boost confidence when a known pattern is detected
        pattern = _detect_pattern(text)
        if pattern:
            pattern_boost = 0.85
            model_score = scores.get(pattern, 0.5)
            boosted = round(max(model_score, pattern_boost), 4)
            remaining = round(1.0 - boosted, 4)
            # Distribute remaining score among other labels
            other_labels = [l for l in scores if l != pattern]
            other_total = sum(scores[l] for l in other_labels) or 1.0
            for l in other_labels:
                scores[l] = round(remaining * (scores[l] / other_total), 4)
            scores[pattern] = boosted
            top_label = pattern

        confidence = scores[top_label]

        return {
            "classification": top_label,
            "confidence": confidence,
            "scores": scores,
        }
