from app.classifier import TerminalClassifier


def test_classify_returns_all_three_labels():
    classifier = TerminalClassifier()
    result = classifier.classify("Waiting for user input...")
    assert "classification" in result
    assert "confidence" in result
    assert "scores" in result
    assert set(result["scores"].keys()) == {"waiting_input", "processing", "completed"}


def test_classify_scores_sum_to_one():
    classifier = TerminalClassifier()
    result = classifier.classify("Installing packages...")
    total = sum(result["scores"].values())
    assert abs(total - 1.0) < 0.01


def test_classify_confidence_matches_top_score():
    classifier = TerminalClassifier()
    result = classifier.classify("Done.")
    top_label = result["classification"]
    assert result["confidence"] == result["scores"][top_label]


def test_classify_waiting_input():
    classifier = TerminalClassifier()
    result = classifier.classify("Waiting for user input...")
    assert result["classification"] == "waiting_input"


def test_classify_processing():
    classifier = TerminalClassifier()
    result = classifier.classify("⠋ Generating response...")
    assert result["classification"] == "processing"


def test_classify_completed():
    classifier = TerminalClassifier()
    result = classifier.classify("Task completed successfully. Goodbye!")
    assert result["classification"] == "completed"


