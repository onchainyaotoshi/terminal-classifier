from app.classifier import TerminalClassifier, _detect_pattern


def test_classify_returns_three_labels():
    classifier = TerminalClassifier()
    result = classifier.classify("Waiting for user input...")
    assert "classification" in result
    assert "confidence" in result
    assert "scores" in result
    assert set(result["scores"].keys()) == {"idle", "waiting_confirmation", "processing"}


def test_classify_scores_sum_to_one():
    classifier = TerminalClassifier()
    result = classifier.classify("Installing packages...")
    total = sum(result["scores"].values())
    assert abs(total - 1.0) < 0.01


def test_classify_confidence_matches_top_score():
    classifier = TerminalClassifier()
    result = classifier.classify("Some terminal output")
    top_label = result["classification"]
    assert result["confidence"] == result["scores"][top_label]


def test_classify_processing():
    classifier = TerminalClassifier()
    result = classifier.classify("⠋ Generating response...")
    assert result["classification"] == "processing"


# Idle prompt tests
def test_idle_claude_code():
    classifier = TerminalClassifier()
    result = classifier.classify("Claude Code v2.1.87\n~/terminal-classifier\n\n❯  \n")
    assert result["classification"] == "idle"
    assert result["confidence"] >= 0.85


def test_idle_shell():
    classifier = TerminalClassifier()
    result = classifier.classify("$ ")
    assert result["classification"] == "idle"
    assert result["confidence"] >= 0.85


def test_idle_gemini():
    classifier = TerminalClassifier()
    result = classifier.classify("Gemini CLI v1.0.0\n\n❯ ")
    assert result["classification"] == "idle"
    assert result["confidence"] >= 0.85


# Confirmation prompt tests
def test_confirmation_yn():
    classifier = TerminalClassifier()
    result = classifier.classify("Do you want to proceed? (y/n)")
    assert result["classification"] == "waiting_confirmation"
    assert result["confidence"] >= 0.85


def test_confirmation_trust_folder():
    classifier = TerminalClassifier()
    text = """Quick safety check: Is this a project you created or one you trust?

 ❯ 1. Yes, I trust this folder
   2. No, exit

 Enter to confirm · Esc to cancel"""
    result = classifier.classify(text)
    assert result["classification"] == "waiting_confirmation"
    assert result["confidence"] >= 0.85


# Pattern detection unit tests
def test_detect_pattern_idle():
    assert _detect_pattern("❯  ") == "idle"
    assert _detect_pattern("$ ") == "idle"
    assert _detect_pattern("") == "idle"
    assert _detect_pattern("> ") == "idle"


def test_detect_pattern_confirmation():
    assert _detect_pattern("Do you want to proceed? (y/n)") == "waiting_confirmation"
    assert _detect_pattern("Enter to confirm · Esc to cancel") == "waiting_confirmation"
    assert _detect_pattern("  ❯ 1. Yes, I trust this folder") == "waiting_confirmation"
    assert _detect_pattern("Are you sure you want to delete?") == "waiting_confirmation"


def test_detect_pattern_none():
    assert _detect_pattern("⠋ Generating response...") is None
    assert _detect_pattern("Reading file app/main.py") is None
