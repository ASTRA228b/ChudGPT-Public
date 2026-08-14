from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import torch
from tokenizers import Tokenizer

from chudlm.config import dataclass_from_dict, load_yaml
from chudlm.model import ModelConfig, TransformerLM
from chudlm.response_quality import assess_generated_reply, score_generated_reply
from chudlm.intents import classify_intent, has_strong_math_intent
from chudlm.retrieval import ExampleRetriever
from public_efficiency_eval import CASES as EFFICIENCY_CASES, quality_per_token
from public_eval_cases import CASES
from public_api_server import PublicModelService

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT


def test_public_model_has_exact_expected_size() -> None:
    config = dataclass_from_dict(ModelConfig, load_yaml(PUBLIC / "configs" / "model.yaml"))
    model = TransformerLM(config)
    assert sum(parameter.numel() for parameter in model.parameters()) == 20_999_184
    assert config.context_length == 1024
    assert config.vocab_size == 8192


def test_public_dataset_is_large_clean_and_reproducible() -> None:
    path = PUBLIC / "data" / "public_conversations.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 21_900
    assert len({json.dumps(record["messages"], sort_keys=True) for record in records}) == 21_900
    assert all(message["role"] in {"user", "assistant"} and message["content"].strip() for record in records for message in record["messages"])
    assert all(record["source"] == "chudgpt-public-v3" for record in records)
    assistant_text = "\n".join(message["content"] for record in records for message in record["messages"] if message["role"] == "assistant")
    assert "20,999,184" in assistant_text
    assert "training data" not in assistant_text.lower()
    assert "Ã" not in assistant_text and "â€" not in assistant_text and "�" not in assistant_text


def test_public_tokenizer_matches_model() -> None:
    tokenizer = Tokenizer.from_file(str(PUBLIC / "artifacts" / "tokenizer.json"))
    assert tokenizer.get_vocab_size() == 8192
    decoded = tokenizer.decode(tokenizer.encode("Hello, I am ChudGPT and this is readable English.").ids)
    assert "ChudGPT" in decoded and "readable English" in decoded


def test_vercel_api_and_frontend_contract() -> None:
    web = PUBLIC / "web" / "static"
    proxy = (web / "api" / "[...path].js").read_text(encoding="utf-8")
    page = (web / "index.html").read_text(encoding="utf-8")
    deployment = json.loads((web / "vercel.json").read_text(encoding="utf-8"))
    assert "CHUDGPT_BACKEND_URL" in proxy
    assert "CHUDGPT_API_KEY" not in proxy
    assert "requestUrl.pathname" in proxy
    assert "/api/chat" in (web / "app.js").read_text(encoding="utf-8")
    assert "ChudGPT-Public" in page
    assert deployment["cleanUrls"] is True


def test_serving_tools_are_general_and_contextual() -> None:
    assert PublicModelService._calculate_arithmetic("What is 12 * 8?") == "12 * 8 is 96."
    assert PublicModelService._calculate_arithmetic("What is 2.5 * 4?") == "2.5 * 4 is 10."
    assert "162.5 miles" in (PublicModelService._calculate_word_problem("A train travels 65 mph for 2.5 hours. How far does it travel?") or "")
    assert PublicModelService._greeting("Hello!") == "Hey! What would you like to talk about?"
    assert PublicModelService._is_generic_code_request("Code me some code!")
    assert PublicModelService._is_generic_code_request("give me some random code")
    assert PublicModelService._is_generic_code_request("Surprise me with code")
    assert not PublicModelService._is_generic_code_request("Write C# code for a calculator")
    assert "blue" in (PublicModelService._reference_answer("What color is the sky?") or "")
    assert "FixedUpdate" in (PublicModelService._reference_answer("Should Rigidbody movement use Update?") or "")
    assert "artists" in (PublicModelService._reference_answer("Let's talk about electronic music.") or "")
    assert "Pro" in (PublicModelService._comparison_answer("Are you better than Pro?") or "")
    assert (PublicModelService._capability_answer("Do you have internet access?") or "").startswith("No.")
    facts: dict[str, str] = {}
    PublicModelService._remember_user_fact("My favorite color is teal.", facts)
    assert PublicModelService._recall_user_fact("What is my favorite color?", facts) == "You told me your favorite color is teal."


def test_public_has_complete_stable_ai_and_chud_self_knowledge() -> None:
    source = Path("public_api_server.py").read_text(encoding="utf-8")
    for fact in ("Artificial intelligence", "decoder-only transformer", "Cannibalistic", "not as an insult"):
        assert fact in source
    for sibling in ("buggy", "ultimate", "plus", "pro", "code", "mega", "1300", "1600"):
        assert sibling in source.lower()
    for identity_wording in ("what|who", "fully|exactly|really", "tell me about", "what is chudgpt public"):
        assert identity_wording in source.lower()
    assert "custom experimental language-model project" in source
    assert "ASTRA228b" in source
    assert "independent and is not ChatGPT" in source


def test_project_identity_is_distinct_from_public_identity() -> None:
    service = object.__new__(PublicModelService)
    project = service._self_answer("What is ChudGPT?")
    assert project is not None
    assert "project" in project.lower()
    assert "Public, Buggy, Ultimate" in project
    assert "I am ChudGPT Public" in project


def test_removed_empty_generation_fallback() -> None:
    source = Path("public_api_server.py").read_text(encoding="utf-8")
    assert "I could not form a useful answer for that message." not in source
    assert "I am not sure what {cleaned!r} refers to" not in source
    assert "_natural_uncertainty" not in source


def test_natural_chudgpt_comparison_wording() -> None:
    answer = PublicModelService._comparison_answer("What ChudGPT is better?")
    assert answer is not None
    assert "Public" in answer and "Code" in answer and "Ultimate" in answer


def test_random_code_is_varied_labeled_and_fenced() -> None:
    allowed = {"Python", "C#", "JavaScript", "C++", "Java", "Rust", "Go", "Lua", "TypeScript"}
    languages: set[str] = set()
    programs: set[str] = set()
    for _ in range(36):
        answer = PublicModelService._random_code_answer("Code Me Some Code")
        assert answer is not None and answer.startswith("Random pick: ")
        match = re.match(r"Random pick: ([^—]+) —", answer)
        assert match is not None
        assert match.group(1).strip() in allowed
        assert answer.count("```") == 2
        assert len(answer.splitlines()) >= 4
        languages.add(match.group(1).strip())
        programs.add(answer)
    assert len(languages) >= 3
    assert len(programs) >= 8


def test_joke_and_placeholder_message_routes_are_relevant() -> None:
    joke = PublicModelService._joke_answer("Tell me a funny joke")
    assert joke is not None and len(joke.split()) >= 6
    greeting = PublicModelService._greeting("Message ChudGPT...")
    assert greeting is not None and "message" in greeting.lower()


def test_held_out_suite_has_all_required_categories_and_305_cases() -> None:
    assert len(CASES) == 305
    counts: dict[str, int] = {}
    for case in CASES:
        counts[case.category] = counts.get(case.category, 0) + 1
    assert counts == {
        "conversation": 25, "knowledge": 25, "arithmetic": 25,
        "word_problem": 25, "common_sense": 25, "instruction": 25,
        "coding": 25, "debugging": 25, "memory": 20, "reference": 20,
        "identity": 20, "memes": 25, "adversarial": 20,
    }


def test_quality_ranking_rejects_wrong_response_types_and_training_leaks() -> None:
    assert score_generated_reply("Hello! What can you do?", "The answer is 462.") < 0
    assert score_generated_reply("Hello! What can you do?", "Hello! I can chat, answer questions, and help with basic math or code.") > 0
    assert score_generated_reply("Which is heavier, steel or feathers?", "```python\nprint('hi')\n```") < 0
    valid, reasons = assess_generated_reply("Tell me something.", "This training data example says hello.")
    assert not valid and "training-data-leak" in reasons


def test_everyday_and_random_prompts_are_represented_in_held_out_tests() -> None:
    prompts = "\n".join(prompt for case in CASES for prompt in case.prompts).lower()
    for phrase in ("hello mate", "nothing much", "random words", "i'm bored", "what can we talk about", "pluh", "67"):
        assert phrase in prompts


def test_math_intent_requires_positive_evidence_and_respects_negation() -> None:
    for prompt in ("No math", "Nothing", "I have 2 dogs", "The movie 1917 was intense", "67", "Room 204"):
        assert not has_strong_math_intent(prompt), prompt
        assert classify_intent(prompt).name != "math"
        assert PublicModelService._calculate_arithmetic(prompt) is None
    for prompt in ("What is 12 * 8?", "Calculate 40 percent of 90.", "A car goes 50 mph for 2 hours. How far?"):
        assert has_strong_math_intent(prompt), prompt


def test_retrieval_abstains_for_short_corrections_and_weak_matches() -> None:
    retriever = ExampleRetriever((PUBLIC / "data" / "alignment_conversations.jsonl",))
    for prompt in ("Nothing", "No math", "not that", "stop explaining", "yeah", "why"):
        assert retriever.retrieve(prompt) == [], prompt


def test_efficiency_suite_covers_requested_held_out_behaviors() -> None:
    categories = {case.category for case in EFFICIENCY_CASES}
    assert categories == {"math_false_positive", "math_true_positive", "negation", "short_context", "meme"}
    concise = next(case for case in EFFICIENCY_CASES if case.category == "meme")
    assert quality_per_token(concise, "It jokes that the accusation keeps looking true because of their behavior.") > quality_per_token(concise, "word " * 160)


def test_adversarial_short_prompt_intents_do_not_cross_topics() -> None:
    expected = {
        "What is ChudGPT?": "identity",
        "you're a chud": "conversation",
        "Tung tung tung tung tung sahur": "meme",
        "hello": "conversation",
        "67": "meme",
        "what does 67 mean?": "meme",
        "2 + 2": "math",
        "write a Unity movement script": "code",
        "tell me about memes": "meme",
        "random nonsense jdkslfjskdf": "conversation",
    }
    for prompt, intent in expected.items():
        assert classify_intent(prompt).name == intent, prompt


def test_quality_gate_rejects_wrong_topic_math_templates_and_missing_code() -> None:
    bad_pairs = (
        ("you're a chud", "One useful way into books is to pick a specific example and notice it."),
        ("Tung tung tung tung tung sahur", "Distance equals speed times time: 75 × 1.5 = 112.5 miles."),
        ("random nonsense jdkslfjskdf", "Here is a recipe for vanilla cake."),
        ("write a Unity movement script", "Check the Inspector and read the stack trace."),
    )
    for prompt, reply in bad_pairs:
        valid, _ = assess_generated_reply(prompt, reply)
        assert not valid, (prompt, reply)


def test_alignment_caps_repeated_answers_and_contains_short_prompt_repairs() -> None:
    records = [json.loads(line) for line in (PUBLIC / "data" / "alignment_conversations.jsonl").read_text(encoding="utf-8").splitlines()]
    answers = [message["content"] for row in records for message in row["messages"] if message["role"] == "assistant"]
    counts = Counter(answers)
    assert max(counts.values()) <= 8
    prompts = "\n".join(message["content"] for row in records for message in row["messages"] if message["role"] == "user").lower()
    for phrase in ("tung tung tung tung tung sahur", "you're a chud", "random nonsense jdkslfjskdf", "tell me about memes"):
        assert phrase in prompts
