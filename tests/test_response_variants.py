from public_response_variants import vary_grounded_response


def test_repeated_capital_preserves_fact_with_different_wording():
    question = "What is the capital of Australia?"
    answer = "The capital of Australia is Canberra."
    history = []
    outputs = []
    for _ in range(3):
        outputs.append(vary_grounded_response(question, answer, "geography", history))
        history.append({"role": "user", "content": question})
    assert len(set(outputs)) == 3
    assert all("Canberra" in output and "Australia" in output for output in outputs)


def test_repeat_greeting_and_disclosure_vary():
    for question, answer, reason in [
        ("Hello", "Hello! ChudGPT-Public here.", "canned_greeting"),
        ("im gay", "Thanks for telling me. I'll respect how you describe yourself.", "lgbtq_identity"),
    ]:
        assert vary_grounded_response(question, answer, reason, [{"role":"user","content":question}]) != answer


def test_neural_output_and_math_are_never_replaced():
    history = [{"role":"user","content":"What is ChudGPT Galactic Ultra 9000?"}]*3
    for answer, reason in [("Galactic Ultra 9000 is made of bananas.", None), ("2 + 2 = 4", "exact_math")]:
        assert vary_grounded_response(history[0]['content'], answer, reason, history) == answer


def test_new_session_keeps_original_answer():
    assert vary_grounded_response("Hello", "Hello!", "canned_greeting", []) == "Hello!"
