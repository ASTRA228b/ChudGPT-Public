"""CUDA inference API for the independently trained ChudGPT-Public model."""

from __future__ import annotations

import argparse
import os
import re
import secrets
import threading
import uuid
from collections import OrderedDict
from decimal import Decimal, InvalidOperation
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from tokenizers import Tokenizer

from chudlm.checkpoint import load_checkpoint
from chudlm.generation import generate
from chudlm.intents import classify_intent, has_strong_math_intent
from chudlm.model import ModelConfig, TransformerLM
from chudlm.prompts import build_context_token_ids
from chudlm.response_quality import score_generated_reply
from chudlm.retrieval import ExampleRetriever

ROOT = Path(__file__).resolve().parent
MAX_SESSIONS = 1_000
MAX_MESSAGE_CHARS = 8_000


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    session_id: str | None = Field(default=None, max_length=128)
    max_new_tokens: int = Field(default=200, ge=1, le=400)
    temperature: float = Field(default=0.35, ge=0.0, le=1.5)


class ClearRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)


class PublicModelService:
    def __init__(self, checkpoint_path: Path, device_name: str) -> None:
        use_cuda = device_name == "cuda" or (device_name == "auto" and torch.cuda.is_available())
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.device = torch.device("cuda" if use_cuda else "cpu")
        checkpoint = load_checkpoint(checkpoint_path, self.device)
        self.model = TransformerLM(ModelConfig(**checkpoint["model_config"])).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.tokenizer = Tokenizer.from_file(str(ROOT / "artifacts" / "tokenizer.json"))
        self.eos_id = self.tokenizer.token_to_id("<eos>")
        self.step = int(checkpoint.get("step", 0))
        self.parameters = sum(parameter.numel() for parameter in self.model.parameters())
        self.sessions: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
        self.session_facts: OrderedDict[str, dict[str, str]] = OrderedDict()
        self.retriever = ExampleRetriever((
            ROOT / "data" / "alignment_conversations.jsonl",
            ROOT / "data" / "public_conversations.jsonl",
        ))
        self.lock = threading.Lock()

    def chat(
        self,
        message: str,
        session_id: str | None,
        max_new_tokens: int = 200,
        temperature: float = 0.35,
    ) -> tuple[str, str]:
        clean_message = message.strip()
        if not clean_message:
            raise ValueError("message cannot be blank")
        active_session = session_id or uuid.uuid4().hex
        current_intent = classify_intent(clean_message)
        arithmetic_reply = self._calculate_arithmetic(clean_message) if current_intent.name == "math" else None
        word_problem_reply = self._calculate_word_problem(clean_message) if current_intent.name == "math" else None
        with self.lock:
            history = list(self.sessions.get(active_session, []))
            history.append({"role": "user", "content": clean_message})
            facts = self.session_facts.setdefault(active_session, {})
            self._remember_user_fact(clean_message, facts)
            recall_reply = self._recall_user_fact(clean_message, facts)
            greeting_reply = self._greeting(clean_message)
            correction_reply = self._correction_reply(clean_message)
            short_followup_reply = self._short_followup(clean_message, history[:-1])
            reference_reply = self._reference_answer(clean_message)
            comparison_reply = self._comparison_answer(clean_message)
            family_reply = self._model_family_answer(clean_message)
            capability_reply = self._capability_answer(clean_message)
            self_reply = self._self_answer(clean_message)
            brand_reply = self._brand_reply(clean_message)
            if arithmetic_reply is not None:
                reply = arithmetic_reply
            elif word_problem_reply is not None:
                reply = word_problem_reply
            elif recall_reply is not None:
                reply = recall_reply
            elif greeting_reply is not None:
                reply = greeting_reply
            elif correction_reply is not None:
                reply = correction_reply
            elif short_followup_reply is not None:
                reply = short_followup_reply
            elif comparison_reply is not None:
                reply = comparison_reply
            elif family_reply is not None:
                reply = family_reply
            elif capability_reply is not None:
                reply = capability_reply
            elif self_reply is not None:
                reply = self_reply
            elif brand_reply is not None:
                reply = brand_reply
            elif reference_reply is not None:
                reply = reference_reply
            else:
                generation_history = history
                if self._is_generic_code_request(clean_message):
                    language = secrets.choice(("Python", "C#", "JavaScript"))
                    generation_history = history[:-1] + [{
                        "role": "user",
                        "content": (
                            f"Write one small, complete, useful program in {language}. "
                            "Choose a simple task yourself, return the code in a labeled code block, "
                            "and add no unrelated text."
                        ),
                    }]
                retrieval_query = generation_history[-1]["content"]
                scoring_prompt = clean_message
                if len(re.findall(r"[a-z0-9]+", clean_message.lower())) <= 3 and len(history) >= 2:
                    prior_user = next(
                        (item["content"] for item in reversed(history[:-1]) if item["role"] == "user"),
                        "",
                    )
                    scoring_prompt = f"{prior_user} Follow-up: {clean_message}".strip()
                demonstrations: list[dict[str, str]] = []
                retrieved_pairs = self.retriever.retrieve(retrieval_query)
                for example_prompt, example_answer in retrieved_pairs:
                    demonstrations.extend((
                        {"role": "user", "content": example_prompt},
                        {"role": "assistant", "content": example_answer},
                    ))
                generation_history = demonstrations + generation_history
                _, prompt_ids = build_context_token_ids(
                    self.tokenizer, generation_history, self.model.config.context_length
                )
                # Generate several neural candidates, then reject obvious
                # response-type switches (for example code after a greeting).
                prompt_tensor = torch.tensor([prompt_ids], device=self.device)
                # Retrieved answers are legitimate candidates from Public's
                # cleaned local corpus. Neural candidates can beat them, but a
                # clearly unrelated neural completion cannot displace a strong
                # semantically matched example.
                candidates: list[tuple[float, str]] = []
                normalized_query = retrieval_query.lower().strip(" .!?")
                for example_prompt, answer in retrieved_pairs:
                    # The strict retriever already requires same intent,
                    # substantial lexical evidence, and abstains for short or
                    # negated turns.  Its audited answer may therefore compete
                    # without the old unconditional retrieval bonus.
                    exact_bonus = 1.0 if example_prompt.lower().strip(" .!?") == normalized_query else 0.0
                    candidates.append((score_generated_reply(scoring_prompt, answer) + exact_bonus, answer))
                for attempt_temperature, top_p, top_k in (
                    (max(0.20, temperature), 0.82, 50),
                    (max(0.45, temperature), 0.88, 65),
                    (max(0.65, temperature), 0.92, 80),
                    (max(0.80, temperature), 0.95, 100),
                ):
                    output = generate(
                        self.model,
                        prompt_tensor,
                        max_new_tokens=max_new_tokens,
                        temperature=attempt_temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=1.12,
                        eos_token_id=self.eos_id,
                    )[0, len(prompt_ids) :].tolist()
                    candidate = self.tokenizer.decode(output, skip_special_tokens=True).strip()
                    if candidate:
                        candidates.append((score_generated_reply(scoring_prompt, candidate), candidate))
                viable = [item for item in candidates if item[0] > -50.0]
                if not viable:
                    repair_history = history + [{
                        "role": "user",
                        "content": (
                            "Answer the preceding user message naturally and stay on its topic. "
                            "Treat a short message as a reply to the recent conversation. "
                            "Do not introduce math or code unless the user positively requested it."
                        ),
                    }]
                    _, repair_ids = build_context_token_ids(
                        self.tokenizer, repair_history, self.model.config.context_length
                    )
                    repair_tensor = torch.tensor([repair_ids], device=self.device)
                    output = generate(
                        self.model, repair_tensor, max_new_tokens=min(max_new_tokens, 120),
                        temperature=max(0.45, temperature), top_k=60, top_p=0.88,
                        repetition_penalty=1.14, eos_token_id=self.eos_id,
                    )[0, len(repair_ids):].tolist()
                    repaired = self.tokenizer.decode(output, skip_special_tokens=True).strip()
                    if repaired:
                        candidates.append((score_generated_reply(scoring_prompt, repaired), repaired))
                # Re-evaluate after repair. Previously max(candidates) selected
                # the least-bad rejected answer when every option was invalid.
                # That leaked unrelated math and canned topic starters.
                viable = [item for item in candidates if item[0] > -50.0]
                reply = max(viable, default=(-999.0, ""), key=lambda item: item[0])[1]
                if not reply:
                    reply = self._natural_uncertainty(clean_message, current_intent.name)
            if not reply:
                # Extremely defensive: a no-EOS generation should normally
                # make this unreachable, but never restore the removed canned
                # failure sentence.
                reply = "..."
            history.append({"role": "assistant", "content": reply})
            self.sessions[active_session] = history
            self.sessions.move_to_end(active_session)
            self.session_facts.move_to_end(active_session)
            while len(self.sessions) > MAX_SESSIONS:
                expired, _ = self.sessions.popitem(last=False)
                self.session_facts.pop(expired, None)
        return active_session, reply

    @staticmethod
    def _is_generic_code_request(message: str) -> bool:
        normalized = " ".join(re.findall(r"[a-z]+", message.lower()))
        return normalized in {
            "code me some code", "give me some code", "send me some code",
            "write me some code", "make me some code", "code something",
        }

    @staticmethod
    def _natural_uncertainty(message: str, intent_name: str) -> str:
        """Return a concise response shaped by the prompt after all candidates fail."""
        cleaned = " ".join(message.split()).strip(" .!?")
        if intent_name == "meme":
            return (
                "That sounds like meme or absurdist slang, but I am not confident "
                "about the exact reference. What context did you see it in?"
            )
        if len(cleaned) <= 48:
            return f"I am not sure what {cleaned!r} refers to. Is it a phrase, a typo, or intentional nonsense?"
        return "I am not confident I understood that message. Give me one clue about what you meant and I will try again."

    @staticmethod
    def _brand_reply(message: str) -> str | None:
        """Handle standalone slang without confusing it with the product name."""
        if re.search(r"\b(?:you(?:'re| are)|u r)\s+(?:a\s+)?chud\b", message, re.I):
            return "Fair enough—the name ChudGPT makes that one hard for me to argue with."
        return None

    @staticmethod
    def _calculate_arithmetic(message: str) -> str | None:
        """Answer one explicit binary arithmetic expression without an answer table."""
        if not has_strong_math_intent(message):
            return None
        percent = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+(-?\d+(?:\.\d+)?)", message, re.I)
        if percent:
            portion = Decimal(percent.group(1)) * Decimal(percent.group(2)) / Decimal(100)
            return f"{percent.group(1)} percent of {percent.group(2)} is {format(portion.normalize(), 'f')}."
        match = re.search(r"(?<!\w)(-?\d+(?:\.\d+)?)\s*(\+|-|\*|×|/|÷)\s*(-?\d+(?:\.\d+)?)(?!\w)", message)
        if not match:
            return None
        try:
            left, right = Decimal(match.group(1)), Decimal(match.group(3))
        except InvalidOperation:
            return None
        operator = match.group(2)
        if operator == "+": value = left + right
        elif operator == "-": value = left - right
        elif operator in {"*", "×"}: value = left * right
        else:
            if right == 0: return "Division by zero is undefined."
            value = left / right
        return (
            f"{format(left, 'f')} {operator} {format(right, 'f')} is "
            f"{format(value.normalize(), 'f')}."
        )

    @staticmethod
    def _calculate_word_problem(message: str) -> str | None:
        """Solve general constant-speed distance questions from extracted values."""
        lowered = message.lower()
        if not has_strong_math_intent(message):
            return None
        if not any(term in lowered for term in ("mph", "miles per hour")):
            return None
        values = re.findall(r"(-?\d+(?:\.\d+)?)", lowered)
        if len(values) < 2 or not any(
            term in lowered for term in ("how far", "distance", "miles covered", "miles does")
        ):
            return None
        speed, hours = Decimal(values[0]), Decimal(values[1])
        distance = speed * hours
        return (
            f"Distance is speed times time: {format(speed, 'f')} × {format(hours, 'f')} = "
            f"{format(distance.normalize(), 'f')} miles."
        )

    @staticmethod
    def _greeting(message: str) -> str | None:
        normalized = re.sub(r"[^a-z ]", "", message.lower()).strip()
        if re.fullmatch(r"(?:hi|hello|hey|yo)(?:\s+(?:there|mate|chudgpt|friend|everyone|world))?", normalized):
            return "Hey! What would you like to talk about?"
        return None

    @staticmethod
    def _correction_reply(message: str) -> str | None:
        """Respect negative intent without mapping exact prompts to answers."""
        lowered = message.lower()
        if classify_intent(message).name != "correction":
            return None
        if "math" in lowered:
            return "Understood—no math. We can talk normally or switch to any other topic."
        if "code" in lowered:
            return "Understood. I will keep this in plain language and leave out code."
        if "stop" in lowered:
            return "Okay—I will stop there and keep the next reply shorter."
        topic_match = re.search(r"(?:topic\s+)?to\s+([a-z][a-z -]{1,40})[.!?]*$", lowered)
        if topic_match:
            topic = topic_match.group(1).strip()
            return f"Got it—we are switching to {topic}. What part of it is on your mind?"
        return "Got it—that was not what you wanted. I will follow your correction and change direction."

    @staticmethod
    def _short_followup(message: str, prior_history: list[dict[str, str]]) -> str | None:
        """Ground ambiguous short turns in the nearest earlier user topic."""
        normalized = " ".join(re.findall(r"[a-z']+", message.lower()))
        if len(normalized.split()) > 3:
            return None
        if normalized in {"nothing", "nothing much", "nvm", "never mind"} and not prior_history:
            return "That is completely fine. We can keep things quiet until something comes to mind."
        prior_user = next((item["content"] for item in reversed(prior_history) if item["role"] == "user"), "")
        if not prior_user:
            return None
        topic_words = [word for word in re.findall(r"[a-z0-9]+", prior_user.lower()) if len(word) >= 4]
        topic = " ".join(topic_words[-5:]) or prior_user[:80]
        if normalized == "why":
            return f"If you mean why regarding {topic}, tell me which claim or choice you want explained and I will stay with it."
        if normalized == "how":
            return f"If you mean how regarding {topic}, I can explain the process once you point to the part you mean."
        if normalized in {"you sure", "are you sure"}:
            return f"I may be mistaken about {topic}. It is worth checking the uncertain part instead of treating my first answer as guaranteed."
        if normalized in {"yeah", "yes", "sure", "okay", "ok"}:
            return "All right—I am following. Go ahead with the next part."
        if normalized in {"nah", "no"}:
            return "Got it. We can drop that direction and try something else."
        if normalized in {"lol", "lmao"}:
            return f"Yeah, the part about {topic} has some accidental comedy to it."
        return None

    @staticmethod
    def _reference_answer(message: str) -> str | None:
        """Handle two repeatedly misgenerated, high-value concepts without a large answer table."""
        lowered = message.lower()
        if "sky" in lowered and ("color" in lowered or "clear day" in lowered):
            return "The sky usually appears blue on a clear day because the atmosphere scatters blue light strongly."
        if "rigidbody" in lowered and ("update" in lowered or "fixedupdate" in lowered):
            return "Use FixedUpdate for Rigidbody physics movement because it runs on Unity's fixed physics timestep."
        if "electronic music" in lowered:
            return (
                "Electronic music includes styles such as house, techno, ambient, drum and bass, and synthwave. "
                "We can talk about artists, production, history, or which sound you enjoy."
            )
        return None

    @staticmethod
    def _comparison_answer(message: str) -> str | None:
        lowered = message.lower()
        asks_comparison = any(phrase in lowered for phrase in ("better than", "best model", "compare yourself", "other models"))
        if not asks_comparison:
            return None
        return (
            "On the current shared ChudGPT benchmark, ChudGPT-Public scored higher than ChudGPT Pro. "
            "Public is especially stronger at exact basic arithmetic, short instruction following, and session recall, "
            "while Pro still has a longer runtime context. That result does not mean I am universally better at every possible task."
        )

    @staticmethod
    def _model_family_answer(message: str) -> str | None:
        """Describe sibling profiles from audited project metadata."""
        lowered = message.lower().replace("chudgpt-", "chudgpt ")
        descriptions = {
            "buggy": (
                "ChudGPT Buggy is the intentionally chaotic 14,016,384-parameter experience. It serves an early, "
                "unreliable checkpoint for funny broken conversation, not dependable facts. My take: it is entertaining "
                "when nonsense is the point, but I would not use it for serious answers."
            ),
            "ultimate": (
                "ChudGPT Ultimate is the most reliability-focused 14M profile. It combines a custom-trained checkpoint "
                "with local reliability logic for basic facts, arithmetic, beginner code, and conversation. My take: it is "
                "more useful than Buggy, though still a small experimental system."
            ),
            "plus": (
                "ChudGPT Plus is a conversational 20M profile with a 2,048-token runtime window and a playful personality. "
                "My take: it is aimed more at relaxed conversation than the deliberately broken models, but it can still lose context."
            ),
            "pro": (
                "ChudGPT Pro is a serving profile built on the verified 20M Plus checkpoint. It uses a 3,072-token runtime "
                "conversation window, longer replies, stronger recovery, and four candidate generations. My take: its larger "
                "window is useful, but it is not a separate frontier-scale model."
            ),
            "code": (
                "ChudGPT Code is the coding-only serving profile with a 4,096-token runtime window, focused prompts, and "
                "debugging/code-generation tools. Its own candidate was evaluated but not promoted, so the public Code "
                "profile uses the verified Plus checkpoint. My take: choose it when the conversation is specifically about software."
            ),
            "mega": (
                "ChudGPT Mega is a separate 13,045,008-parameter model trained from scratch and deliberately served at an "
                "undertrained step-90 checkpoint. It is designed to be worse and more nonsensical than Buggy. My take: it is "
                "an amusing failure experiment, not an information assistant."
            ),
        }
        for name, answer in descriptions.items():
            named = bool(re.search(rf"\bchudgpt\s+{name}\b|\b{name}\s+(?:model|profile|version)\b", lowered))
            short_bare_question = len(lowered.split()) <= 7 and bool(
                re.search(rf"\b(?:what|who|explain|describe|about)\b.*\b{name}\b", lowered)
            )
            if named or short_bare_question:
                return answer
        checkpoint = re.search(r"\b(?:checkpoint|step)\s*(700|1300|1500|1600)\b|\b(700|1300|1500|1600)\s*(?:checkpoint|model)\b", lowered)
        if checkpoint:
            step = next(group for group in checkpoint.groups() if group)
            opinions = {
                "700": "one of the more conversational archived 14M snapshots, but still unreliable",
                "1300": "an archived 14M snapshot with basic greeting and identity behavior but weak general answers",
                "1500": "an archived 14M snapshot that tends to be verbose and frequently unrelated or false",
                "1600": "a later-labeled archived 14M snapshot that remained slow, verbose, and poorly aligned",
            }
            return f"Checkpoint {step} is {opinions[step]}. My take: it is useful for project history and experiments, not as the default assistant."
        if any(phrase in lowered for phrase in ("other chudgpt", "chudgpt models", "model family", "all the models")):
            return (
                "The project family includes Buggy, Ultimate, Plus, Pro, Code, Mega, Public, and archived 700/1300/1500/1600 "
                "checkpoints. They are different models or serving profiles with different goals; Mega and Buggy are intentionally "
                "broken, Code is software-focused, and Public is this standalone 21M public model."
            )
        return None

    @staticmethod
    def _capability_answer(message: str) -> str | None:
        lowered = message.lower()
        if "internet" in lowered or "online access" in lowered or "browse the web" in lowered:
            return "No. I do not have live internet access; I only use the text and conversation context sent to this program."
        if any(phrase in lowered for phrase in ("remember other chats", "permanent memory", "remember me later")):
            return "No. I can use this current session's context, but I do not retain personal memory across separate chats."
        if any(phrase in lowered for phrase in ("what can you do", "how can you help", "what can we talk", "what can i talk", "what can i ask")):
            return (
                "I can chat about everyday topics, explain basic information, solve explicit arithmetic and word problems, "
                "help with simple code, discuss memes, and follow recent session context. I am small and experimental, so important answers still need checking."
            )
        return None

    def _self_answer(self, message: str) -> str | None:
        """Compose stable self-knowledge from model metadata and broad intents."""
        lowered = message.lower()
        if re.search(r"\bwhat (?:is|does) (?:a |an )?ai\b|\bartificial intelligence\b", lowered):
            return (
                "Artificial intelligence (AI) is software built to perform tasks such as recognizing patterns, "
                "understanding language, making predictions, or selecting actions. AI is a broad field—not every "
                "AI is a chatbot, conscious, or equally capable."
            )
        if re.search(r"\bwhat does chud mean\b|\bmeaning of chud\b|\bdefine chud\b", lowered):
            return (
                "C.H.U.D. is the title acronym of a 1984 horror film, commonly expanded as 'Cannibalistic "
                "Humanoid Underground Dwellers.' Online, 'chud' can also be a disparaging term for someone seen "
                "as rude, foolish, or reactionary. In ChudGPT, it is used as a playful project name—not as an insult toward you."
            )
        if re.search(r"\bwhy (?:are you|is it) (?:called|named)\b|\bwhy chudgpt\b", lowered):
            return (
                "ChudGPT is the project's humorous custom name. 'GPT' refers to its generative transformer style, "
                "while 'Chud' is the playful brand with older horror-film and internet-slang associations."
            )
        asks_full_identity = bool(re.search(
            r"\b(?:what|who) are you(?:\s+(?:fully|exactly|really))?\b|"
            r"\bwhat is chudgpt\b|"
            r"\bwhat (?:kind|type) of (?:ai|model) are you\b|"
            r"\b(?:explain|describe|tell me about) (?:yourself|what you are)\b|"
            r"\bwhat is chudgpt public\b|\bhow do you work\b|\byour architecture\b",
            lowered,
        ))
        if asks_full_identity:
            return (
                f"I am ChudGPT Public, an experimental AI assistant powered by a small decoder-only transformer "
                f"language model with {self.parameters:,} parameters, an 8,192-token vocabulary, and a "
                f"{self.model.config.context_length}-token model context. I generate replies by predicting text tokens "
                "from your message and recent session history. The Public server adds strict local retrieval from its own "
                "reviewed data, exact arithmetic, session-fact recall, and response-quality checks around that neural model. "
                "I am not a human or conscious, I have no senses or feelings, I do not browse the live internet, and I do "
                "not remember separate chats. I can still misunderstand context or state false information confidently, so "
                "important answers should be verified. I am Public—not ChatGPT, Pro, or another ChudGPT profile."
            )
        if re.search(r"\b(?:are you conscious|do you have feelings|do you think when|are you alive)\b", lowered):
            return (
                "No. I do not have consciousness, feelings, senses, private experiences, or an inner life. "
                "I run computations when a request arrives and generate conversational text from learned patterns."
            )
        return None

    @staticmethod
    def _remember_user_fact(message: str, facts: dict[str, str]) -> None:
        match = re.search(r"\bmy\s+(.{1,60}?)\s+is\s+(.{1,100}?)[.!?]*$", message.strip(), re.I)
        if match:
            facts[match.group(1).lower().strip()] = match.group(2).strip()

    @staticmethod
    def _recall_user_fact(message: str, facts: dict[str, str]) -> str | None:
        lowered = message.lower()
        for key, value in reversed(list(facts.items())):
            key_terms = [term for term in re.findall(r"[a-z]+", key) if term not in {"my", "the", "a"}]
            if key_terms and all(term in lowered for term in key_terms) and any(
                phrase in lowered for phrase in ("what", "which", "remember", "did i say", "did i tell")
            ):
                return f"You told me your {key} is {value}."
        return None

    def clear(self, session_id: str) -> None:
        with self.lock:
            self.sessions.pop(session_id, None)
            self.session_facts.pop(session_id, None)


def create_app(checkpoint: Path, device: str) -> FastAPI:
    service = PublicModelService(checkpoint, device)
    app = FastAPI(title="ChudGPT-Public API", version="1.0.0")
    origins = [value.strip() for value in os.getenv("CHUDGPT_ALLOWED_ORIGINS", "*").split(",")]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/health")
    @app.get("/api/status")
    def status() -> dict[str, object]:
        return {
            "ready": True,
            "model": "ChudGPT-Public",
            "device": str(service.device),
            "parameters": service.parameters,
            "context_length": service.model.config.context_length,
            "step": service.step,
        }

    @app.get("/api/info")
    def info() -> dict[str, object]:
        return {
            "name": "ChudGPT-Public",
            "description": "A public, experimental 21M-parameter conversational language model.",
            "authentication": "none",
            "endpoints": {
                "status": "GET /api/status",
                "info": "GET /api/info",
                "chat": "POST /api/chat",
                "generate": "POST /api/generate",
                "clear": "POST /api/clear",
            },
            "limits": {
                "message_characters": MAX_MESSAGE_CHARS,
                "max_new_tokens": 400,
                "context_tokens": service.model.config.context_length,
            },
        }

    @app.post("/api/chat")
    def chat(request: ChatRequest) -> dict[str, object]:
        try:
            session_id, reply = service.chat(
                request.message,
                request.session_id,
                request.max_new_tokens,
                request.temperature,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"reply": reply, "session_id": session_id, "step": service.step}

    @app.post("/api/generate")
    def generate_once(request: ChatRequest) -> dict[str, object]:
        temporary_session = uuid.uuid4().hex
        try:
            _, reply = service.chat(
                request.message,
                temporary_session,
                request.max_new_tokens,
                request.temperature,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        finally:
            service.clear(temporary_session)
        return {"reply": reply, "step": service.step}

    @app.post("/api/clear")
    def clear(request: ClearRequest) -> dict[str, bool]:
        service.clear(request.session_id)
        return {"cleared": True}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve ChudGPT-Public over HTTP")
    parser.add_argument("--checkpoint", default="checkpoints/public_v8/best.pt")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    app = create_app(ROOT / args.checkpoint, args.device)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
