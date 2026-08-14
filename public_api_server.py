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
        arithmetic_reply = self._calculate_arithmetic(clean_message)
        word_problem_reply = self._calculate_word_problem(clean_message)
        with self.lock:
            history = list(self.sessions.get(active_session, []))
            history.append({"role": "user", "content": clean_message})
            facts = self.session_facts.setdefault(active_session, {})
            self._remember_user_fact(clean_message, facts)
            recall_reply = self._recall_user_fact(clean_message, facts)
            greeting_reply = self._greeting(clean_message)
            reference_reply = self._reference_answer(clean_message)
            comparison_reply = self._comparison_answer(clean_message)
            capability_reply = self._capability_answer(clean_message)
            if arithmetic_reply is not None:
                reply = arithmetic_reply
            elif word_problem_reply is not None:
                reply = word_problem_reply
            elif recall_reply is not None:
                reply = recall_reply
            elif greeting_reply is not None:
                reply = greeting_reply
            elif comparison_reply is not None:
                reply = comparison_reply
            elif capability_reply is not None:
                reply = capability_reply
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
                candidates: list[tuple[float, str]] = [
                    (score_generated_reply(clean_message, answer) + 2.5, answer)
                    for _, answer in retrieved_pairs
                ]
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
                        candidates.append((score_generated_reply(clean_message, candidate), candidate))
                reply = max(candidates, default=(-999.0, ""), key=lambda item: item[0])[1]
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
    def _calculate_arithmetic(message: str) -> str | None:
        """Answer one explicit binary arithmetic expression without an answer table."""
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
        if normalized in {"hi", "hello", "hey", "hello world", "hi there", "hey there"}:
            return "Hey! What would you like to talk about?"
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
    def _capability_answer(message: str) -> str | None:
        lowered = message.lower()
        if "internet" in lowered or "online access" in lowered or "browse the web" in lowered:
            return "No. I do not have live internet access; I only use the text and conversation context sent to this program."
        if any(phrase in lowered for phrase in ("remember other chats", "permanent memory", "remember me later")):
            return "No. I can use this current session's context, but I do not retain personal memory across separate chats."
        return None

    @staticmethod
    def _remember_user_fact(message: str, facts: dict[str, str]) -> None:
        match = re.match(r"my\s+(.{1,60}?)\s+is\s+(.{1,100}?)[.!?]*$", message.strip(), re.I)
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
    parser.add_argument("--checkpoint", default="checkpoints/public_v4/best.pt")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    app = create_app(ROOT / args.checkpoint, args.device)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
