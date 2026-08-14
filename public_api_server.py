"""Raw CUDA inference API for the independently trained ChudGPT-Public model."""

from __future__ import annotations

import argparse
import os
import re
import threading
import uuid
from collections import OrderedDict
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
from project_facts import FAMILY_FACTS, FAMILY_SUMMARY, PUBLIC_IDENTITY

ROOT = Path(__file__).resolve().parent
MAX_SESSIONS = 1_000
MAX_MESSAGE_CHARS = 8_000


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    session_id: str | None = Field(default=None, max_length=128)
    max_new_tokens: int = Field(default=200, ge=1, le=400)
    temperature: float = Field(default=0.6, ge=0.0, le=1.5)


class ClearRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)


class PublicModelService:
    """Serve neural conversation with narrow stable-project identity repair."""

    def __init__(self, checkpoint_path: Path, device_name: str, assistance_enabled: bool = True) -> None:
        use_cuda = device_name == "cuda" or (device_name == "auto" and torch.cuda.is_available())
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.device = torch.device("cuda" if use_cuda else "cpu")
        checkpoint = load_checkpoint(checkpoint_path, self.device)
        self.model = TransformerLM(ModelConfig(**checkpoint["model_config"])).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.tokenizer = Tokenizer.from_file(str(ROOT / "artifacts/tokenizer.json"))
        self.eos_id = self.tokenizer.token_to_id("<eos>")
        self.step = int(checkpoint.get("step", 0))
        self.parameters = sum(parameter.numel() for parameter in self.model.parameters())
        self.checkpoint_path = checkpoint_path
        self.sessions: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
        self.lock = threading.Lock()
        self.assistance_enabled = assistance_enabled
        self.last_assistance_reason: str | None = None

    @staticmethod
    def _identity_subject(message: str) -> str | None:
        """Recognize explicit project-identity questions, never general topics."""
        normalized = re.sub(r"[^a-z0-9+#]+", " ", message.lower()).strip()
        is_question = bool(re.search(r"\b(what|who|which|tell|explain|describe|list|are|am|identify|name|how|is)\b", normalized))
        if not is_question:
            return None
        if re.search(r"\b(who|what) (are|is) you\b|\bwhich chudgpt (are|is) (you|this)\b|\bwhat model (are|is)", normalized):
            return "public"
        if "identify yourself" in normalized or re.search(r"\b(plus|pro) or (?:the )?public\b", normalized):
            return "public"
        if "assistant part of" in normalized and "chudgpt" in normalized:
            return "public"
        if "code differ" in normalized and "chudgpt" in normalized:
            return "code"
        if "other chudgpt" in normalized or "chudgpt family" in normalized or "which chudgpt is better" in normalized:
            return "family"
        if "archived" in normalized and ("checkpoint" in normalized or "chudgpt" in normalized):
            return "archived"
        if re.search(r"\b(old|historical) (?:training )?(?:snapshots|checkpoints)\b", normalized):
            return "archived"
        if "deliberately chaotic" in normalized and "chudgpt" in normalized:
            return "mega"
        aliases = {
            "public": ("chudgpt public",), "plus": ("chudgpt plus",),
            "pro": ("chudgpt pro",), "code": ("chudgpt code",),
            "ultimate": ("chudgpt ultimate",), "buggy": ("buggy chudgpt", "buggy mode"),
            "mega": ("mega chud", "chudgpt mega"),
        }
        for subject, names in aliases.items():
            if any(name in normalized for name in names):
                return subject
        if re.search(r"\bwhat is chudgpt\b|\bexplain chudgpt\b|\btell me about chudgpt\b", normalized):
            return "family"
        return None

    @staticmethod
    def _identity_reply_is_sound(reply: str, subject: str) -> bool:
        lowered = reply.lower()
        if "�" in reply or len(reply.split()) < 4:
            return False
        if subject == "public":
            return "chudgpt" in lowered and "public" in lowered and not re.search(r"\bi am (?:chudgpt )?(?:pro|plus|chatgpt)\b", lowered)
        if subject == "family":
            return "chudgpt" in lowered and any(term in lowered for term in ("family", "project", "public", "plus", "pro"))
        expected = {"archived": "checkpoint", "mega": "mega", "buggy": "buggy"}.get(subject, subject)
        return expected in lowered and "chudgpt" in lowered

    def _assist_identity(self, message: str, raw_reply: str) -> tuple[str, str | None]:
        subject = self._identity_subject(message)
        if not self.assistance_enabled or subject is None or self._identity_reply_is_sound(raw_reply, subject):
            return raw_reply, None
        if subject == "public":
            return PUBLIC_IDENTITY, "stable-public-identity"
        if subject == "family":
            return FAMILY_SUMMARY, "stable-family-metadata"
        return FAMILY_FACTS[subject], "stable-family-metadata"

    def _generate_raw(
        self,
        history: list[dict[str, str]],
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        """Generate neural candidates and select the least broken relevant reply."""
        _, prompt_ids = build_context_token_ids(
            self.tokenizer, history, self.model.config.context_length
        )
        prompt_tensor = torch.tensor([prompt_ids], device=self.device)
        temperatures = (max(0.48, temperature - 0.12), temperature, max(0.72, temperature), max(0.86, temperature))
        candidates: list[str] = []
        for attempt_temperature in temperatures:
            output = generate(
                self.model,
                prompt_tensor,
                max_new_tokens=max_new_tokens,
                temperature=attempt_temperature,
                top_k=60,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=self.eos_id,
            )[0, len(prompt_ids):].tolist()
            reply = self.tokenizer.decode(output, skip_special_tokens=True).strip()
            if reply:
                candidates.append(reply)
        if candidates:
            return max(candidates, key=lambda reply: self._candidate_score(history[-1]["content"], reply))
        raise RuntimeError("Model produced empty output after three generation attempts")

    @staticmethod
    def _candidate_score(message: str, reply: str) -> float:
        """Rank neural outputs for readability and topical overlap, never replace them."""
        msg_words = set(re.findall(r"[a-z]{3,}", message.lower()))
        reply_words = re.findall(r"[a-z]{2,}", reply.lower())
        reply_set = set(reply_words)
        score = min(len(reply_words), 45) * 0.025
        score += min(len(msg_words & reply_set), 4) * 1.15
        score += 0.5 if reply.endswith((".", "?", "!", "```")) else 0.0
        score += 0.35 if 4 <= len(reply_words) <= 80 else 0.0
        score -= reply.count("�") * 4.0
        score -= 2.0 if "```" in reply and not re.search(r"\b(code|python|javascript|c#|unity|script|program)\b", message.lower()) else 0.0
        score -= 1.4 if len(reply_words) != len(reply_set) and len(reply_words) > 8 and len(reply_set) / len(reply_words) < 0.58 else 0.0
        score -= 1.2 * sum(fragment in reply.lower() for fragment in ("caption and conversation around it", "the main reason is that cha", "i am the joke-", "that has cha"))
        score -= 0.8 if re.search(r"\b(?:is|are|the|a) (?:a |an )?(?:and|but|or|because)\b", reply.lower()) else 0.0
        return score

    def chat(
        self,
        message: str,
        session_id: str | None,
        max_new_tokens: int = 200,
        temperature: float = 0.6,
    ) -> tuple[str, str]:
        clean_message = message.strip()
        if not clean_message:
            raise ValueError("message cannot be blank")
        active_session = session_id or uuid.uuid4().hex
        with self.lock:
            history = list(self.sessions.get(active_session, []))
            history.append({"role": "user", "content": clean_message})
            # A 21M model becomes self-contaminating when dozens of its own bad
            # generations remain in view. Keep the four most recent exchanges;
            # this is context selection only and never changes model output.
            generation_history = history[-8:]
            raw_reply = self._generate_raw(generation_history, max_new_tokens, temperature)
            reply, self.last_assistance_reason = self._assist_identity(clean_message, raw_reply)
            history.append({"role": "assistant", "content": reply})
            self.sessions[active_session] = history
            self.sessions.move_to_end(active_session)
            while len(self.sessions) > MAX_SESSIONS:
                self.sessions.popitem(last=False)
        return active_session, reply

    def clear(self, session_id: str) -> None:
        with self.lock:
            self.sessions.pop(session_id, None)


def create_app(checkpoint: Path, device: str, assistance_enabled: bool = True) -> FastAPI:
    service = PublicModelService(checkpoint, device, assistance_enabled=assistance_enabled)
    app = FastAPI(title="ChudGPT-Public API", version="10.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/status")
    @app.get("/api/info")
    def info() -> dict[str, object]:
        return {
            "name": "ChudGPT-Public",
            "model": "ChudGPT-Public",
            "status": "online",
            "ready": True,
            "device": str(service.device),
            "parameters": service.parameters,
            "step": service.step,
            "context_length": service.model.config.context_length,
            "checkpoint": str(service.checkpoint_path.relative_to(ROOT)),
            "raw_model_generation": True,
            "identity_assistance": service.assistance_enabled,
            "assistance_scope": "stable ChudGPT identity and family metadata only",
        }

    @app.get("/api")
    def api_index() -> dict[str, object]:
        return {"name": "ChudGPT-Public API", "endpoints": {"chat": "POST /api/chat", "generate": "POST /api/generate", "clear": "POST /api/clear", "info": "GET /api/info"}}

    def run_chat(request: ChatRequest, keep_session: bool) -> dict[str, object]:
        try:
            requested_session = request.session_id if keep_session else uuid.uuid4().hex
            session_id, reply = service.chat(
                request.message, requested_session, request.max_new_tokens, request.temperature
            )
        except (ValueError, RuntimeError) as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        if not keep_session:
            service.clear(session_id)
        return {"reply": reply, "session_id": session_id, "step": service.step,
                "raw_model_generation": True, "assistance_used": service.last_assistance_reason is not None,
                "assistance_reason": service.last_assistance_reason}

    @app.post("/api/chat")
    def chat(request: ChatRequest) -> dict[str, object]:
        return run_chat(request, True)

    @app.post("/api/generate")
    def generate_once(request: ChatRequest) -> dict[str, object]:
        return run_chat(request, False)

    @app.post("/api/clear")
    def clear(request: ClearRequest) -> dict[str, bool]:
        service.clear(request.session_id)
        return {"cleared": True}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve raw ChudGPT-Public inference")
    parser.add_argument("--checkpoint", default="checkpoints/public_v10_balanced/best.pt")
    parser.add_argument("--disable-assistance", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--host", default=os.getenv("CHUDGPT_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("CHUDGPT_PORT", "8010")))
    args = parser.parse_args()
    app = create_app(ROOT / args.checkpoint, args.device, assistance_enabled=not args.disable_assistance)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
