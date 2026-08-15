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
from chudlm.prompts import DEFAULT_SYSTEM_PROMPT, build_context_token_ids
from project_facts import FAMILY_FACTS, FAMILY_SUMMARY, PUBLIC_IDENTITY
from public_meme_facts import find_meme_fact

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

    def __init__(self, checkpoint_path: Path, device_name: str, assistance_enabled: bool = True,
                 tokenizer_path: Path | None = None,
                 system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> None:
        use_cuda = device_name == "cuda" or (device_name == "auto" and torch.cuda.is_available())
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.device = torch.device("cuda" if use_cuda else "cpu")
        checkpoint = load_checkpoint(checkpoint_path, self.device)
        self.model = TransformerLM(ModelConfig(**checkpoint["model_config"])).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.tokenizer_path = tokenizer_path or ROOT / "artifacts/tokenizer.json"
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        if self.tokenizer.get_vocab_size() != self.model.config.vocab_size:
            raise ValueError("Tokenizer vocabulary does not match checkpoint model configuration")
        self.eos_id = self.tokenizer.token_to_id("<eos>")
        self.step = int(checkpoint.get("step", 0))
        self.parameters = sum(parameter.numel() for parameter in self.model.parameters())
        self.checkpoint_path = checkpoint_path
        self.sessions: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
        self.lock = threading.Lock()
        self.assistance_enabled = assistance_enabled
        self.last_assistance_reason: str | None = None
        self.system_prompt = system_prompt

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
        # Keep project metadata exact without pretending that an arbitrary
        # made-up suffix is a real released profile.
        unknown_profile = re.search(r"\bchudgpt[ -]([a-z][a-z0-9_-]{1,30})\b", normalized)
        if unknown_profile:
            candidate = unknown_profile.group(1)
            if candidate not in {"is", "family", "model", "project"}:
                return f"unknown:{candidate}"
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
        if subject.startswith("unknown:"):
            return False
        expected = {"archived": "checkpoint", "mega": "mega", "buggy": "buggy"}.get(subject, subject)
        return expected in lowered and "chudgpt" in lowered

    def _assist_identity(self, message: str, raw_reply: str) -> tuple[str, str | None]:
        subject = self._identity_subject(message)
        if not self.assistance_enabled or subject is None or self._identity_reply_is_sound(raw_reply, subject):
            return raw_reply, None
        if subject == "public":
            return PUBLIC_IDENTITY, "stable-public-identity"
        if subject == "family":
            reply = FAMILY_SUMMARY
            if re.search(r"\b(?:better|best|stronger|worse|compare)\b", message, re.I):
                reply += " There is no single best variant: Public is the general public model, Code is best suited to programming, Pro favors longer conversations, and Buggy or MEGA CHUD are intentionally unreliable."
            return reply, "stable-family-metadata"
        if subject.startswith("unknown:"):
            display_name = subject.split(":", 1)[1].replace("_", "-").title()
            return (
                f"I do not have a verified ChudGPT profile named ChudGPT-{display_name}. "
                "The known family includes Public, Plus, Pro, Code, Ultimate, Buggy, and MEGA CHUD."
            ), "stable-family-metadata"
        reply = FAMILY_FACTS[subject]
        if re.search(r"\b(?:better|best|stronger|worse|compare)\b", message, re.I):
            reply += " Whether it is better depends on the job: Public is the public general model, Code focuses on programming, and Pro favors longer general conversations."
        return reply, "stable-family-metadata"

    def _assist_meme(self, message: str, raw_reply: str) -> tuple[str, str | None]:
        """Repair only explicitly named, reviewed memes; leave all other text neural."""
        if not self.assistance_enabled:
            return raw_reply, None
        fact = find_meme_fact(message)
        if fact is None:
            return raw_reply, None
        return fact, "reviewed-meme-context"

    def _generate_raw(
        self,
        history: list[dict[str, str]],
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        """Generate neural candidates and select the least broken relevant reply."""
        _, prompt_ids = build_context_token_ids(
            self.tokenizer, history, self.model.config.context_length,
            system_prompt=self.system_prompt,
        )
        prompt_tensor = torch.tensor([prompt_ids], device=self.device)
        # Draw several neural candidates, then rank generated text for relevance
        # and basic fluency. The selector never supplies or rewrites an answer.
        sampling_profiles = (
            (max(0.48, temperature - 0.12), 60, 0.90),
            (temperature, 60, 0.90),
            (max(0.72, temperature), 60, 0.90),
            (max(0.86, temperature), 60, 0.90),
        )
        candidates: list[str] = []
        for attempt_temperature, top_k, top_p in sampling_profiles:
            output = generate(
                self.model,
                prompt_tensor,
                max_new_tokens=max_new_tokens,
                temperature=attempt_temperature,
                top_k=top_k,
                top_p=top_p,
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
        message_lower = message.lower()
        reply_lower = reply.lower()
        msg_words = set(re.findall(r"[a-z]{3,}", message_lower))
        reply_words = re.findall(r"[a-z]{2,}", reply.lower())
        reply_set = set(reply_words)
        score = min(len(reply_words), 45) * 0.025
        score += min(len(msg_words & reply_set), 4) * 1.15
        score += 0.5 if reply.endswith((".", "?", "!", "```")) else 0.0
        score += 0.35 if 4 <= len(reply_words) <= 80 else 0.0
        score -= reply.count("�") * 4.0
        code_request = bool(re.search(r"\b(code|python|javascript|typescript|c#|c\+\+|unity|script|program|html|css|sql|debug)\b", message_lower))
        math_request = bool(re.search(r"\b(calculate|solve|sum|product|percent|percentage|factorial|prime|plus|minus|times|divided)\b|\d\s*(?:[+*/%]|-(?=\s*\d))", message_lower))
        code_output = "```" in reply or bool(re.search(r"\b(?:const|def|class|function|console\.log|using unityengine|public static|return)\b", reply_lower))
        math_output = bool(re.search(r"(?:^|\s)-?\d+(?:\.\d+)?\s*(?:[+*/%=]|-(?=\s*\d))", reply_lower))
        score -= 4.5 if code_output and not code_request else 0.0
        score -= 3.5 if math_output and not math_request else 0.0
        score += 1.5 if code_request and code_output else 0.0
        score += 1.5 if math_request and re.search(r"\d", reply) else 0.0
        greeting = bool(re.fullmatch(r"\s*(?:hi|hello|hey|yo)(?:\s+(?:there|mate|chudgpt))?[!.?]*\s*", message_lower))
        if greeting:
            score += 3.0 if re.search(r"\b(hi|hello|hey|welcome)\b", reply_lower) else -3.0
            score -= 2.0 if re.search(r"\b(?:gravity|python|javascript|percent|calculate|recipe)\b", reply_lower) else 0.0
        # Topic leakage was the main route into "Buggy" behavior. Penalize a
        # response that invents an unrelated domain when the prompt supplied a
        # clear content word, while leaving short/nonsense prompts generative.
        domain_terms = {"book", "music", "movie", "game", "space", "food", "travel", "physics", "recipe", "computer", "robot"}
        introduced = domain_terms & reply_set
        requested = domain_terms & set(re.findall(r"[a-z]+", message_lower))
        score -= 2.5 * len(introduced - requested) if requested or len(msg_words) >= 2 else 0.0
        score -= 1.4 if len(reply_words) != len(reply_set) and len(reply_words) > 8 and len(reply_set) / len(reply_words) < 0.58 else 0.0
        score -= 1.2 * sum(fragment in reply_lower for fragment in ("caption and conversation around it", "the main reason is that cha", "i am the joke-", "that has cha"))
        score -= 0.8 if re.search(r"\b(?:is|are|the|a) (?:a |an )?(?:and|but|or|because)\b", reply_lower) else 0.0
        identity_request = bool(re.search(r"\b(?:who|what) (?:are|is) (?:you|chudgpt)|\bchudgpt (?:family|model|public|code|plus|pro|mega|buggy)\b", message_lower))
        score -= 3.0 if "chudgpt" in reply_lower and not identity_request and "chudgpt" not in message_lower else 0.0
        # Penalize characteristic fragments produced by damaged checkpoints:
        # glued punctuation/words, replacement characters, unbalanced fences,
        # and sentences with very little ordinary connective language.
        score -= 2.5 if re.search(r"[a-z][A-Z]|\w[�]|ï¿½", reply) else 0.0
        score -= 2.0 if reply.count("```") % 2 else 0.0
        score -= min(3.0, len(re.findall(r"\b\w{18,}\b", reply)) * 0.75)
        common = {"the", "a", "an", "is", "are", "was", "to", "of", "and", "or", "but", "it", "that", "this", "you", "i", "for", "with", "in", "on"}
        if len(reply_words) >= 10 and len(common & reply_set) < 2:
            score -= 2.0
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
            if self.last_assistance_reason is None:
                reply, self.last_assistance_reason = self._assist_meme(clean_message, reply)
            history.append({"role": "assistant", "content": reply})
            self.sessions[active_session] = history
            self.sessions.move_to_end(active_session)
            while len(self.sessions) > MAX_SESSIONS:
                self.sessions.popitem(last=False)
        return active_session, reply

    def clear(self, session_id: str) -> None:
        with self.lock:
            self.sessions.pop(session_id, None)


def create_app(checkpoint: Path, device: str, assistance_enabled: bool = True,
               tokenizer_path: Path | None = None) -> FastAPI:
    service = PublicModelService(checkpoint, device, assistance_enabled=assistance_enabled,
                                 tokenizer_path=tokenizer_path)
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
            "assistance_scope": "stable ChudGPT identity/family metadata and explicitly named reviewed memes only",
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
    parser.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    parser.add_argument("--disable-assistance", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--host", default=os.getenv("CHUDGPT_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("CHUDGPT_PORT", "8010")))
    args = parser.parse_args()
    app = create_app(ROOT / args.checkpoint, args.device, assistance_enabled=not args.disable_assistance,
                     tokenizer_path=ROOT / args.tokenizer)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
