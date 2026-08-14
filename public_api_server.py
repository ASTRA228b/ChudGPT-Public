"""Raw CUDA inference API for the independently trained ChudGPT-Public model."""

from __future__ import annotations

import argparse
import os
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
    """Serve only model-generated conversation, with history and technical retries."""

    def __init__(self, checkpoint_path: Path, device_name: str) -> None:
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

    def _generate_raw(
        self,
        history: list[dict[str, str]],
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        """Generate a reply; retries only recover technically empty decoding."""
        _, prompt_ids = build_context_token_ids(
            self.tokenizer, history, self.model.config.context_length
        )
        prompt_tensor = torch.tensor([prompt_ids], device=self.device)
        temperatures = (temperature, max(0.72, temperature), max(0.88, temperature))
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
                return reply
        raise RuntimeError("Model produced empty output after three generation attempts")

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
            reply = self._generate_raw(generation_history, max_new_tokens, temperature)
            history.append({"role": "assistant", "content": reply})
            self.sessions[active_session] = history
            self.sessions.move_to_end(active_session)
            while len(self.sessions) > MAX_SESSIONS:
                self.sessions.popitem(last=False)
        return active_session, reply

    def clear(self, session_id: str) -> None:
        with self.lock:
            self.sessions.pop(session_id, None)


def create_app(checkpoint: Path, device: str) -> FastAPI:
    service = PublicModelService(checkpoint, device)
    app = FastAPI(title="ChudGPT-Public API", version="9.0")
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
            "conversational_fallbacks": False,
            "response_substitution": False,
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
        return {"reply": reply, "session_id": session_id, "step": service.step, "raw_model_generation": True}

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
    parser.add_argument("--checkpoint", default="checkpoints/public_v9_refined/best.pt")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--host", default=os.getenv("CHUDGPT_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("CHUDGPT_PORT", "8010")))
    args = parser.parse_args()
    app = create_app(ROOT / args.checkpoint, args.device)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
