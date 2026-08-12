"""CUDA inference API for the independently trained ChudGPT-Public model."""

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
        self.lock = threading.Lock()

    def chat(self, message: str, session_id: str | None) -> tuple[str, str]:
        clean_message = message.strip()
        if not clean_message:
            raise ValueError("message cannot be blank")
        active_session = session_id or uuid.uuid4().hex
        with self.lock:
            history = list(self.sessions.get(active_session, []))
            history.append({"role": "user", "content": clean_message})
            _, prompt_ids = build_context_token_ids(
                self.tokenizer, history, self.model.config.context_length
            )
            output = generate(
                self.model,
                torch.tensor([prompt_ids], device=self.device),
                max_new_tokens=200,
                temperature=0.72,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.15,
                eos_token_id=self.eos_id,
            )[0, len(prompt_ids) :].tolist()
            reply = self.tokenizer.decode(output, skip_special_tokens=True).strip()
            if not reply:
                reply = "I could not form a useful answer for that message."
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

    @app.post("/api/chat")
    def chat(request: ChatRequest) -> dict[str, object]:
        try:
            session_id, reply = service.chat(request.message, request.session_id)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"reply": reply, "session_id": session_id, "step": service.step}

    @app.post("/api/clear")
    def clear(request: ClearRequest) -> dict[str, bool]:
        service.clear(request.session_id)
        return {"cleared": True}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve ChudGPT-Public over HTTP")
    parser.add_argument("--checkpoint", default="checkpoints/chat/best.pt")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    app = create_app(ROOT / args.checkpoint, args.device)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
