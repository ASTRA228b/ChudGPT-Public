"""Raw CUDA inference API for the independently trained ChudGPT-Public model."""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
import uuid
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from tokenizers import Tokenizer

from chudlm.checkpoint import load_checkpoint
from chudlm.emoji_awareness import (
    add_emoji_context,
    emoji_database,
    emoji_semantic_response,
    strip_emoji_context,
)
from chudlm.generation import generate
from chudlm.model import ModelConfig, TransformerLM
from chudlm.prompts import DEFAULT_SYSTEM_PROMPT, build_context_token_ids
from chudlm.response_quality import (
    assess_generated_reply,
    has_structured_list,
    requests_structured_response,
    score_generated_reply,
)
from chudlm.text_normalization import normalize_user_text
from project_facts import FAMILY_FACTS, FAMILY_SUMMARY, PUBLIC_IDENTITY
from public_meme_facts import find_meme_fact
from public_math import exact_math_response
from public_instructions import exact_instruction_response
from public_reliable import PublicReliableResponder
from music_instructions import MUSIC_MODEL_NAME, MUSIC_SYSTEM_PROMPT

ROOT = Path(__file__).resolve().parent
MAX_SESSIONS = 1_000
MAX_MESSAGE_CHARS = 8_000
SERVING_CONFIG_PATH = ROOT / "serving_config.json"
PUBLIC_VERSION = "20.0"
DISCORD_BOT_INSTRUCTION = (ROOT / "discord_bot_instruction.txt").read_text(encoding="utf-8").strip()
DISCORD_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT + " " + DISCORD_BOT_INSTRUCTION


def selected_checkpoint() -> str:
    """Load the selected relative checkpoint without moving archived models."""
    try:
        config = json.loads(SERVING_CONFIG_PATH.read_text(encoding="utf-8"))
        checkpoint = config["selected_checkpoint"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid serving configuration: {SERVING_CONFIG_PATH}") from error
    if not isinstance(checkpoint, str) or not checkpoint.strip():
        raise RuntimeError("selected_checkpoint must be a non-empty string")
    resolved = (ROOT / checkpoint).resolve()
    if ROOT.resolve() not in resolved.parents or resolved.suffix != ".pt":
        raise RuntimeError("selected_checkpoint must be a .pt file inside the project")
    if not resolved.is_file():
        raise RuntimeError(f"Selected checkpoint does not exist: {checkpoint}")
    return checkpoint


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    session_id: str | None = Field(default=None, max_length=128)
    max_new_tokens: int = Field(default=200, ge=1, le=400)
    temperature: float = Field(default=0.6, ge=0.0, le=1.5)
    context_mode: Literal["default", "discord"] = "default"
    system_instruction: str | None = Field(default=None, max_length=1_500)
    discord_context: str | None = Field(default=None, max_length=1_000)
    source: Literal["api", "webclient", "discord"] = "api"


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
        self.shorten_casual_generation = True
        # Build and retain the immutable Unicode metadata once at startup.
        self.emoji_database = emoji_database()
        self.reliable = PublicReliableResponder(ROOT / "data/public_v20_conversations.jsonl")

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
            if any(re.search(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", normalized) for name in names):
                return subject
        # Keep project metadata exact without pretending that an arbitrary
        # made-up suffix is a real released profile.
        unknown_profile = re.search(r"\bchudgpt[ -]([a-z][a-z0-9_-]{1,30})\b", normalized)
        if unknown_profile:
            candidate = unknown_profile.group(1)
            if candidate not in {"and", "or", "is", "family", "model", "project"}:
                return f"unknown:{candidate}"
        if re.search(r"\bwhat is chudgpt\b|\bexplain (?:the )?chudgpt(?: project)?\b|\btell me about (?:the )?chudgpt(?: project)?\b", normalized):
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
        # Identity is stable project metadata, so explicit identity questions
        # always use the verified source. A tiny model can mention the correct
        # name while surrounding it with malformed or invented details.
        if not self.assistance_enabled or subject is None:
            return raw_reply, None
        if subject == "public":
            return (
                f"{PUBLIC_IDENTITY} The currently loaded model has {self.parameters:,} parameters, "
                f"a {self.model.config.context_length}-token context window, and checkpoint step {self.step}."
            ), "stable-public-identity"
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
        # Product/family identity always outranks the unrelated word glossary.
        if self._identity_subject(message) is not None:
            return raw_reply, None
        fact = find_meme_fact(message)
        if fact is None:
            return raw_reply, None
        return fact, "reviewed-meme-context"

    @staticmethod
    def _discord_context_reply(message: str, discord_context: str | None) -> str | None:
        if not discord_context:
            return None
        fields = dict(re.findall(
            r"(?:^|;\s*)(server|channel|speaker|relationship|member_roles|developer_name|developer_mention)=([^;]+)",
            discord_context,
        ))
        normalized = message.lower()
        if re.search(r"\b(?:what|which) server\b|\bwhere are we(?: talking)?\b", normalized) and fields.get("server"):
            if fields["server"].strip().lower() == "direct messages":
                if re.search(r"\b(?:what|which) server\b", normalized):
                    return "This is a private Discord direct message, not a server channel."
                return "We're talking in a private Discord direct message."
            return f"We're talking in the {fields['server']} Discord server."
        if re.search(r"\b(?:who|what) am i\b|\bdo you know me\b", normalized) and fields.get("speaker"):
            relation = fields.get("relationship", "Discord user")
            return f"You're {fields['speaker']}, identified here as {relation}."
        if re.search(r"\b(?:what|which) (?:is |are )?my (?:server )?(?:tag|role|roles)\b", normalized):
            roles = fields.get("member_roles", "none")
            return f"Your Discord server role{'s are' if ',' in roles else ' is'} {roles}."
        if (
            re.search(r"\b(?:who|what) is astra\b|\btell me about astra\b", normalized)
            or re.search(r"\bwho (?:made|created|developed) (?:you|chudgpt)\b", normalized)
            or re.search(r"\bwho is (?:your|the) developer\b", normalized)
        ):
            developer = fields.get("developer_name", "Astra")
            mention = fields.get("developer_mention", "")
            visible_mention = f" ({mention})" if mention and mention != "unavailable" else ""
            return f"{developer}{visible_mention} is ChudGPT's developer and the owner of this Discord bot."
        return None

    def _generate_raw(
        self,
        history: list[dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        system_prompt: str,
    ) -> str:
        """Generate neural candidates and select the least broken relevant reply."""
        current_message = strip_emoji_context(history[-1]["content"])
        structured_request = requests_structured_response(current_message)
        conversational = len(re.findall(r"[a-z0-9']+", current_message.lower())) <= 8 and not structured_request
        if conversational and self.shorten_casual_generation:
            system_prompt += (
                " This is casual conversation. Reply naturally and directly in one to three short sentences. "
                "Do not use numbered steps, bullets, a tutorial, or an unrelated example unless the user asks for one."
            )
            max_new_tokens = min(max_new_tokens, 80)
        _, prompt_ids = build_context_token_ids(
            self.tokenizer, history, self.model.config.context_length,
            system_prompt=system_prompt,
        )
        prompt_tensor = torch.tensor([prompt_ids], device=self.device)
        # Draw several neural candidates, then rank generated text for relevance
        # and basic fluency. The selector never supplies or rewrites an answer.
        sampling_profiles = (
            (max(0.48, temperature - 0.12), 60, 0.90),
            (temperature, 60, 0.90),
            (max(0.72, temperature), 60, 0.90),
            (max(0.86, temperature), 60, 0.90),
            (max(0.54, temperature - 0.04), 40, 0.84),
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
                repetition_penalty=1.16,
                eos_token_id=self.eos_id,
            )[0, len(prompt_ids):].tolist()
            reply = self.tokenizer.decode(output, skip_special_tokens=True).strip()
            if reply:
                candidates.append(reply)
        if candidates:
            prompt = current_message
            previous_replies = [turn["content"] for turn in history if turn["role"] == "assistant"]
            previous_user = next((
                strip_emoji_context(turn["content"])
                for turn in reversed(history[:-1]) if turn["role"] == "user"
            ), "")
            previous_assistant = previous_replies[-1] if previous_replies else ""
            conversation_context = f"{previous_user} {previous_assistant}".strip()
            valid = [candidate for candidate in candidates if assess_generated_reply(
                prompt, candidate, previous_replies, conversation_context
            )[0]]
            if valid:
                return max(valid, key=lambda reply: score_generated_reply(prompt, reply) + self._candidate_score(prompt, reply))
            if previous_assistant and re.fullmatch(
                r"\s*(?:what|what\?|huh|bro(?: what)?|why|what are you talking about)[?!.]*\s*",
                prompt,
                re.I,
            ):
                return "Yeah, that last answer wandered off. I was responding to your previous message, but I clearly missed it."
            # Draw fresh candidates.  A small model can legitimately miss the
            # complete quality gate several times in a row; that must not turn
            # an otherwise healthy API request into HTTP 503.
            for retry in range(8):
                output = generate(
                    self.model,
                    prompt_tensor,
                    max_new_tokens=max(24, min(max_new_tokens, 80)),
                    temperature=min(1.35, max(0.88, temperature) + retry * 0.05),
                    top_k=80,
                    top_p=0.95,
                    repetition_penalty=1.18,
                    eos_token_id=self.eos_id,
                )[0, len(prompt_ids):].tolist()
                retry_reply = self.tokenizer.decode(output, skip_special_tokens=True).strip()
                if retry_reply and assess_generated_reply(
                    prompt, retry_reply, previous_replies, conversation_context
                )[0]:
                    return retry_reply
                if retry_reply:
                    candidates.append(retry_reply)

            # The complete gate is intentionally conservative.  If it rejects
            # every draw, retain its ranking signal but distinguish repairable
            # relevance/style faults from output that must never be exposed.
            # This keeps Public generative and available without allowing
            # prompt/training leaks, malformed text, broken code, or loops.
            never_expose = {
                "prompt-leak", "emoji-context-leak", "training-data-leak", "replacement-character",
                "broken-code-fence", "corrupt-fragment", "repetition-loop",
                "degenerate-repetition", "repeated-clause", "identity-repetition",
                "broken-identity-grammar", "recursive-self-definition",
                "wrong-programming-language", "mixed-programming-languages", "code-only-constraint",
            }
            usable: list[str] = []
            for candidate in candidates:
                _, reasons = assess_generated_reply(
                    prompt, candidate, previous_replies, conversation_context
                )
                if not (never_expose & set(reasons)):
                    usable.append(candidate)
            if usable:
                return max(
                    usable,
                    key=lambda reply: score_generated_reply(prompt, reply)
                    + self._candidate_score(prompt, reply),
                )
            raise RuntimeError("Model produced no displayable output after generation attempts")
        raise RuntimeError("Model produced empty output after generation attempts")

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
        score -= 12.0 if has_structured_list(reply) and not requests_structured_response(message) else 0.0
        if len(re.findall(r"[a-z0-9']+", message_lower)) <= 8 and not requests_structured_response(message):
            score -= max(0, len(reply_words) - 45) * 0.08
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
        context_mode: Literal["default", "discord"] = "default",
        discord_context: str | None = None,
    ) -> tuple[str, str]:
        clean_message = message.strip()
        if not clean_message:
            raise ValueError("message cannot be blank")
        active_session = session_id or uuid.uuid4().hex
        with self.lock:
            history = list(self.sessions.get(active_session, []))
            normalized_message = normalize_user_text(clean_message, include_emoji_hints=False)
            model_message = add_emoji_context(
                normalized_message,
                include_discord=context_mode == "discord",
            )
            history.append({"role": "user", "content": model_message})
            # A 21M model becomes self-contaminating when dozens of its own bad
            # generations remain in view. Keep the four most recent exchanges;
            # this is context selection only and never changes model output.
            generation_history = history[-8:]
            # Deterministic routing must see normalized user text, never the
            # private model-facing annotation appended above.
            instruction_reply = exact_instruction_response(normalized_message)
            arithmetic_reply = exact_math_response(normalized_message)
            reliable_reply = (
                self.reliable.answer(normalized_message, generation_history[:-1])
                if self.reliable is not None
                else None
            )
            discord_reply = self._discord_context_reply(normalized_message, discord_context)
            meme_reply = find_meme_fact(normalized_message)
            emoji_reply = emoji_semantic_response(
                normalized_message,
                include_discord=context_mode == "discord",
            )
            if instruction_reply is not None:
                reply = instruction_reply
                self.last_assistance_reason = "exact-user-instruction"
            elif arithmetic_reply is not None:
                reply = arithmetic_reply
                self.last_assistance_reason = "exact-math"
            elif discord_reply is not None:
                reply = discord_reply
                self.last_assistance_reason = "discord-session-context"
            elif meme_reply is not None and self._identity_subject(normalized_message) is None:
                reply = meme_reply
                self.last_assistance_reason = "reviewed-meme-context"
            elif emoji_reply is not None:
                reply = emoji_reply
                self.last_assistance_reason = "emoji-semantic-context"
            elif reliable_reply is not None:
                reply = reliable_reply
                self.last_assistance_reason = "reviewed-local-response"
            else:
                active_prompt = DISCORD_SYSTEM_PROMPT if context_mode == "discord" else self.system_prompt
                if context_mode == "discord" and discord_context:
                    active_prompt += " Current Discord context: " + discord_context
                raw_reply = self._generate_raw(generation_history, max_new_tokens, temperature, active_prompt)
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


class MusicModelService(PublicModelService):
    """A separately loaded, purely generative Music V1 checkpoint."""

    def __init__(self, checkpoint_path: Path, device_name: str, tokenizer_path: Path | None = None) -> None:
        super().__init__(
            checkpoint_path,
            device_name,
            assistance_enabled=False,
            tokenizer_path=tokenizer_path,
            system_prompt=MUSIC_SYSTEM_PROMPT,
        )
        # Music output is always produced by its checkpoint. It intentionally
        # has no response table, retrieval responder, or canned fallback.
        self.reliable = None
        self.shorten_casual_generation = False
        self.generation_log_path = Path(
            os.getenv("CHUDGPT_MUSIC_LOG_PATH", str(ROOT / "logs" / "music" / "generations.jsonl"))
        )
        self.last_music_metadata: dict[str, object] = {}
        self.allowed_sections = {
            "intro": "Intro", "verse": "Verse", "verse 1": "Verse 1",
            "verse 2": "Verse 2", "verse 3": "Verse 3", "pre-chorus": "Pre-Chorus",
            "chorus": "Chorus", "hook": "Hook", "bridge": "Bridge",
            "breakdown": "Breakdown", "final chorus": "Final Chorus", "outro": "Outro",
        }
        self.overrepresented_music_lines = self._load_overrepresented_lines()

    def chat(
        self,
        message: str,
        session_id: str | None,
        max_new_tokens: int = 420,
        temperature: float = 0.72,
        context_mode: Literal["default", "discord"] = "default",
        discord_context: str | None = None,
        source: Literal["api", "webclient", "discord"] = "api",
    ) -> tuple[str, str]:
        """Generate every Music reply directly from the Music checkpoint.

        Music deliberately bypasses Public's instruction, arithmetic, meme,
        emoji, identity, Discord-context, reviewed-response, and repair-answer
        routers. Conversation history and the fixed layout instruction are
        model input only; no helper is allowed to provide reply text.
        """
        del context_mode, discord_context
        clean_message = message.strip()
        if not clean_message:
            raise ValueError("message cannot be blank")
        active_session = session_id or uuid.uuid4().hex
        with self.lock:
            history = list(self.sessions.get(active_session, []))
            history.append({"role": "user", "content": clean_message})
            generation_history = history[-8:]
            started = time.perf_counter()
            reply = self._generate_raw(
                generation_history,
                max_new_tokens,
                temperature,
                self.system_prompt,
            )
            history.append({"role": "assistant", "content": reply})
            self.sessions[active_session] = history
            self.sessions.move_to_end(active_session)
            while len(self.sessions) > MAX_SESSIONS:
                self.sessions.popitem(last=False)
            self.last_music_metadata["generation_time_ms"] = round((time.perf_counter() - started) * 1000, 2)
            self._log_music_generation(active_session, clean_message, reply, source)
        self.last_assistance_reason = None
        return active_session, reply

    def _generate_raw(
        self,
        history: list[dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        system_prompt: str,
    ) -> str:
        """Generate and rank full neural music drafts without a text fallback.

        Public's general quality gate deliberately rejects long structured
        answers and limits repair attempts to 80 tokens. Those are sensible
        defaults for chat, but they truncate songs. Music instead samples full
        drafts and selects the strongest generated candidate; it never inserts
        or rewrites lyrics.
        """
        current_message = strip_emoji_context(history[-1]["content"])
        previous_music = next(
            (turn["content"] for turn in reversed(history[:-1]) if turn["role"] == "assistant"),
            "",
        )
        previous_title = re.search(r"(?im)^title\s*:\s*([^\n]+)", previous_music)
        previous_style = re.search(r"(?im)^style\s*:\s*([^\n]+)", previous_music)
        recent_replies = [turn["content"] for turn in history[:-1] if turn["role"] == "assistant"]
        if previous_title or previous_style:
            continuity: list[str] = []
            if previous_title:
                continuity.append(f"the established title is {previous_title.group(1).strip()}")
            if previous_style:
                continuity.append(f"the established style is {previous_style.group(1).strip()}")
            system_prompt += (
                " Maintain the current conversation's musical choices: "
                + "; ".join(continuity)
                + ". Do not silently replace them unless the user requests a change."
            )
        _, prompt_ids = build_context_token_ids(
            self.tokenizer,
            history,
            self.model.config.context_length,
            system_prompt=system_prompt,
        )
        prompt_tensor = torch.tensor([prompt_ids], device=self.device)
        requested_tokens = max(160, min(max_new_tokens, 560))
        wants_complete_song = bool(
            re.search(
                r"\b(?:(?:full|complete)(?:\s+(?:original|new))?\s+(?:song|lyrics)|(?:song|lyrics)\s+(?:in\s+)?full)\b",
                current_message,
                re.I,
            )
        )
        minimum_draft_tokens = min(140, requested_tokens - 1) if wants_complete_song else 0
        profiles = ((0.52, 45, 0.86), (0.58, 50, 0.88), (0.64, 60, 0.90),
                    (0.70, 70, 0.92), (0.76, 80, 0.94), (0.82, 90, 0.95))
        candidates: list[tuple[str, list[str]]] = []
        for sample_temperature, top_k, top_p in profiles:
            output = generate(
                self.model,
                prompt_tensor,
                max_new_tokens=requested_tokens,
                temperature=sample_temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=1.16,
                eos_token_id=self.eos_id,
                min_new_tokens=minimum_draft_tokens,
                no_repeat_ngram_size=4,
            )[0, len(prompt_ids):].tolist()
            reply = self.tokenizer.decode(output, skip_special_tokens=True).strip()
            if reply:
                validated, corrections = self._validate_structure(reply)
                candidates.append((validated, corrections))
        if not candidates:
            raise RuntimeError("Music model produced empty output")
        def continuity_score(reply: str) -> float:
            score = self._candidate_score(current_message, reply)
            lowered = reply.lower()
            if previous_title:
                score += 8.0 if previous_title.group(1).strip().lower() in lowered else 0.0
            if previous_style:
                style_terms = set(re.findall(r"[a-z]{4,}", previous_style.group(1).lower()))
                score += min(len(style_terms & set(re.findall(r"[a-z]{4,}", lowered))), 4) * 1.5
            continuity_request = bool(re.search(r"(?i)\b(?:keep|same|remember|remind|revise|rewrite)\b", current_message))
            if not continuity_request:
                candidate_title = re.search(r"(?im)^title\s*:\s*([^\n]+)", reply)
                candidate_style = re.search(r"(?im)^style\s*:\s*([^\n]+)", reply)
                for old_reply in recent_replies:
                    old_title = re.search(r"(?im)^title\s*:\s*([^\n]+)", old_reply)
                    old_style = re.search(r"(?im)^style\s*:\s*([^\n]+)", old_reply)
                    if candidate_title and old_title:
                        score -= 14.0 * self._text_similarity(candidate_title.group(1), old_title.group(1))
                    if candidate_style and old_style:
                        score -= 12.0 * self._text_similarity(candidate_style.group(1), old_style.group(1))
                    old_lines = set(self._lyric_lines(old_reply))
                    score -= len(old_lines & set(self._lyric_lines(reply))) * 2.5
            # This list is derived from prior Music generations at startup,
            # rather than a hand-authored phrase blacklist. It suppresses
            # checkpoint memorization while leaving novel oddness intact.
            score -= len(set(self._lyric_lines(reply)) & self.overrepresented_music_lines) * 4.0
            return score
        scored = [(continuity_score(candidate), candidate, corrections)
                  for candidate, corrections in candidates]
        selected_score, selected, corrections = max(scored, key=lambda item: item[0])
        selected, removed_lines = self._remove_overrepresented_lines(selected)
        if removed_lines:
            corrections = [*corrections, f"removed-overrepresented-lines:{removed_lines}"]
            selected, final_corrections = self._validate_structure(selected)
            corrections.extend(final_corrections)
        title_match = re.search(r"(?im)^title\s*:\s*([^\n]+)", selected)
        style_match = re.search(r"(?im)^style\s*:\s*([^\n]+)", selected)
        selected_lines = self._lyric_lines(selected)
        line_counts = Counter(selected_lines)
        repeated_line_count = sum(count - 1 for count in line_counts.values() if count > 1)
        recent_line_overlap = sum(
            len(set(selected_lines) & set(self._lyric_lines(old_reply)))
            for old_reply in recent_replies
        )
        self.last_music_metadata = {
            "candidate_count": len(candidates),
            "selected_score": round(selected_score, 3),
            "title": title_match.group(1).strip() if title_match else None,
            "style": style_match.group(1).strip() if style_match else None,
            "sections": re.findall(r"(?m)^\[([^]\n]+)\]", selected),
            "output_words": len(re.findall(r"\b\w+\b", selected)),
            "temperature_profiles": [item[0] for item in profiles],
            "repetition_penalty": 1.16,
            "no_repeat_ngram_size": 4,
            "repeated_line_count": repeated_line_count,
            "recent_line_overlap": recent_line_overlap,
            "repetition_detection_triggered": repeated_line_count > 0 or recent_line_overlap > 0,
            "topic_relevance_score": round(self._topic_relevance(current_message, selected), 3),
            "structure_validation_corrections": corrections,
            "title_style_regeneration": False,
        }
        return selected

    def _remove_overrepresented_lines(self, reply: str) -> tuple[str, int]:
        """Remove log-proven memorized lyric lines, never generated answer ideas."""
        kept: list[str] = []
        removed = 0
        for line in reply.splitlines():
            normalized = re.sub(r"\s+", " ", line.strip().lower())
            if normalized in self.overrepresented_music_lines:
                removed += 1
                continue
            kept.append(line)
        return "\n".join(kept).strip(), removed

    def _validate_structure(self, reply: str) -> tuple[str, list[str]]:
        """Normalize only obvious metadata/section defects in neural text."""
        corrections: list[str] = []
        normalized: list[str] = []
        verse_number = 0
        for line in reply.splitlines():
            fixed = line
            if re.match(r"(?i)^\s*(?:static\s+)+title\s*:", fixed):
                fixed = re.sub(r"(?i)^\s*(?:static\s+)+title\s*:", "Title:", fixed)
                corrections.append("normalized-title-label")
            if re.match(r"(?i)^\s*sty(?:p|l|las?|les?)\s*:", fixed):
                fixed = re.sub(r"(?i)^\s*sty(?:p|l|las?|les?)\s*:", "Style:", fixed)
                corrections.append("normalized-style-label")
            if re.match(r"(?i)^\s*style\s*:", fixed):
                cleaned = re.sub(
                    r"(?i),?\s*with a clear pulse and a slightly unwise finale\.?\s*$", "", fixed
                ).rstrip(" ,;.")
                if cleaned != fixed:
                    fixed = cleaned
                    corrections.append("removed-memorized-style-suffix")
            match = re.match(r"^\s*\[([^]\n]+)\]\s*$", fixed)
            if match:
                raw = re.sub(r"\s+", " ", match.group(1).strip().lower())
                canonical = self.allowed_sections.get(raw)
                if canonical is None:
                    fixed = match.group(1).strip()
                    corrections.append(f"unwrapped-unknown-section:{raw}")
                else:
                    if canonical.startswith("Verse"):
                        verse_number += 1
                        canonical = f"Verse {verse_number}"
                    fixed = f"[{canonical}]"
                    if canonical.lower() != raw:
                        corrections.append(f"normalized-section:{raw}->{canonical}")
            normalized.append(fixed)
        section_indexes = [i for i, line in enumerate(normalized) if re.match(r"^\[[^]]+\]$", line.strip())]
        outro_index = next((i for i in section_indexes if normalized[i].strip() == "[Outro]"), None)
        if outro_index is not None and any(i > outro_index for i in section_indexes):
            next_section = next(i for i in section_indexes if i > outro_index)
            outro_block = normalized[outro_index:next_section]
            del normalized[outro_index:next_section]
            while normalized and not normalized[-1].strip():
                normalized.pop()
            normalized.extend(["", *outro_block])
            corrections.append("moved-outro-to-end")
        return "\n".join(normalized).strip(), corrections

    def _load_overrepresented_lines(self) -> set[str]:
        counts: Counter[str] = Counter()
        for path in (ROOT / "reports" / "music_v1_generations.jsonl",
                     ROOT / "logs" / "music" / "generations.jsonl"):
            if not path.is_file():
                continue
            try:
                for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
                    record = json.loads(raw)
                    text = str(record.get("reply", record.get("output", "")))
                    counts.update(set(self._lyric_lines(text)))
            except (OSError, json.JSONDecodeError):
                continue
        return {line for line, count in counts.items() if count >= 4}

    @staticmethod
    def _lyric_lines(text: str) -> list[str]:
        return [re.sub(r"\s+", " ", line.strip().lower()) for line in text.splitlines()
                if len(re.findall(r"[a-z0-9']+", line.lower())) >= 4 and not line.lstrip().startswith("[")]

    @staticmethod
    def _text_similarity(left: str, right: str) -> float:
        left_words = set(re.findall(r"[a-z0-9']+", left.lower()))
        right_words = set(re.findall(r"[a-z0-9']+", right.lower()))
        return len(left_words & right_words) / max(len(left_words | right_words), 1)

    def _log_music_generation(self, session_id: str, prompt: str, reply: str, source: str) -> None:
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "prompt": prompt,
            "reply": reply,
            "source": source,
            "requested_model": MUSIC_MODEL_NAME,
            "actual_model": MUSIC_MODEL_NAME,
            "model": MUSIC_MODEL_NAME,
            "checkpoint": str(self.checkpoint_path),
            **self.last_music_metadata,
        }
        try:
            self.generation_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.generation_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass

    @staticmethod
    def _candidate_score(message: str, reply: str) -> float:
        """Prefer complete on-request songs while ranking only neural output."""
        score = PublicModelService._candidate_score(message, reply)
        request = message.lower()
        wants_song = bool(re.search(r"\b(?:song|music|lyrics)\b", request))
        wants_complete_song = bool(
            re.search(r"\b(?:(?:full|complete)(?:\s+(?:original|new))?\s+(?:song|lyrics)|(?:song|lyrics)\s+(?:in\s+)?full)\b", request)
        )
        wants_fragment = bool(re.search(r"\b(?:only|just)\s+(?:a\s+)?(?:title|style|hook|chorus|verse|bridge|outro)\b", request))
        asks_choice = bool(re.search(r"\b(?:what|which|remind).*(?:title|name|style|genre)\b", request))
        asks_title = bool(re.search(r"\b(?:title|song name|name (?:the|this|my) song)\b", request))
        asks_style = bool(re.search(r"\b(?:style|genre|sound|production)\b", request))
        content_words = {word for word in re.findall(r"[a-z]{4,}", request)
                         if word not in {"write", "make", "song", "music", "lyrics", "full", "complete", "about", "with", "give"}}
        reply_words = set(re.findall(r"[a-z]{4,}", reply.lower()))
        relevance = MusicModelService._topic_relevance(message, reply)
        score += min(len(content_words & reply_words), 6) * 3.0
        score += relevance * 24.0
        if content_words and relevance < 0.08:
            score -= 18.0
        lines = MusicModelService._lyric_lines(reply)
        repeated_line_count = sum(count - 2 for count in __import__("collections").Counter(lines).values() if count > 2)
        score -= repeated_line_count * 5.0
        title_match = re.search(r"(?im)^title\s*:\s*([^\n]+)", reply)
        if title_match:
            title_words = re.findall(r"[a-z0-9]+", title_match.group(1).lower())
            if not title_words or all(len(word) == 1 for word in title_words) or len(set(title_words)) == 1:
                score -= 24.0
            score += MusicModelService._topic_relevance(message, title_match.group(1)) * 10.0
            if re.search(r"[\[\]{}:]|\b(?:title|style|verse|chorus)\b", title_match.group(1), re.I):
                score -= 14.0
        style_match = re.search(r"(?im)^style\s*:\s*([^\n]+)", reply)
        if style_match and len(re.findall(r"[a-z]+", style_match.group(1).lower())) < 4:
            score -= 10.0
        malformed_metadata = bool(re.search(r"(?im)^\s*(?:styp|styl|stylas|styles?)\s*:", reply))
        if malformed_metadata:
            score -= 18.0
        if asks_choice:
            score += 6.0 if re.search(r"(?im)^title\s*:", reply) else -4.0
            score += 6.0 if re.search(r"(?im)^style\s*:", reply) else -4.0
        if wants_song and not wants_fragment:
            has_title = bool(re.search(r"(?im)^title\s*:", reply))
            has_style = bool(re.search(r"(?im)^style\s*:", reply))
            metadata_expected = wants_complete_song or asks_title or asks_style
            score += (8.0 if has_title else -8.0) if metadata_expected else (-3.0 if has_title else 3.0)
            score += (8.0 if has_style else -8.0) if metadata_expected else (-3.0 if has_style else 3.0)
            section_names = set(re.findall(r"(?im)^\[(verse|chorus|bridge|outro)[^]]*\]", reply))
            score += len(section_names) * 3.0
            desired_sections = 3 if wants_complete_song else 2
            score -= max(0, desired_sections - len(section_names)) * 4.0
            repeated_sections = len(re.findall(r"(?im)^\[(?:verse|chorus|bridge|outro)[^]]*\]", reply)) - len(section_names)
            score -= repeated_sections * 2.0
            if wants_complete_song:
                score -= 8.0 if len(reply.split()) < 80 else 0.0
            else:
                score -= 18.0 if len(reply.split()) < 30 else 0.0
                score -= 12.0 if not section_names else 0.0
            score += min(len(reply.split()), 220) * 0.025
        return score

    @staticmethod
    def _topic_relevance(message: str, reply: str) -> float:
        """Estimate lexical/semantic topic coverage without supplying answer text."""
        stop = {"write", "make", "give", "song", "music", "lyrics", "full", "complete",
                "original", "about", "with", "that", "this", "your", "please", "some"}
        request_words = {word for word in re.findall(r"[a-z]{3,}", message.lower()) if word not in stop}
        reply_words = set(re.findall(r"[a-z]{3,}", reply.lower()))
        concept_groups = {
            "water": {"water", "rain", "river", "ocean", "sea", "wave", "waves", "tide", "shore", "flow", "drop", "drown"},
            "chudgpt": {"chudgpt", "model", "token", "prompt", "reply", "answer", "glitch", "code", "machine", "bot", "ai"},
            "yourself": {"chudgpt", "model", "token", "prompt", "reply", "answer", "glitch", "machine", "bot", "ai", "voice"},
            "keyboard": {"keyboard", "key", "keys", "spacebar", "typing", "type", "letter"},
            "thunderstorm": {"thunder", "storm", "lightning", "rain", "cloud", "sky"},
            "microwave": {"microwave", "kitchen", "heat", "beep", "plate", "timer"},
            "space": {"space", "star", "stars", "orbit", "planet", "moon", "galaxy", "rocket"},
            "coding": {"coding", "code", "bug", "debug", "compile", "screen", "keyboard", "program"},
            "robot": {"robot", "metal", "circuit", "servo", "machine", "dance", "dancing"},
        }
        expanded = set(request_words)
        if re.search(r"\b(?:you|yourself)\b", message.lower()):
            expanded |= concept_groups["yourself"]
        for key, values in concept_groups.items():
            if key in request_words or request_words & values:
                expanded |= values
        if not expanded:
            return 0.5
        return len(expanded & reply_words) / max(min(len(expanded), 8), 1)


def create_app(checkpoint: Path, device: str, assistance_enabled: bool = True,
               tokenizer_path: Path | None = None,
               music_checkpoint: Path | None = None) -> FastAPI:
    service = PublicModelService(checkpoint, device, assistance_enabled=assistance_enabled,
                                 tokenizer_path=tokenizer_path)
    music_service = (
        MusicModelService(music_checkpoint, device, tokenizer_path=tokenizer_path)
        if music_checkpoint is not None and music_checkpoint.is_file()
        else None
    )
    app = FastAPI(title="ChudGPT-Public API", version=PUBLIC_VERSION)
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
            "version": "V20",
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
            "assistance_scope": "exact operations, stable project facts, reviewed local responses, and neural candidate quality checks",
            "emoji_awareness": {
                "library": "emoji 2.15.0",
                "unicode_emoji_version": service.emoji_database.max_emoji_version,
                "recognized_sequences": service.emoji_database.sequence_count,
                "discord_custom_emoji": True,
            },
            "music": music_service is not None,
            "available_models": ["ChudGPT-Public"] + ([MUSIC_MODEL_NAME] if music_service else []),
        }

    @app.get("/api/music/status")
    @app.get("/api/music/info")
    def music_info() -> dict[str, object]:
        if music_service is None:
            raise HTTPException(status_code=503, detail="ChudGPT-Public-Music V1 checkpoint is not loaded")
        return {
            "name": MUSIC_MODEL_NAME,
            "version": "V1",
            "model": MUSIC_MODEL_NAME,
            "specialization": "Music / Original Lyrics",
            "status": "online",
            "ready": True,
            "device": str(music_service.device),
            "parameters": music_service.parameters,
            "step": music_service.step,
            "context_length": music_service.model.config.context_length,
            "checkpoint": str(music_service.checkpoint_path.relative_to(ROOT)),
            "music": True,
            "original_lyrics_only": True,
        }

    @app.get("/api")
    def api_index() -> dict[str, object]:
        return {"name": "ChudGPT-Public API", "endpoints": {"chat": "POST /api/chat", "generate": "POST /api/generate", "clear": "POST /api/clear", "info": "GET /api/info"}}

    def run_chat(request: ChatRequest, keep_session: bool) -> dict[str, object]:
        try:
            if request.context_mode == "discord":
                # Only the official bot instruction is accepted. Public API
                # callers cannot replace the protected base system prompt.
                if request.system_instruction != DISCORD_BOT_INSTRUCTION:
                    raise ValueError("Discord context requires the official bot system instruction")
            requested_session = request.session_id if keep_session else uuid.uuid4().hex
            session_id, reply = service.chat(
                request.message, requested_session, request.max_new_tokens, request.temperature,
                request.context_mode, request.discord_context,
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

    @app.post("/api/music/chat")
    def music_chat(request: ChatRequest) -> dict[str, object]:
        if music_service is None:
            raise HTTPException(status_code=503, detail="ChudGPT-Public-Music V1 checkpoint is not loaded")
        try:
            session_id, reply = music_service.chat(
                request.message,
                request.session_id,
                min(request.max_new_tokens, 400),
                request.temperature,
                source=request.source,
            )
        except (ValueError, RuntimeError) as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return {
            "reply": reply,
            "session_id": session_id,
            "step": music_service.step,
            "model": MUSIC_MODEL_NAME,
            "music": True,
        }

    @app.post("/api/music/clear")
    def music_clear(request: ClearRequest) -> dict[str, bool]:
        if music_service is None:
            raise HTTPException(status_code=503, detail="ChudGPT-Public-Music V1 checkpoint is not loaded")
        music_service.clear(request.session_id)
        return {"cleared": True, "music": True}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve raw ChudGPT-Public inference")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint override. By default, use serving_config.json (or CHUDGPT_CHECKPOINT).",
    )
    parser.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    parser.add_argument("--music-checkpoint", default="checkpoints/public_music_v1/best.pt")
    parser.add_argument("--disable-assistance", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--host", default=os.getenv("CHUDGPT_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("CHUDGPT_PORT", "8010")))
    args = parser.parse_args()
    checkpoint = args.checkpoint or os.getenv("CHUDGPT_CHECKPOINT") or selected_checkpoint()
    app = create_app(ROOT / checkpoint, args.device, assistance_enabled=not args.disable_assistance,
                     tokenizer_path=ROOT / args.tokenizer,
                     music_checkpoint=ROOT / args.music_checkpoint)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
