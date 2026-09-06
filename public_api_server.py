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
from music_instructions import MUSIC_MODEL_NAME, MUSIC_SYSTEM_PROMPT
from public_identity import project_identity_response
from public_math import exact_math_response

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
    """Serve neural conversation plus narrowly scoped, auditable fact systems."""

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
        # Kept as a compatibility attribute for older callers and response
        # schemas. The legacy broad responder remains disabled. Exact math and
        # immutable identity facts are routed explicitly in chat(), while all
        # unknown and general requests remain neural.
        self.assistance_enabled = False
        self.last_assistance_reason: str | None = None
        self.system_prompt = system_prompt
        self.shorten_casual_generation = True
        # Build and retain the immutable Unicode metadata once at startup.
        self.emoji_database = emoji_database()
        self.reliable = None

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
                "wrong-programming-language", "mixed-programming-languages", "broken-csharp-integrity",
                "code-only-constraint",
                "missing-requested-code", "no-shared-subject", "missing-story-subject",
                "wrong-item-count", "sentence-count-constraint", "forbidden-word-constraint",
                "line-count-constraint", "yes-no-constraint",
                "generic-uncertainty-fallback", "unrelated-topic-starter",
                "unrequested-technical-topic", "technical-reply-to-emotional-prompt",
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
            # Never substitute a canned answer when every neural sample fails.
            # The API reports a generation failure so clients can retry without
            # presenting hand-written text as model output.
            raise RuntimeError("Public V20 did not produce a usable neural reply")
        raise RuntimeError("Public V20 produced empty neural output")

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
            math_reply = exact_math_response(normalized_message)
            if math_reply is not None:
                reply = math_reply
                self.last_assistance_reason = "exact_math"
            else:
                identity_reply = project_identity_response(
                    normalized_message,
                    history[:-1],
                    parameters=self.parameters,
                    context_length=self.model.config.context_length,
                )
                if identity_reply is not None:
                    reply = identity_reply
                    self.last_assistance_reason = "project_identity"
                else:
                    # A 21M model becomes self-contaminating when dozens of its own bad
                    # generations remain in view. Keep the four most recent exchanges;
                    # this is context selection only and never changes model output.
                    generation_history = history[-8:]
                    active_prompt = DISCORD_SYSTEM_PROMPT if context_mode == "discord" else self.system_prompt
                    if context_mode == "discord" and discord_context:
                        active_prompt += " Current Discord context: " + discord_context
                    reply = self._generate_raw(generation_history, max_new_tokens, temperature, active_prompt)
                    self.last_assistance_reason = None
            history.append({"role": "assistant", "content": reply})
            self.sessions[active_session] = history
            self.sessions.move_to_end(active_session)
            while len(self.sessions) > MAX_SESSIONS:
                self.sessions.popitem(last=False)
        return active_session, reply

    def clear(self, session_id: str) -> None:
        with self.lock:
            self.sessions.pop(session_id, None)


class MusicGenerationRejected(RuntimeError):
    """The Music server is healthy, but its neural draft failed quality checks."""


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
        self.overrepresented_music_phrases = self._load_overrepresented_phrases()
        self.generation_profiles = (
            (0.52, 45, 0.86), (0.58, 50, 0.88), (0.64, 60, 0.90),
            (0.70, 70, 0.92), (0.76, 80, 0.94), (0.82, 90, 0.95),
        )

    @staticmethod
    def _classify_music_request(message: str) -> dict[str, object]:
        """Classify requested musical work without supplying answer text."""
        request = re.sub(r"\s+", " ", message.strip().lower())
        asks_short = bool(re.search(
            r"\b(?:short|tiny|quick|small|brief|20[ -]?second|one verse|one chorus|"
            r"four[ -]?line|4[ -]?line)\b", request
        ))
        asks_full = bool(re.search(
            r"\b(?:full|complete|long|extended)\b.*\b(?:song|lyrics)\b|"
            r"\b(?:song|lyrics)\b.*\b(?:in full|complete|extended)\b|"
            r"\b(?:write|make|create|generate)(?: me)? (?:a |an )?(?:original )?song\b",
            request,
        ))
        if re.search(r"\b(?:mash(?:up)?|combine|merge|blend|splice|mix)\b.*\b(?:lyrics?|lines?|verses?|songs?)\b|"
                     r"\b(?:lyrics?|lines?|verses?|songs?)\b.*\b(?:together|mash(?:up)?|combine|merge|blend)\b",
                     request):
            intent = "MASH_LYRICS"
        elif re.search(r"\b(?:titles?|song names?|name ideas?)\b", request):
            intent = "TITLE_IDEAS"
        elif re.search(r"\b(?:styles?|genres?|production ideas?|sound ideas?)\b", request):
            intent = "STYLE_IDEAS"
        elif re.search(r"\b(?:rhyme|rhymes|rhyming)\b", request):
            intent = "RHYME_HELP"
        elif re.search(r"\b(?:rewrite|reword|make this (?:line |lyric )?(?:better|darker|angrier|sadder|"
                       r"more emotional|less cringe|catchier))\b", request):
            intent = "REWRITE_LYRIC"
        elif re.search(r"\b(?:continue|finish|complete this|second verse|next verse|build around)\b", request):
            intent = "CONTINUE_LYRICS"
        elif re.search(r"\b(?:feedback|critique|review|what do you think)\b", request):
            intent = "LYRIC_FEEDBACK"
        else:
            section_match = re.search(
                r"\b(?:intro|pre-chorus|chorus|hook|bridge|breakdown|outro|verse)\b", request
            )
            if section_match and not asks_full:
                intent = section_match.group(0).upper().replace("-", "_")
            elif asks_short:
                intent = "SHORT_SONG"
            elif asks_full:
                intent = "FULL_SONG"
            else:
                intent = "GENERAL_MUSIC_HELP"
        requested_length = "short" if asks_short else (
            "full" if asks_full or intent == "MASH_LYRICS" else "focused"
        )
        return {"intent": intent, "requested_length": requested_length}

    @staticmethod
    def _music_generation_instruction(intent: str, requested_length: str) -> str:
        """Return structure-only guidance; never lyric, title, or style content."""
        instructions = {
            "FULL_SONG": (
                " Generate a genuinely complete original song. Begin with generated Title: and Style: fields, "
                "then use a genre-appropriate sequence of at least five labeled lyric sections, normally including "
                "two verses, a recurring hook or chorus, a contrasting section, and an ending. Aim for 25-60 lyric "
                "lines. Vary structure when the genre calls for it; do not return an idea list or a tiny fragment."
            ),
            "SHORT_SONG": (
                " Generate an intentionally short original song of roughly 4-12 lyric lines using only the few "
                "sections needed. Do not expand it into a full-length song."
            ),
            "MASH_LYRICS": (
                " Mash the lyric material supplied in this conversation into one coherent song or section. Preserve "
                "the user's important lines and meaning, reorder or bridge them when useful, remove accidental "
                "duplication, and keep a consistent voice. Do not replace the source material with an unrelated song."
            ),
            "TITLE_IDEAS": " Generate only intentional, topic-relevant song title options; do not write lyrics.",
            "STYLE_IDEAS": (
                " Generate only distinct, understandable style options describing genre, mood, instrumentation, "
                "tempo, vocal approach, and production where useful; do not write lyrics."
            ),
            "RHYME_HELP": " Give only useful rhyme or replacement-line options matching the supplied meaning and tone.",
            "REWRITE_LYRIC": (
                " Rewrite only the supplied lyric material as requested, preserving its core meaning unless the user "
                "asks for a larger change."
            ),
            "CONTINUE_LYRICS": (
                " Continue the supplied lyrics directly in the same perspective, topic, tone, and rough rhyme pattern. "
                "Do not restart the song or silently replace the supplied lines."
            ),
            "LYRIC_FEEDBACK": " Give concise, specific songwriting feedback on the supplied material; do not replace it.",
            "INTRO": " Generate only the requested intro.",
            "PRE_CHORUS": " Generate only the requested pre-chorus.",
            "CHORUS": " Generate only a memorable chorus centered on the requested subject, with a repeatable hook.",
            "HOOK": " Generate only the requested hook ideas or hook, matching the requested count.",
            "BRIDGE": " Generate only the requested bridge, providing a meaningful contrast to the existing song.",
            "BREAKDOWN": " Generate only the requested breakdown.",
            "OUTRO": " Generate only the requested outro.",
            "VERSE": " Generate only the requested verse, matching any supplied lyrics.",
            "GENERAL_MUSIC_HELP": (
                " Answer the specific music or songwriting request directly. Do not force a full song when the user "
                "asked for advice, an idea, or one component."
            ),
        }
        return instructions.get(intent, instructions["GENERAL_MUSIC_HELP"]) + (
            f" The requested output scope is {requested_length}."
        )

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
            # Older Music builds could persist an empty neural reply after the
            # repetition filter removed every lyric line.  Never feed those
            # poisoned turns back into the conversation formatter.
            history = [
                turn for turn in self.sessions.get(active_session, [])
                if isinstance(turn.get("content"), str) and turn["content"].strip()
            ]
            history.append({"role": "user", "content": clean_message})
            generation_history = history[-8:]
            started = time.perf_counter()
            reply = self._generate_raw(
                generation_history,
                max_new_tokens,
                temperature,
                self.system_prompt,
            )
            if not reply.strip():
                raise RuntimeError("Music model produced empty output after candidate selection")
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
        request_profile = self._classify_music_request(current_message)
        detected_intent = str(request_profile["intent"])
        requested_length = str(request_profile["requested_length"])
        validation_message = current_message
        if detected_intent == "MASH_LYRICS":
            supplied_material = "\n".join(
                turn["content"] for turn in history[:-1]
                if turn["role"] == "user"
            )
            validation_message = supplied_material + "\n" + current_message
        # Reinforce the user's subject inside the model prompt. This supplies no
        # lyric text; it only helps the tiny checkpoint keep its own generation
        # attached to the requested concept instead of drifting to old motifs.
        system_prompt += (
            " Keep every generated music response centered on this exact current request: "
            + current_message[:500]
            + ". Use concrete words or close musical imagery from that subject."
        )
        system_prompt += self._music_generation_instruction(detected_intent, requested_length)
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
        # Build context after intent guidance is complete. Older code appended
        # its full-song guidance after this step, so the model never saw it.
        _, prompt_ids = build_context_token_ids(
            self.tokenizer,
            history,
            self.model.config.context_length,
            system_prompt=system_prompt,
        )
        prompt_tensor = torch.tensor([prompt_ids], device=self.device)
        wants_complete_song = detected_intent in {"FULL_SONG", "MASH_LYRICS"}
        if detected_intent == "FULL_SONG":
            requested_tokens = max(420, min(max_new_tokens, 560))
            minimum_draft_tokens = min(220, requested_tokens - 1)
        elif detected_intent == "MASH_LYRICS":
            requested_tokens = max(320, min(max_new_tokens, 520))
            minimum_draft_tokens = min(120, requested_tokens - 1)
        elif detected_intent == "SHORT_SONG":
            requested_tokens = max(90, min(max_new_tokens, 180))
            minimum_draft_tokens = 30
        elif detected_intent in {"TITLE_IDEAS", "STYLE_IDEAS", "RHYME_HELP", "LYRIC_FEEDBACK"}:
            requested_tokens = max(80, min(max_new_tokens, 180))
            minimum_draft_tokens = 0
        else:
            requested_tokens = max(120, min(max_new_tokens, 300))
            minimum_draft_tokens = 20
        # Long drafts dominate inference cost on the 21M checkpoint. Two
        # diverse full-song samples leave useful choice for ranking without
        # pushing Discord requests toward its 120-second timeout. Shorter
        # requests retain the wider six-profile search.
        profiles = self.generation_profiles[:2] if wants_complete_song else self.generation_profiles
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
            score -= self._overrepresented_phrase_hits(reply, current_message) * 5.0
            return score
        shaped = [item for item in candidates
                  if self._candidate_meets_music_shape(validation_message, item[0])]
        # A tiny checkpoint may end a long draft too early. Retry once as a
        # section-by-section neural composition: the model still generates
        # every title, style, and lyric; code supplies only layout labels.
        if detected_intent in {"FULL_SONG", "MASH_LYRICS"} and not shaped:
            assembled, assembly_corrections = self._assemble_neural_song(
                history, system_prompt, current_message, detected_intent
            )
            if assembled:
                candidates.append((assembled, assembly_corrections))
                if self._candidate_meets_music_shape(validation_message, assembled):
                    shaped.append((assembled, assembly_corrections))
        # Select structurally valid neural generations when any exist.
        selection_pool = shaped or candidates
        scored = [(continuity_score(candidate), candidate, corrections)
                  for candidate, corrections in selection_pool]
        selected_score, selected, corrections = max(scored, key=lambda item: item[0])
        selected, filter_corrections = self._safely_filter_repetition(selected, validation_message)
        corrections = [*corrections, *filter_corrections]
        if wants_complete_song and not self._candidate_meets_music_shape(validation_message, selected):
            # Do not present a malformed fragment as a completed song. Music
            # has no canned answer fallback: a failed quality gate remains an
            # explicit generation failure for the API/client to report.
            raise MusicGenerationRejected("Music model did not produce a complete, relevant neural draft")
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
            "detected_intent": detected_intent,
            "requested_length": requested_length,
            "candidate_count": len(candidates),
            "shape_valid_candidate_count": len(shaped),
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

    def _sample_neural_piece(
        self,
        history: list[dict[str, str]],
        system_prompt: str,
        instruction: str,
        max_tokens: int,
        min_tokens: int = 0,
    ) -> str:
        """Ask the same Music checkpoint for one compositional component."""
        del system_prompt
        # Music V1 was fine-tuned to follow musical work in the user turn.
        # Appending stage work to an already long system prompt made the 21M
        # model ignore the subject and regress to memorized phrases. Replace
        # only the current user turn with the compact stage request while
        # retaining earlier conversation context and the normal Music system
        # prompt. The checkpoint still generates every output token.
        stage_history = [*history[:-1], {"role": "user", "content": instruction}]
        _, prompt_ids = build_context_token_ids(
            self.tokenizer,
            stage_history,
            self.model.config.context_length,
            system_prompt=self.system_prompt,
        )
        prompt_tensor = torch.tensor([prompt_ids], device=self.device)
        output = generate(
            self.model,
            prompt_tensor,
            max_new_tokens=max_tokens,
            temperature=0.64,
            top_k=60,
            top_p=0.90,
            repetition_penalty=1.16,
            eos_token_id=self.eos_id,
            min_new_tokens=min_tokens,
            no_repeat_ngram_size=4,
        )[0, len(prompt_ids):].tolist()
        return self.tokenizer.decode(output, skip_special_tokens=True).strip()

    @staticmethod
    def _piece_content(text: str) -> str:
        """Keep model-generated content while removing conflicting wrappers."""
        lines = []
        for line in text.splitlines():
            if re.match(r"(?i)^\s*(?:title|style|styp|styl|styse|styles?|genre|hook)\s*:", line):
                continue
            if re.match(r"^\s*\[[^]]+\]\s*$", line):
                continue
            cleaned = line.strip()
            if not cleaned:
                continue
            # Numbered option lists and code fragments are common failure
            # modes for the tiny checkpoint and are not lyric lines. This
            # only removes wrappers; it never supplies replacement lyrics.
            if re.match(r"^\d+[.)]\s+", cleaned) or re.search(r"(?:==|\bdef\s+|\breturn\s+|\bimport\s+)", cleaned):
                continue
            lines.append(cleaned.lstrip("-*•> "))
        return "\n".join(lines).strip()

    @staticmethod
    def _mash_source_content(history: list[dict[str, str]]) -> str:
        """Extract user-provided lyric material; never invent source lines."""
        user_turns = [turn["content"].strip() for turn in history[:-1]
                      if turn["role"] == "user" and turn["content"].strip()]
        if not user_turns:
            return ""
        source = user_turns[-1]
        source = re.sub(
            r"(?is)^.*?(?:lyric scraps?|lyrics?|lines?)\s*(?:are|:)\s*[:\-]?\s*",
            "",
            source,
            count=1,
        ).strip()
        fragments = [fragment.strip(" \t-*'\"") for fragment in re.split(r"\s*(?:/|\n+)\s*", source)]
        return "\n".join(fragment for fragment in fragments if fragment)[:800]

    def _assemble_neural_song(
        self,
        history: list[dict[str, str]],
        system_prompt: str,
        current_message: str,
        intent: str,
    ) -> tuple[str, list[str]]:
        """Compose a complete result from model-generated parts only."""
        metadata_options = [
            self._sample_neural_piece(
                history,
                system_prompt,
                "Generate only a new Title: line and detailed Style: line for this exact request: "
                f"{current_message[:300]}. Do not write lyrics or alternatives.",
                80,
                10,
            )
            for _ in range(3)
        ]

        def metadata_score(option: str) -> float:
            title_match = re.search(r"(?im)^title\s*:\s*([^\n]+)", option)
            style_match = re.search(r"(?im)^style\s*:\s*([^\n]+)", option)
            score = 0.0
            if title_match:
                title_value = title_match.group(1).strip()
                score += 8.0 if not re.search(r"[\[\]{}:]", title_value) else -12.0
                score += self._topic_relevance(current_message, title_value) * 8.0
            if style_match:
                style_value = style_match.group(1).strip()
                style_words = len(re.findall(r"[a-z]+", style_value.lower()))
                score += 8.0 if style_words >= 3 else -8.0
                score -= 8.0 if re.search(r"[\[\]{}]", style_value) else 0.0
            return score

        metadata = max(metadata_options, key=metadata_score)
        title = re.search(r"(?im)^title\s*:\s*([^\n]+)", metadata)
        style = re.search(r"(?im)^style\s*:\s*([^\n]+)", metadata)
        raw_metadata_lines = [line.strip() for line in metadata.splitlines() if line.strip()]
        title_text = title.group(1).strip() if title else (raw_metadata_lines[0] if raw_metadata_lines else "")
        style_text = style.group(1).strip() if style else (
            raw_metadata_lines[1] if len(raw_metadata_lines) > 1 else ""
        )
        if not title_text or not style_text:
            return "", ["neural-assembly-metadata-empty"]

        section_sets = (
            ("Verse 1", "Pre-Chorus", "Chorus", "Verse 2", "Bridge", "Final Chorus", "Outro"),
            ("Intro", "Verse 1", "Hook", "Verse 2", "Bridge", "Final Chorus", "Outro"),
            ("Verse 1", "Chorus", "Verse 2", "Breakdown", "Bridge", "Final Chorus", "Outro"),
        )
        layout_index = sum(ord(character) for character in current_message) % len(section_sets)
        section_names = section_sets[layout_index]
        pieces: list[str] = []
        mash_source = "\n".join(
            turn["content"] for turn in history[:-1]
            if turn["role"] == "user"
        )[-800:]
        preserved_source = self._mash_source_content(history) if intent == "MASH_LYRICS" else ""
        for section_name in section_names:
            if intent == "MASH_LYRICS" and section_name in {"Verse 1", "Intro"} and preserved_source:
                pieces.append(f"[{section_name}]\n{preserved_source}")
                continue
            source_guidance = (
                " Preserve and recombine this user-supplied lyric material wherever it fits: "
                + mash_source
                if intent == "MASH_LYRICS" else ""
            )
            prior_lyrics = "\n\n".join(pieces)[-1400:]
            novelty_guidance = (
                " These sections are already written:\n" + prior_lyrics
                + "\nWrite new imagery and phrasing for the next section. Do not repeat or closely "
                  "paraphrase any existing line, except when intentionally developing a short hook."
                if prior_lyrics else ""
            )
            options: list[str] = []
            # This path already follows several rejected whole-song drafts.
            # Two candidates per section avoids accepting the first collapsed
            # motif without multiplying worst-case latency unnecessarily.
            for _attempt in range(2):
                generated = self._sample_neural_piece(
                    history,
                    system_prompt,
                    f"For this composition stage, generate only the lyric lines for [{section_name}]. "
                    f"Keep them tightly relevant to this request: {current_message[:300]}.{source_guidance}"
                    f"{novelty_guidance} "
                    "Do not output a title, style, explanation, option list, or another section label.",
                    90 if section_name not in {"Pre-Chorus", "Outro"} else 65,
                    24,
                )
                content = self._piece_content(generated)
                if content:
                    options.append(content)
            prior_lines = set(self._lyric_lines("\n".join(pieces)))
            content = max(
                options,
                key=lambda option: (
                    self._topic_relevance(current_message, option) * 20.0
                    - len(prior_lines & set(self._lyric_lines(option))) * 6.0
                    - self._overrepresented_phrase_hits(option, current_message) * 5.0
                ),
                default="",
            )
            if content:
                pieces.append(f"[{section_name}]\n{content}")
        if len(pieces) < 5:
            return "", ["neural-assembly-too-few-sections"]
        assembled = f"Title: {title_text}\nStyle: {style_text}\n\n" + "\n\n".join(pieces)
        validated, corrections = self._validate_structure(assembled)
        return validated, ["neural-section-assembly", *corrections]

    def _safely_filter_repetition(self, reply: str, prompt: str = "") -> tuple[str, list[str]]:
        """Suppress memorized lines without destroying the neural draft."""
        filtered, intra_removed = self._remove_intra_song_repetition(reply)
        filtered, removed = self._remove_overrepresented_lines(filtered, prompt)
        filtered = self._remove_empty_sections(filtered)
        filter_corrections: list[str] = []
        if intra_removed:
            filter_corrections.append(f"removed-intra-song-repetitions:{intra_removed}")
        if not removed:
            return filtered, filter_corrections
        if (
            self._candidate_meets_music_shape(prompt, reply)
            and not self._candidate_meets_music_shape(prompt, filtered)
        ):
            # Keep intra-song duplicate removal even when removing learned log
            # phrases would make a complete draft too short.
            if intra_removed:
                intra_only, _ = self._remove_intra_song_repetition(reply)
                return intra_only, [*filter_corrections, "overrepresented-filter-reverted-shape-loss"]
            return reply, ["repetition-filter-reverted-shape-loss"]
        remaining_lyrics = self._lyric_lines(filtered)
        if len(re.findall(r"\b\w+\b", filtered)) < 8 or not remaining_lyrics:
            return reply, ["repetition-filter-reverted-destructive"]
        validated, structure_corrections = self._validate_structure(filtered)
        return validated, [
            *filter_corrections,
            f"removed-overrepresented-lines:{removed}",
            *structure_corrections,
        ]

    @classmethod
    def _remove_intra_song_repetition(cls, reply: str) -> tuple[str, int]:
        """Remove repeated or near-duplicate lyric lines within one draft."""
        kept: list[str] = []
        lyric_lines: list[str] = []
        removed = 0
        for line in reply.splitlines():
            normalized = re.sub(r"\s+", " ", line.strip().lower())
            words = re.findall(r"[a-z0-9']+", normalized)
            is_metadata = bool(re.match(r"(?i)^\s*(?:title|style)\s*:", line))
            is_section = bool(re.match(r"^\s*\[[^]]+\]\s*$", line))
            if len(words) >= 4 and not is_metadata and not is_section:
                repeated = any(
                    normalized == old
                    or cls._text_similarity(normalized, old) >= 0.65
                    for old in lyric_lines
                )
                if repeated:
                    removed += 1
                    continue
                lyric_lines.append(normalized)
            kept.append(line)
        return "\n".join(kept).strip(), removed

    @staticmethod
    def _remove_empty_sections(reply: str) -> str:
        """Drop labels whose generated section contains no lyric text."""
        blocks = re.split(r"(?m)(^\[[^]\n]+\]\s*$)", reply)
        kept = [blocks[0]]
        for index in range(1, len(blocks), 2):
            label = blocks[index]
            body = blocks[index + 1] if index + 1 < len(blocks) else ""
            if re.search(r"[A-Za-z0-9]", body):
                kept.extend((label, body))
        return "".join(kept).strip()

    def _remove_overrepresented_lines(self, reply: str, prompt: str = "") -> tuple[str, int]:
        """Remove exact or paraphrased log-proven memorized lyric lines."""
        kept: list[str] = []
        removed = 0
        prompt_normalized = re.sub(r"\s+", " ", prompt.strip().lower())
        for line in reply.splitlines():
            normalized = re.sub(r"\s+", " ", line.strip().lower())
            repeated_phrase = any(
                phrase in normalized and phrase not in prompt_normalized
                for phrase in getattr(self, "overrepresented_music_phrases", set())
            )
            if normalized in self.overrepresented_music_lines or repeated_phrase:
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
            cleaned_memorized_suffix = re.sub(
                r"(?i),?\s*with a clear pulse and a slightly unwise finale\.?\s*$",
                "",
                fixed,
            )
            if cleaned_memorized_suffix != fixed:
                fixed = cleaned_memorized_suffix.rstrip(" ,;.")
                corrections.append("removed-memorized-style-suffix")
                if not fixed:
                    continue
            if re.match(r"(?i)^\s*(?:static\s+)+title\s*:", fixed):
                fixed = re.sub(r"(?i)^\s*(?:static\s+)+title\s*:", "Title:", fixed)
                corrections.append("normalized-title-label")
            if re.match(r"(?i)^\s*sty(?:p|l|las?|les?)\s*:", fixed):
                fixed = re.sub(r"(?i)^\s*sty(?:p|l|las?|les?)\s*:", "Style:", fixed)
                corrections.append("normalized-style-label")
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

    def _load_overrepresented_phrases(self) -> set[str]:
        """Learn recurring 3-5 word phrase families from private Music logs."""
        counts: Counter[str] = Counter()
        for path in (ROOT / "reports" / "music_v1_generations.jsonl",
                     ROOT / "logs" / "music" / "generations.jsonl"):
            if not path.is_file():
                continue
            try:
                for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
                    record = json.loads(raw)
                    text = str(record.get("reply", record.get("output", "")))
                    phrases: set[str] = set()
                    for line in self._lyric_lines(text):
                        words = re.findall(r"[a-z0-9']+", line)
                        for width in (3, 4, 5):
                            for index in range(len(words) - width + 1):
                                phrase_words = words[index:index + width]
                                if sum(len(word) >= 4 for word in phrase_words) >= 2:
                                    phrases.add(" ".join(phrase_words))
                    counts.update(phrases)
            except (OSError, json.JSONDecodeError):
                continue
        # Prefer the shortest meaningful learned form. This catches a model
        # swapping one adjective (for example, tiny -> little) while repeating
        # the same memorized phrase family.
        frequent = sorted(
            (phrase for phrase, count in counts.items() if count >= 4),
            key=lambda phrase: (len(phrase.split()), phrase),
        )
        selected: list[str] = []
        for phrase in frequent:
            if not any(shorter in phrase for shorter in selected):
                selected.append(phrase)
        return set(selected)

    @staticmethod
    def _candidate_meets_music_shape(message: str, reply: str) -> bool:
        """Check requested song shape without supplying any lyric content."""
        request = message.lower()
        profile = MusicModelService._classify_music_request(message)
        intent = str(profile["intent"])
        full = intent == "FULL_SONG"
        asks_song = bool(re.search(r"\b(?:song|lyrics|music)\b", request))
        asks_fragment = bool(re.search(r"\b(?:chorus|hook|verse|bridge|outro)\b", request))
        sections = re.findall(
            r"(?im)^\[(?:intro|verse(?:\s+\d+)?|pre-chorus|chorus|hook|bridge|breakdown|final chorus|outro)\]$",
            reply,
        )
        words = len(re.findall(r"\b\w+\b", reply))
        numbered_list = len(re.findall(r"(?m)^\s*\d+[.)]\s+", reply)) >= 3
        if numbered_list and intent not in {"TITLE_IDEAS", "STYLE_IDEAS", "RHYME_HELP"}:
            return False
        if intent in {"TITLE_IDEAS", "STYLE_IDEAS", "RHYME_HELP", "LYRIC_FEEDBACK"}:
            return words >= 3 and not sections
        if full:
            metadata_lines = re.findall(r"(?im)^(?:title|style)\s*:\s*\S.*$", reply)
            section_blocks = re.findall(
                r"(?ims)^\[(?:intro|verse(?:\s+\d+)?|pre-chorus|chorus|hook|bridge|breakdown|final chorus|outro)\]\s*"
                r"(.+?)(?=^\[[^]\n]+\]|\Z)",
                reply,
            )
            meaningful_sections = [
                block for block in section_blocks
                if len(re.findall(r"\b\w+\b", block)) >= 4
                and not re.search(r"(?im)^\s*(?:title|style|styp|styl|styse|styles?|genre)\s*:", block)
            ]
            topic_words = {
                word for word in re.findall(r"[a-z]{4,}", request)
                if word not in {"write", "make", "create", "generate", "song", "lyrics", "full", "complete", "about", "with", "original"}
            }
            # A full song must cover at least half of the meaningful subject
            # concepts. The older 0.08 threshold let one generic word such as
            # "machine" approve a song that completely omitted "garden".
            topic_ok = not topic_words or MusicModelService._topic_relevance(message, reply) >= 0.5
            return (
                bool(re.search(r"(?im)^title\s*:\s*\S", reply))
                and bool(re.search(r"(?im)^style\s*:\s*\S", reply))
                and len(sections) >= 5
                and len(meaningful_sections) >= 5
                and len(metadata_lines) == 2
                and words >= 90
                and topic_ok
            )
        if intent == "SHORT_SONG":
            return bool(sections) and len(sections) <= 3 and 8 <= words <= 120
        if intent == "MASH_LYRICS":
            ignored = {
                "mash", "mashup", "combine", "merge", "blend", "splice", "mix",
                "these", "those", "this", "that", "lyrics", "lyric", "lines", "line",
                "verse", "verses", "songs", "song", "together", "into", "complete",
                "coherent", "with", "from", "about", "please", "make", "write",
            }
            source_words = {
                word for word in re.findall(r"[a-z0-9']{4,}", request)
                if word not in ignored
            }
            reply_words = set(re.findall(r"[a-z0-9']{4,}", reply.lower()))
            required_overlap = min(2, len(source_words))
            preserves_source = not source_words or len(source_words & reply_words) >= required_overlap
            return len(sections) >= 5 and words >= 55 and preserves_source
        if asks_fragment:
            return bool(sections) and words >= 8
        if asks_song:
            return bool(sections) and words >= 20
        return words >= 2

    def _overrepresented_phrase_hits(self, reply: str, prompt: str) -> int:
        normalized_reply = re.sub(r"\s+", " ", reply.lower())
        normalized_prompt = re.sub(r"\s+", " ", prompt.lower())
        return sum(
            phrase in normalized_reply and phrase not in normalized_prompt
            for phrase in getattr(self, "overrepresented_music_phrases", set())
        )

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
        profile = MusicModelService._classify_music_request(message)
        intent = str(profile["intent"])
        wants_complete_song = intent == "FULL_SONG"
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
            desired_sections = 5 if wants_complete_song else (1 if intent == "SHORT_SONG" else 2)
            score -= max(0, desired_sections - len(section_names)) * 4.0
            repeated_sections = len(re.findall(r"(?im)^\[(?:verse|chorus|bridge|outro)[^]]*\]", reply)) - len(section_names)
            score -= repeated_sections * 2.0
            if wants_complete_song:
                score -= 20.0 if len(reply.split()) < 90 else 0.0
                score += min(len(reply.split()), 320) * 0.035
            elif intent == "SHORT_SONG":
                score -= 16.0 if len(reply.split()) < 12 else 0.0
                score -= 14.0 if len(reply.split()) > 120 else 0.0
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
        if not request_words:
            return 0.5
        covered = 0
        for word in request_words:
            if word in {"you", "yourself"}:
                aliases = concept_groups["yourself"]
            else:
                aliases = concept_groups.get(word, {word})
            if aliases & reply_words:
                covered += 1
        return covered / max(min(len(request_words), 8), 1)


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
    @app.get("/api/models/public")
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
            "exact_math": True,
            "identity_grounding": True,
            "fallbacks": False,
            "grounded_systems": ["exact_math", "project_identity"],
            "generation_policy": "neural generation with narrow exact-math and immutable project-identity grounding; unknown and general requests remain neural",
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
    @app.get("/api/models/music")
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
            "fallbacks": False,
            "generation_policy": "neural-only; invalid samples are rejected, never replaced",
        }

    @app.get("/api")
    def api_index() -> dict[str, object]:
        return {"name": "ChudGPT-Public API", "endpoints": {"models": "GET /api/models", "chat": "POST /api/chat", "generate": "POST /api/generate", "clear": "POST /api/clear", "info": "GET /api/info"}}

    @app.get("/api/models")
    def models() -> dict[str, object]:
        entries: list[dict[str, object]] = [
            {
                "id": "public",
                "name": "ChudGPT-Public V20",
                "family": "public",
                "ready": True,
                "endpoints": {
                    "info": "GET /api/models/public",
                    "chat": "POST /api/models/public/chat",
                    "generate": "POST /api/models/public/generate",
                    "clear": "POST /api/models/public/clear",
                },
            }
        ]
        if music_service is not None:
            entries.append({
                "id": "music",
                "name": MUSIC_MODEL_NAME,
                "family": "public",
                "ready": True,
                "endpoints": {
                    "info": "GET /api/models/music",
                    "chat": "POST /api/models/music/chat",
                    "generate": "POST /api/models/music/generate",
                    "clear": "POST /api/models/music/clear",
                },
            })
        return {"models": entries, "count": len(entries)}

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
                "raw_model_generation": service.last_assistance_reason is None,
                "assistance_used": service.last_assistance_reason is not None,
                "assistance_reason": service.last_assistance_reason}

    @app.post("/api/chat")
    @app.post("/api/models/public/chat")
    def chat(request: ChatRequest) -> dict[str, object]:
        return run_chat(request, True)

    @app.post("/api/generate")
    @app.post("/api/models/public/generate")
    def generate_once(request: ChatRequest) -> dict[str, object]:
        return run_chat(request, False)

    @app.post("/api/clear")
    @app.post("/api/models/public/clear")
    def clear(request: ClearRequest) -> dict[str, bool]:
        service.clear(request.session_id)
        return {"cleared": True}

    def run_music_chat(request: ChatRequest, keep_session: bool) -> dict[str, object]:
        if music_service is None:
            raise HTTPException(status_code=503, detail="ChudGPT-Public-Music V1 checkpoint is not loaded")
        try:
            requested_session = request.session_id if keep_session else uuid.uuid4().hex
            session_id, reply = music_service.chat(
                request.message,
                requested_session,
                min(request.max_new_tokens, 400),
                request.temperature,
                source=request.source,
            )
        except MusicGenerationRejected as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except (ValueError, RuntimeError) as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        if not keep_session:
            music_service.clear(session_id)
        return {
            "reply": reply,
            "session_id": session_id,
            "step": music_service.step,
            "model": MUSIC_MODEL_NAME,
            "music": True,
        }

    @app.post("/api/music/chat")
    @app.post("/api/models/music/chat")
    def music_chat(request: ChatRequest) -> dict[str, object]:
        return run_music_chat(request, True)

    @app.post("/api/models/music/generate")
    def music_generate(request: ChatRequest) -> dict[str, object]:
        return run_music_chat(request, False)

    @app.post("/api/music/clear")
    @app.post("/api/models/music/clear")
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
