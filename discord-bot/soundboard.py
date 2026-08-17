"""Local, administrator-controlled Discord soundboard support."""

from __future__ import annotations

import asyncio
import shutil
import threading
from pathlib import Path
from typing import Any

import discord
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".m4a", ".flac", ".webm", ".aac"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


def ffmpeg_executable() -> str | None:
    """Find system FFmpeg or the bundled imageio-ffmpeg binary."""
    system_binary = shutil.which("ffmpeg")
    if system_binary:
        return system_binary
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError, OSError):
        return None


class SoundboardError(RuntimeError):
    """A user-facing soundboard failure."""


class SoundboardController:
    def __init__(self, audio_dir: Path) -> None:
        self.audio_dir = audio_dir
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = False
        self.guild_id: int | None = None
        self.channel_id: int | None = None
        self.volume = 0.65
        self.current_track: str | None = None
        self._lock = threading.RLock()

    def configure(self, guild_id: int, channel_id: int) -> None:
        with self._lock:
            self.enabled = True
            self.guild_id = guild_id
            self.channel_id = channel_id

    def disable(self) -> None:
        with self._lock:
            self.enabled = False
            self.current_track = None

    def set_volume_percent(self, value: int | float) -> int:
        percent = max(0, min(100, int(float(value))))
        with self._lock:
            self.volume = percent / 100.0
        return percent

    def list_tracks(self) -> list[dict[str, Any]]:
        tracks: list[dict[str, Any]] = []
        for path in sorted(self.audio_dir.iterdir(), key=lambda item: item.name.lower()):
            if path.is_file() and path.suffix.lower() in ALLOWED_AUDIO_EXTENSIONS:
                tracks.append({"name": path.name, "bytes": path.stat().st_size})
        return tracks

    def resolve_track(self, filename: str) -> Path:
        safe_name = secure_filename(filename)
        if not safe_name or Path(safe_name).suffix.lower() not in ALLOWED_AUDIO_EXTENSIONS:
            raise SoundboardError("Unsupported or invalid audio filename.")
        path = (self.audio_dir / safe_name).resolve()
        if path.parent != self.audio_dir.resolve() or not path.is_file():
            raise SoundboardError("That sound does not exist.")
        return path

    def save_upload(self, upload: FileStorage) -> str:
        safe_name = secure_filename(upload.filename or "")
        if not safe_name or Path(safe_name).suffix.lower() not in ALLOWED_AUDIO_EXTENSIONS:
            allowed = ", ".join(sorted(ALLOWED_AUDIO_EXTENSIONS))
            raise SoundboardError(f"Choose a supported audio file: {allowed}.")
        destination = self.audio_dir / safe_name
        upload.save(destination)
        if destination.stat().st_size > MAX_UPLOAD_BYTES:
            destination.unlink(missing_ok=True)
            raise SoundboardError("Audio files must be 50 MB or smaller.")
        return safe_name

    def delete_track(self, filename: str) -> None:
        path = self.resolve_track(filename)
        path.unlink()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "guild_id": self.guild_id,
                "channel_id": self.channel_id,
                "volume": round(self.volume * 100),
                "current_track": self.current_track,
                "ffmpeg_available": ffmpeg_executable() is not None,
                "tracks": self.list_tracks(),
            }

    async def _voice_client(self, client: discord.Client) -> discord.VoiceClient:
        with self._lock:
            enabled, guild_id, channel_id = self.enabled, self.guild_id, self.channel_id
        if not enabled or guild_id is None or channel_id is None:
            raise SoundboardError("Enable the soundboard from Discord while an admin is in voice first.")
        guild = client.get_guild(guild_id)
        if guild is None:
            raise SoundboardError("The configured Discord server is unavailable.")
        channel = guild.get_channel(channel_id)
        if not isinstance(channel, (discord.VoiceChannel, discord.StageChannel)):
            raise SoundboardError("The configured voice channel no longer exists.")
        voice = guild.voice_client
        try:
            if voice is None:
                voice = await channel.connect(timeout=15.0, reconnect=True)
            elif voice.channel.id != channel.id:
                await voice.move_to(channel)
        except (discord.ClientException, RuntimeError, asyncio.TimeoutError) as error:
            raise SoundboardError(f"Discord voice connection failed: {error}") from error
        return voice

    async def play(self, client: discord.Client, filename: str) -> None:
        executable = ffmpeg_executable()
        if executable is None:
            raise SoundboardError("FFmpeg is not installed or is missing from PATH.")
        path = self.resolve_track(filename)
        voice = await self._voice_client(client)
        if voice.is_playing() or voice.is_paused():
            voice.stop()
        try:
            source = discord.PCMVolumeTransformer(
                discord.FFmpegPCMAudio(str(path), executable=executable), volume=self.volume
            )
        except (OSError, discord.ClientException) as error:
            raise SoundboardError(f"Audio decoder failed: {error}") from error
        with self._lock:
            self.current_track = path.name

        def finished(error: Exception | None) -> None:
            with self._lock:
                self.current_track = None

        voice.play(source, after=finished)

    async def stop(self, client: discord.Client) -> None:
        guild = client.get_guild(self.guild_id) if self.guild_id else None
        if guild and guild.voice_client and (guild.voice_client.is_playing() or guild.voice_client.is_paused()):
            guild.voice_client.stop()
        with self._lock:
            self.current_track = None

    async def leave(self, client: discord.Client) -> None:
        guild = client.get_guild(self.guild_id) if self.guild_id else None
        if guild and guild.voice_client:
            await guild.voice_client.disconnect(force=True)
        self.disable()


def submit_to_discord(loop: asyncio.AbstractEventLoop, coroutine: Any, timeout: float = 20.0) -> Any:
    """Safely submit a voice coroutine from Flask's worker thread."""
    return asyncio.run_coroutine_threadsafe(coroutine, loop).result(timeout=timeout)
