from io import BytesIO

import pytest
from werkzeug.datastructures import FileStorage

from soundboard import SoundboardController, SoundboardError
from bot import create_health_app


def test_soundboard_upload_list_volume_and_delete(tmp_path) -> None:
    board = SoundboardController(tmp_path / "sounds")
    upload = FileStorage(stream=BytesIO(b"RIFFdemo"), filename="Funny Sound.wav")
    assert board.save_upload(upload) == "Funny_Sound.wav"
    assert board.list_tracks()[0]["name"] == "Funny_Sound.wav"
    assert board.set_volume_percent(135) == 100
    assert board.snapshot()["volume"] == 100
    board.delete_track("Funny_Sound.wav")
    assert board.list_tracks() == []


def test_soundboard_rejects_unsafe_and_unsupported_files(tmp_path) -> None:
    board = SoundboardController(tmp_path / "sounds")
    with pytest.raises(SoundboardError):
        board.save_upload(FileStorage(stream=BytesIO(b"bad"), filename="payload.exe"))
    with pytest.raises(SoundboardError):
        board.resolve_track("../secret.mp3")


def test_soundboard_requires_owner_enablement_before_playback(tmp_path) -> None:
    board = SoundboardController(tmp_path / "sounds")
    assert board.snapshot()["enabled"] is False
    board.configure(10, 20)
    assert board.snapshot()["guild_id"] == 10
    assert board.snapshot()["channel_id"] == 20
    board.disable()
    assert board.snapshot()["enabled"] is False


def test_local_soundboard_page_and_api_session_protection(tmp_path) -> None:
    board = SoundboardController(tmp_path / "sounds")
    app = create_health_app({"discord_ready": False}, board)
    client = app.test_client()
    page = client.get("/soundboard")
    assert page.status_code == 200
    assert b"ChudGPT Soundboard" in page.data
    assert client.get("/soundboard/api/status").status_code == 400
