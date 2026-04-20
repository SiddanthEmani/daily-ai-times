"""Optional audio summary stage via google-genai TTS. No-op if key missing."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from src.pipeline.models import ScoredArticle

log = logging.getLogger(__name__)

_AUDIO_DIR = Path(__file__).resolve().parents[2] / "src" / "frontend" / "assets" / "audio"


async def render_brief(
    headlines: list[ScoredArticle],
    *,
    voice: str = "Puck",
    output_dir: Path = _AUDIO_DIR,
) -> Path | None:
    """Synthesize a short audio brief for top headlines. Returns written path or None."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key or not headlines:
        log.info("audio brief skipped (no key or no headlines)")
        return None
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        log.warning("google-genai not installed; skipping audio")
        return None

    script = "Daily AI Times brief. " + " ".join(
        f"Story {i + 1}: {s.article.title}."
        for i, s in enumerate(headlines[:5])
    )
    client = genai.Client(api_key=api_key)
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash-preview-tts",
        contents=script,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                )
            ),
        ),
    )
    data = response.candidates[0].content.parts[0].inline_data.data
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "brief.wav"
    out.write_bytes(data)
    log.info("wrote audio brief to %s (%d bytes)", out, len(data))
    return out
