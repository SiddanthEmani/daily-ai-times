"""Podcast audio generation with the Gemini API.

Gemini's free tier keeps changing: Pro models dropped to a quota of 0 and new
Flash / Flash-Lite / TTS generations ship every few months. Hard-coding one
model ID silently breaks the podcast (the failure is only logged), so this
module asks the API which models the key can use, ranks them newest-first and
falls through to the next one on quota or not-found errors.

Pin a model with GEMINI_SCRIPT_MODEL / GEMINI_TTS_MODEL if needed.

Run standalone (needs GEMINI_API_KEY and src/backend/api/latest.json):
    python -m src.backend.podcast
"""
import json
import logging
import os
import re
import sys
import wave
from pathlib import Path
from typing import Iterable, List, Optional

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

LATEST_JSON = Path('src/backend/api/latest.json')
AUDIO_FILE = Path('src/frontend/assets/audio/latest-podcast.wav')

# Tried first, in order, when the API lists them; anything else matching the
# patterns below is tried afterwards, newest version first.
PREFERRED_SCRIPT_MODELS = [
    'gemini-3.8-flash', 'gemini-3.7-flash', 'gemini-3.5-flash-lite',
    'gemini-3.1-flash-lite', 'gemini-2.5-flash', 'gemini-2.5-flash-lite',
]
PREFERRED_TTS_MODELS = [
    'gemini-3.8-flash-tts', 'gemini-3.8-flash-lite-tts',
    'gemini-3.1-flash-tts-preview', 'gemini-2.5-flash-preview-tts',
]

# Plain Flash / Flash-Lite text models, e.g. gemini-3.5-flash-lite or
# gemini-3.1-flash-lite-preview. Pro is excluded (not on the free tier) as are
# image/audio/live/tts variants.
_SCRIPT_RE = re.compile(r'^gemini-(\d+(?:\.\d+)?)-flash(?:-lite)?(?:-preview(?:-[\d-]+)?)?$')
_TTS_RE = re.compile(r'^gemini-(\d+(?:\.\d+)?)-[\w.-]*tts[\w.-]*$')

SPEAKERS = (('Jane', 'Kore'), ('Joe', 'Puck'))


def _version(name: str) -> float:
    m = re.match(r'^gemini-(\d+(?:\.\d+)?)', name)
    return float(m.group(1)) if m else 0.0


def _rank(available: Iterable[str], preferred: List[str], pattern: re.Pattern, override: Optional[str]) -> List[str]:
    """Order candidate models: override, preferred (if listed), then newest matches.

    Stable IDs sort ahead of previews and Flash ahead of Flash-Lite within a version.
    """
    available = set(available)
    ordered = [override] if override else []
    ordered += [m for m in preferred if m in available or not available]
    extra = sorted(
        (m for m in available if pattern.match(m)),
        key=lambda m: (-_version(m), 'preview' in m, 'lite' in m, m),
    )
    ordered += extra
    seen, result = set(), []
    for m in ordered:
        if m not in seen:
            seen.add(m)
            result.append(m)
    return result


def _list_models(client: genai.Client) -> List[str]:
    """Model IDs the key can call with generateContent; empty if listing fails."""
    names = []
    try:
        for model in client.models.list():
            actions = getattr(model, 'supported_actions', None) or []
            if actions and 'generateContent' not in actions:
                continue
            names.append((model.name or '').removeprefix('models/'))
    except Exception as e:
        logger.warning(f"Could not list Gemini models, using defaults: {e}")
    return names


def _is_retryable(err: Exception) -> bool:
    """Errors that mean 'this model is unavailable to us', so try the next one."""
    if isinstance(err, ValueError):  # empty / malformed response
        return True
    code = getattr(err, 'code', None)
    return code in (400, 403, 404, 429, 500, 503) or 'RESOURCE_EXHAUSTED' in str(err)


def _try_models(models: List[str], label: str, call):
    errors = []
    for model in models:
        try:
            result = call(model)
            logger.info(f"✅ {label} generated with {model}")
            return result
        except Exception as e:
            if not _is_retryable(e):
                raise
            summary = str(e).split('\n')[0][:200]
            logger.warning(f"⚠️ {label} model {model} unavailable: {summary}")
            errors.append(f"{model}: {summary}")
    raise RuntimeError(f"No usable {label} model. Tried: " + ' | '.join(errors or ['none']))


def _build_script_prompt(articles: list) -> str:
    stories = []
    for i, article in enumerate(articles[:5]):
        title = article.get('title', '')
        description = (article.get('description') or article.get('content') or '')[:500]
        stories.append(
            f"Story {i+1}: {title}\nSource: {article.get('source', '')} | "
            f"Category: {article.get('category', '')}\nSummary: {description}")
    content_text = '\n\n'.join(stories)
    return f"""Create a concise 2-3 minute news podcast script with two speakers alternating.
Format as dialogue between Jane and Joe. Every line must start with "Jane:" or "Joe:".
Use plain text only: no markdown, no stage directions, no sound effects.

Open with:
Jane: Welcome to today's AI news update. I'm Jane with the latest developments.
Joe: And I'm Joe. Let's dive into today's top stories.

Cover 3 key stories, alternating speakers, 1-2 sentences each per story, with smooth transitions.

End with:
Jane: That's today's AI update.
Joe: Thanks for listening. See you next time.

Keep the total script under 200 words.

Content to cover:
{content_text}"""


def _clean_script(text: str) -> str:
    """Keep only speaker lines and strip markdown the model may add (e.g. **Jane:**)."""
    lines = []
    for line in text.splitlines():
        line = line.replace('*', '').replace('_', ' ').strip()
        if re.match(r'^(Jane|Joe)\s*:', line):
            lines.append(line)
    return '\n'.join(lines) or text.strip()


def _sample_rate(mime_type: Optional[str]) -> int:
    m = re.search(r'rate=(\d+)', mime_type or '')
    return int(m.group(1)) if m else 24000


def generate_podcast(api_key: str, latest_json: Path = LATEST_JSON, audio_file: Path = AUDIO_FILE) -> Path:
    """Write a two-speaker podcast WAV for the latest articles. Raises on failure."""
    with open(latest_json, 'r') as f:
        articles = json.load(f).get('articles', [])
    if not articles:
        raise ValueError("No articles found for audio generation")

    client = genai.Client(api_key=api_key)
    available = _list_models(client)
    if available:
        logger.info(f"Gemini models available: {len(available)} "
                    f"(tts: {', '.join(sorted(m for m in available if 'tts' in m)) or 'none'})")

    script_models = _rank(available, PREFERRED_SCRIPT_MODELS, _SCRIPT_RE, os.getenv('GEMINI_SCRIPT_MODEL'))
    tts_models = _rank(available, PREFERRED_TTS_MODELS, _TTS_RE, os.getenv('GEMINI_TTS_MODEL'))

    def write_script(model):
        response = client.models.generate_content(model=model, contents=_build_script_prompt(articles))
        if not response or not response.text:
            raise ValueError(f"Empty script response from {model}")
        return response.text

    script = _clean_script(_try_models(script_models, 'Podcast script', write_script))
    logger.debug(f"Generated script preview: {script[:1000]}...")

    speech_config = types.SpeechConfig(
        multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
            speaker_voice_configs=[
                types.SpeakerVoiceConfig(
                    speaker=speaker,
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)))
                for speaker, voice in SPEAKERS
            ]))
    tts_prompt = f"TTS the following conversation between Jane and Joe:\n{script}"

    def synthesize(model):
        response = client.models.generate_content(
            model=model,
            contents=tts_prompt,
            config=types.GenerateContentConfig(response_modalities=["AUDIO"], speech_config=speech_config))
        for candidate in (response.candidates or [])[:1]:
            for part in (candidate.content.parts if candidate.content else None) or []:
                if part.inline_data and part.inline_data.data:
                    return part.inline_data
        raise ValueError(f"No inline audio data in TTS response from {model}")

    audio = _try_models(tts_models, 'Podcast audio', synthesize)

    audio_file.parent.mkdir(parents=True, exist_ok=True)
    if (audio.mime_type or '').startswith(('audio/wav', 'audio/x-wav')):
        audio_file.write_bytes(audio.data)
    else:
        # Raw 16-bit mono PCM (audio/L16;codec=pcm;rate=24000)
        with wave.open(str(audio_file), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(_sample_rate(audio.mime_type))
            wf.writeframes(audio.data)

    size = audio_file.stat().st_size
    if size < 1000:
        raise ValueError(f"Generated audio file too small: {size} bytes")
    logger.info(f"✅ Podcast audio generated successfully: {audio_file} ({size:,} bytes)")
    return audio_file


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    key = os.getenv('GEMINI_API_KEY')
    if not key:
        sys.exit("GEMINI_API_KEY is not set")
    try:
        generate_podcast(key)
    except Exception as e:
        logger.error(f"❌ Podcast generation failed: {e}")
        sys.exit(1)
