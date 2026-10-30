"""Shared helpers for the Experiment 8 notebooks: audio loading, Whisper transcription, WER, TTS, and a cached Gemini chat call.

Gemini answers are stored in llm_cache.json, keyed by model, messages and settings, so re-running a notebook costs no API calls.
Delete the file to query the API afresh.
"""
import os, re, io, json, time, hashlib, string, threading
import numpy as np
import soundfile as sf
import librosa
import jiwer
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

load_dotenv(find_dotenv(usecwd=True))
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
SR = 16000                                      # Whisper always works on 16 kHz mono audio
CACHE_PATH = "llm_cache.json"
_cache = json.load(open(CACHE_PATH)) if os.path.exists(CACHE_PATH) else {}
_lock = threading.Lock()
_client = None


# ------------------------------------------------------------------ audio
def load_audio(path, sr=SR):
    """Read any audio file the soundfile library understands and return mono float32 at 16 kHz."""
    a, file_sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    return librosa.resample(a, orig_sr=file_sr, target_sr=sr) if file_sr != sr else a


def add_noise(audio, snr_db, seed=0):
    """Mix white noise into the audio at a chosen signal-to-noise ratio in decibels."""
    rng = np.random.default_rng(seed)
    p_signal = np.mean(audio ** 2)
    p_noise = p_signal / (10 ** (snr_db / 10))
    return (audio + rng.normal(0, np.sqrt(p_noise), audio.shape)).astype("float32")


# ------------------------------------------------------------------ ASR
def transcribe(model, audio, **kw):
    """Run faster-whisper on an array. Returns (text, segments, info, seconds)."""
    t = time.time()
    segments, info = model.transcribe(audio, **kw)
    segments = list(segments)                       # transcription is lazy: the work happens while iterating
    text = " ".join(s.text.strip() for s in segments).strip()
    return text, segments, info, time.time() - t


_NORM = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip()])


def norm_text(s):
    """Lowercase and strip punctuation so that 'Hello, World.' and 'hello world' compare equal."""
    s = s.replace("-", " ")                          # treat hyphens as word breaks: human-in-the-loop = human in the loop
    return _NORM(s) if s.strip() else ""


def wer(refs, hyps):
    """Corpus word error rate after normalisation: (substitutions + deletions + insertions) / reference words."""
    return jiwer.wer([norm_text(r) for r in refs], [norm_text(h) if h.strip() else "<empty>" for h in hyps])


# ------------------------------------------------------------------ LLM
def client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["GEMINI_API_KEY"], base_url=GEMINI_URL, timeout=90, max_retries=0)
    return _client


def chat(messages, model="gemini-2.5-flash-lite", max_tokens=400, temperature=0.0):
    """One cached Gemini chat call. Returns dict(answer, seconds, cached)."""
    key = hashlib.sha256(json.dumps([model, messages, max_tokens, temperature]).encode()).hexdigest()
    if key in _cache:
        return {**_cache[key], "cached": True}
    for attempt in range(6):
        try:
            t = time.time()
            r = client().chat.completions.create(model=model, messages=messages, max_tokens=max_tokens,
                                                 temperature=temperature)
            out = dict(answer=(r.choices[0].message.content or "").strip(), seconds=round(time.time() - t, 2))
            with _lock:
                _cache[key] = out
                json.dump(_cache, open(CACHE_PATH, "w"))
            return {**out, "cached": False}
        except Exception:
            if attempt == 5:
                raise
            time.sleep(4 * (attempt + 1))


# ------------------------------------------------------------------ TTS
def tts_espeak(text, path, rate=170, voice="en"):
    """Offline text-to-speech with the espeak-ng engine (install: apt/pacman 'espeak-ng'). rate is words per minute. Returns seconds taken."""
    import subprocess
    t = time.time()
    subprocess.run(["espeak-ng", "-v", voice, "-s", str(rate), "-w", path, text], check=True)
    return time.time() - t


def tts_gtts(text, path, lang="en", tld="co.in"):
    """Google Translate text-to-speech (needs internet, no key). tld picks the accent: co.in is Indian English."""
    from gtts import gTTS
    t = time.time()
    gTTS(text, lang=lang, tld=tld).save(path)
    return time.time() - t


def tts_elevenlabs(text, path, voice_id="JBFqnCBsd6RMkjVDRZzb", model_id="eleven_flash_v2_5"):
    """ElevenLabs neural TTS. Needs ELEVENLABS_API_KEY in .env. Returns seconds taken."""
    from elevenlabs.client import ElevenLabs
    t = time.time()
    c = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
    audio = c.text_to_speech.convert(text=text, voice_id=voice_id, model_id=model_id, output_format="mp3_44100_128")
    with open(path, "wb") as f:
        for chunk in audio:
            f.write(chunk)
    return time.time() - t
