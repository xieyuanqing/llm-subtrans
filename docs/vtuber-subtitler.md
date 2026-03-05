# VTuber-LLM-Subtitler (Two-Pass Pipeline)

This repository now includes an experimental end-to-end pipeline for VTuber replay videos:

1. Download audio and metadata from YouTube
2. Split audio into API-safe chunks (<5MB by default)
3. Transcribe Japanese speech with Whisper-compatible ASR endpoint
4. Pass-1 LLM semantic reorganization (merge fragmented ASR lines)
5. Pass-2 LLM contextual translation (JP -> Simplified Chinese)
6. Build final SRT

Script: `scripts/vtuber_subtitler.py`

## Requirements

- Python environment from this project
- `yt-dlp` binary in PATH
- `ffmpeg` + `ffprobe` binaries in PATH
- ASR endpoint compatible with `/audio/transcriptions`
- LLM endpoint compatible with `/chat/completions`

## Quick Start

```bash
python scripts/vtuber_subtitler.py \
  "https://www.youtube.com/watch?v=<VIDEO_ID>" \
  --output ./output/demo.zh.srt \
  --asr-api-base https://api.groq.com/openai/v1 \
  --asr-api-key "$GROQ_API_KEY" \
  --asr-model whisper-large-v3-turbo \
  --llm-api-base https://api.deepseek.com/v1 \
  --llm-api-key "$DEEPSEEK_API_KEY" \
  --llm-model deepseek-chat \
  --terminology-lock warn \
  --strict-json
```

Intermediates are written to `./workspace/vtuber_subtitler` by default:

- `metadata.json`
- `raw_asr.json`
- `pass1_merged.json`
- `pass2_translated.json`

## Notes

- Pass-1 expects JSON output with `source_ids`, `start`, `end`, `text`.
- Pass-2 enforces id/length mapping to avoid subtitle desync.
- Both passes include explicit JSON schema prompts and strict field validation (`--strict-json`, `--no-strict-json`).
- Terminology lock behavior can be controlled with `--terminology-lock off|warn|strict`.
- Chunk splitting includes small overlap to reduce boundary truncation.
- If a chunk exceeds the size limit, the script attempts low-bitrate re-encode once.
