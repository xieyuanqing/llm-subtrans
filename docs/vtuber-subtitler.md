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
- Local ASR mode: OpenAI-compatible Whisper endpoint (`/audio/transcriptions`)
- Cloudflare ASR mode: Workers AI account + API token
- LLM endpoint compatible with `/chat/completions`

## Intended Workflow

This pipeline is meant to be run by the assistant on the VPS after you provide the video/task.
The operator does not need to remember commands; the assistant chooses the ASR backend and runs the pipeline.

## Reference Commands

### A) Local 4060 Whisper (assistant-run, preferred)

```bash
python scripts/vtuber_subtitler.py \
  "https://www.youtube.com/watch?v=<VIDEO_ID>" \
  --output ./output/demo.zh.srt \
  --asr-provider local \
  --local-asr-api-base "http://100.74.157.37:8000/v1" \
  --local-asr-api-key "$LOCAL_ASR_API_KEY" \
  --local-asr-model whisper-large-v3 \
  --llm-api-base http://localhost:3000/v1 \
  --llm-api-key "$NEWAPI_API_KEY" \
  --llm-model deepseek-ai/DeepSeek-V3.2 \
  --terminology-lock warn \
  --strict-json
```

### B) Cloudflare Whisper (assistant-run, manual fallback)

```bash
python scripts/vtuber_subtitler.py \
  "https://www.youtube.com/watch?v=<VIDEO_ID>" \
  --output ./output/demo.zh.srt \
  --asr-provider cloudflare \
  --cloudflare-account-id "$CLOUDFLARE_ACCOUNT_ID" \
  --cloudflare-api-token "$CLOUDFLARE_API_TOKEN" \
  --cloudflare-asr-model "@cf/openai/whisper-large-v3-turbo" \
  --llm-api-base http://localhost:3000/v1 \
  --llm-api-key "$NEWAPI_API_KEY" \
  --llm-model deepseek-ai/DeepSeek-V3.2 \
  --terminology-lock warn \
  --strict-json
```

Intermediates are written to `./workspace/vtuber_subtitler` by default:

- `metadata.json`
- `raw_asr.json`
- `pass1_merged.json`
- `pass2_translated.json`

## Notes

- ASR provider is manual: choose `--asr-provider local` or `--asr-provider cloudflare`.
- No automatic failover is performed between local and cloudflare in this mode.
- Default LLM path should use your configured New API gateway with `deepseek-ai/DeepSeek-V3.2` unless you explicitly switch it.
- Operationally, this is an assistant-run pipeline on the VPS, not an end-user self-serve CLI workflow.
- Pass-1 expects JSON output with `source_ids`, `start`, `end`, `text`.
- Pass-2 enforces id/length mapping to avoid subtitle desync.
- Both passes include explicit JSON schema prompts and strict field validation (`--strict-json`, `--no-strict-json`).
- Terminology lock behavior can be controlled with `--terminology-lock off|warn|strict`.
- Chunk splitting includes small overlap to reduce boundary truncation.
- If a chunk exceeds the size limit, the script attempts low-bitrate re-encode once.
