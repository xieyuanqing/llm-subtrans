from __future__ import annotations

import argparse
import base64
import json
import logging
import math
import os
import pathlib
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any
import re

import requests

from scripts.subtrans_common import InitLogger


class VTuberSubtitlerError(Exception):
    """Raised when the VTuber subtitling pipeline fails."""


@dataclass
class AudioChunk:
    """Description of a chunked audio file."""
    index: int
    path: pathlib.Path
    source_start: float
    source_end: float
    keep_start: float
    keep_end: float


@dataclass
class Segment:
    """A subtitle segment with time range and text."""
    id: int
    start: float
    end: float
    text: str
    source_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize segment to dictionary."""
        return {
            "id": self.id,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "source_ids": self.source_ids,
        }


@dataclass
class PipelineConfig:
    """Runtime configuration for the two-pass subtitler."""
    url: str
    output: pathlib.Path
    workspace: pathlib.Path
    asr_provider: str
    asr_api_base: str
    asr_api_key: str
    asr_model: str
    cloudflare_account_id: str
    llm_api_base: str
    llm_api_key: str
    llm_model: str
    max_chunk_mb: float = 5.0
    overlap_seconds: float = 0.5
    min_chunk_seconds: float = 90.0
    download_audio_format: str = "ba"
    pass1_batch_size: int = 120
    pass2_batch_size: int = 24
    pass2_context_lines: int = 5
    request_timeout: float = 180.0
    retry_count: int = 3
    retry_backoff_seconds: float = 1.8
    glossary_path: pathlib.Path|None = None
    terminology_lock: str = "warn"
    strict_json: bool = True
    title_override: str|None = None
    description_override: str|None = None
    channel_override: str|None = None


class OpenAICompatClient:
    """Thin client for OpenAI-compatible ASR and chat endpoints."""

    def __init__(
        self,
        api_base: str,
        api_key: str,
        timeout: float,
        retries: int,
        backoff_seconds: float,
    ):
        self.api_base = api_base.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries
        self.backoff_seconds = backoff_seconds

    def transcribe_audio(
        self,
        file_path: pathlib.Path,
        model: str,
        prompt: str|None,
    ) -> list[dict[str, Any]]:
        """Call /audio/transcriptions and return segment dictionaries."""
        url = f"{self.api_base}/audio/transcriptions"
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

        payload = {
            "model": model,
            "response_format": "verbose_json",
            "language": "ja",
            "timestamp_granularities[]": "segment",
        }
        if prompt:
            payload["prompt"] = prompt

        for attempt in range(1, self.retries + 1):
            with open(file_path, 'rb') as audio_file:
                files = {
                    "file": (file_path.name, audio_file, "audio/m4a"),
                }
                try:
                    response = requests.post(
                        url,
                        headers=headers,
                        data=payload,
                        files=files,
                        timeout=self.timeout,
                    )
                    response.raise_for_status()
                    data = response.json()
                    segments = data.get("segments")
                    if isinstance(segments, list):
                        return [segment for segment in segments if isinstance(segment, dict)]

                    text = str(data.get("text") or "").strip()
                    if text:
                        return [{"start": 0.0, "end": 0.0, "text": text}]

                    return []
                except (requests.RequestException, ValueError) as error:
                    if attempt >= self.retries:
                        raise VTuberSubtitlerError(f"ASR request failed: {error}") from error
                    sleep_seconds = self.backoff_seconds ** attempt
                    logging.warning("ASR request failed, retrying in %.1fs (%s)", sleep_seconds, error)
                    time.sleep(sleep_seconds)

        return []

    def chat_json(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> Any:
        """Call /chat/completions and parse JSON content from model output."""
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        for attempt in range(1, self.retries + 1):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                choices = data.get("choices")
                if not isinstance(choices, list) or not choices:
                    raise VTuberSubtitlerError("LLM response did not include choices")

                message = choices[0].get("message") if isinstance(choices[0], dict) else None
                content = message.get("content") if isinstance(message, dict) else None
                if not isinstance(content, str) or not content.strip():
                    raise VTuberSubtitlerError("LLM response had empty content")

                return ParseJsonFromText(content)
            except (requests.RequestException, ValueError, VTuberSubtitlerError) as error:
                if attempt >= self.retries:
                    raise VTuberSubtitlerError(f"LLM request failed: {error}") from error
                sleep_seconds = self.backoff_seconds ** attempt
                logging.warning("LLM request failed, retrying in %.1fs (%s)", sleep_seconds, error)
                time.sleep(sleep_seconds)

        raise VTuberSubtitlerError("LLM request failed after retries")


class CloudflareWhisperClient:
    """ASR client for Cloudflare Workers AI whisper models."""

    def __init__(
        self,
        api_base: str,
        account_id: str,
        api_token: str,
        timeout: float,
        retries: int,
        backoff_seconds: float,
    ):
        self.api_base = api_base.rstrip('/')
        self.account_id = account_id
        self.api_token = api_token
        self.timeout = timeout
        self.retries = retries
        self.backoff_seconds = backoff_seconds

    def transcribe_audio(
        self,
        file_path: pathlib.Path,
        model: str,
        prompt: str|None,
    ) -> list[dict[str, Any]]:
        """Call Cloudflare AI run endpoint for whisper transcription."""
        url = BuildCloudflareAsrUrl(self.api_base, self.account_id, model)
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

        audio_base64 = base64.b64encode(file_path.read_bytes()).decode("utf-8")
        payload: dict[str, Any] = {
            "audio": audio_base64,
            "task": "transcribe",
            "language": "ja",
        }
        if prompt:
            payload["initial_prompt"] = prompt

        for attempt in range(1, self.retries + 1):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code >= 400:
                    raise VTuberSubtitlerError(
                        "Cloudflare ASR request failed "
                        f"({response.status_code}): {response.text[:500]}"
                    )

                data = response.json()
                result = data.get("result") if isinstance(data, dict) else None
                if isinstance(result, dict):
                    data = result

                segments = data.get("segments") if isinstance(data, dict) else None
                if isinstance(segments, list):
                    return [segment for segment in segments if isinstance(segment, dict)]

                text = str(data.get("text") or "").strip() if isinstance(data, dict) else ""
                if text:
                    return [{"start": 0.0, "end": 0.0, "text": text}]

                return []
            except (requests.RequestException, ValueError, VTuberSubtitlerError) as error:
                if attempt >= self.retries:
                    raise VTuberSubtitlerError(f"Cloudflare ASR request failed: {error}") from error
                sleep_seconds = self.backoff_seconds ** attempt
                logging.warning("Cloudflare ASR request failed, retrying in %.1fs (%s)", sleep_seconds, error)
                time.sleep(sleep_seconds)

        return []


class VTuberSubtitler:
    """End-to-end implementation of the two-pass VTuber subtitle pipeline."""

    def __init__(self, config: PipelineConfig):
        self.config = config

        if config.asr_provider == "local":
            self.asr_client = OpenAICompatClient(
                api_base=config.asr_api_base,
                api_key=config.asr_api_key,
                timeout=config.request_timeout,
                retries=config.retry_count,
                backoff_seconds=config.retry_backoff_seconds,
            )
        elif config.asr_provider == "cloudflare":
            self.asr_client = CloudflareWhisperClient(
                api_base=config.asr_api_base,
                account_id=config.cloudflare_account_id,
                api_token=config.asr_api_key,
                timeout=config.request_timeout,
                retries=config.retry_count,
                backoff_seconds=config.retry_backoff_seconds,
            )
        else:
            raise VTuberSubtitlerError(f"Unsupported ASR provider: {config.asr_provider}")

        self.llm_client = OpenAICompatClient(
            api_base=config.llm_api_base,
            api_key=config.llm_api_key,
            timeout=config.request_timeout,
            retries=config.retry_count,
            backoff_seconds=config.retry_backoff_seconds,
        )

    def run(self) -> pathlib.Path:
        """Execute all stages and return output SRT path."""
        EnsureBinaryInstalled("yt-dlp")
        EnsureBinaryInstalled("ffmpeg")
        EnsureBinaryInstalled("ffprobe")

        self.config.workspace.mkdir(parents=True, exist_ok=True)

        logging.info("Phase 1/5: download audio and metadata")
        audio_path, metadata = self.download_audio_and_metadata()
        self.write_json(self.config.workspace / "metadata.json", metadata)

        logging.info("Phase 1/5: split audio into safe chunks")
        chunks = self.split_audio(audio_path)
        if not chunks:
            raise VTuberSubtitlerError("No audio chunks were generated")

        logging.info("Phase 2/5: ASR transcription (%d chunks)", len(chunks))
        raw_segments = self.transcribe_chunks(chunks, metadata)
        if not raw_segments:
            raise VTuberSubtitlerError("ASR returned no subtitle segments")
        self.write_json(self.config.workspace / "raw_asr.json", [s.to_dict() for s in raw_segments])

        logging.info("Phase 3/5: semantic reorganization pass")
        merged = self.semantic_reorganize(raw_segments)
        if not merged:
            raise VTuberSubtitlerError("Pass-1 did not return any merged segments")
        self.write_json(self.config.workspace / "pass1_merged.json", [s.to_dict() for s in merged])

        logging.info("Phase 4/5: contextual translation pass")
        translated = self.contextual_translate(merged, metadata)
        if not translated:
            raise VTuberSubtitlerError("Pass-2 did not return translated segments")
        self.write_json(self.config.workspace / "pass2_translated.json", [s.to_dict() for s in translated])

        logging.info("Phase 5/5: build SRT output")
        self.config.output.parent.mkdir(parents=True, exist_ok=True)
        self.config.output.write_text(BuildSrtText(translated), encoding="utf-8")

        logging.info("Pipeline completed: %s", self.config.output)
        return self.config.output

    def download_audio_and_metadata(self) -> tuple[pathlib.Path, dict[str, Any]]:
        """Download source audio with yt-dlp and capture metadata."""
        metadata_command = [
            "yt-dlp",
            "--no-playlist",
            "--dump-single-json",
            self.config.url,
        ]
        metadata_result = subprocess.run(
            metadata_command,
            capture_output=True,
            text=True,
            check=False,
        )
        if metadata_result.returncode != 0:
            raise VTuberSubtitlerError(
                f"Failed to fetch metadata via yt-dlp: {metadata_result.stderr.strip()}"
            )

        try:
            metadata = json.loads(metadata_result.stdout)
        except json.JSONDecodeError as error:
            raise VTuberSubtitlerError("yt-dlp metadata output was not valid JSON") from error

        output_template = self.config.workspace / "source.%(ext)s"
        download_command = [
            "yt-dlp",
            "--no-playlist",
            "-f",
            self.config.download_audio_format,
            "-x",
            "--audio-format",
            "m4a",
            "--audio-quality",
            "0",
            "-o",
            str(output_template),
            "--print",
            "after_move:filepath",
            self.config.url,
        ]

        download_result = subprocess.run(
            download_command,
            capture_output=True,
            text=True,
            check=False,
        )
        if download_result.returncode != 0:
            raise VTuberSubtitlerError(
                f"Failed to download audio via yt-dlp: {download_result.stderr.strip()}"
            )

        path_lines = [line.strip() for line in download_result.stdout.splitlines() if line.strip()]
        if not path_lines:
            raise VTuberSubtitlerError("yt-dlp did not output downloaded file path")

        audio_path = pathlib.Path(path_lines[-1]).resolve()
        if not audio_path.exists():
            raise VTuberSubtitlerError(f"Downloaded audio file not found: {audio_path}")

        return audio_path, {
            "title": self.config.title_override or str(metadata.get("title") or ""),
            "description": self.config.description_override or str(metadata.get("description") or ""),
            "channel": self.config.channel_override or str(metadata.get("uploader") or ""),
            "webpage_url": str(metadata.get("webpage_url") or self.config.url),
        }

    def split_audio(self, source_audio: pathlib.Path) -> list[AudioChunk]:
        """Split source audio into chunks below max chunk size."""
        max_bytes = int(self.config.max_chunk_mb * 1024 * 1024)
        source_size = source_audio.stat().st_size
        total_duration = ProbeDuration(source_audio)

        if total_duration <= 0:
            raise VTuberSubtitlerError("Could not determine audio duration")

        if source_size <= max_bytes:
            return [
                AudioChunk(
                    index=1,
                    path=source_audio,
                    source_start=0.0,
                    source_end=total_duration,
                    keep_start=0.0,
                    keep_end=total_duration,
                )
            ]

        bytes_per_second = source_size / total_duration
        chunk_duration = max(
            self.config.min_chunk_seconds,
            math.floor((max_bytes / max(bytes_per_second, 1.0)) * 0.9),
        )

        chunks_dir = self.config.workspace / "chunks"
        if chunks_dir.exists():
            shutil.rmtree(chunks_dir)
        chunks_dir.mkdir(parents=True, exist_ok=True)

        chunks: list[AudioChunk] = []
        cursor = 0.0
        index = 1

        while cursor < total_duration:
            keep_start = cursor
            keep_end = min(total_duration, cursor + chunk_duration)
            source_start = max(0.0, keep_start - self.config.overlap_seconds)
            source_end = min(total_duration, keep_end + self.config.overlap_seconds)

            chunk_path = chunks_dir / f"chunk_{index:04d}.m4a"
            self.extract_audio_slice(source_audio, chunk_path, source_start, source_end)

            if chunk_path.stat().st_size > max_bytes:
                self.extract_audio_slice(
                    source_audio,
                    chunk_path,
                    source_start,
                    source_end,
                    reencode=True,
                )

            if chunk_path.stat().st_size > max_bytes:
                raise VTuberSubtitlerError(
                    f"Chunk {chunk_path.name} exceeded {self.config.max_chunk_mb}MB even after reencode"
                )

            chunks.append(
                AudioChunk(
                    index=index,
                    path=chunk_path,
                    source_start=source_start,
                    source_end=source_end,
                    keep_start=keep_start,
                    keep_end=keep_end,
                )
            )

            if keep_end >= total_duration:
                break

            cursor = keep_end
            index += 1

        return chunks

    def extract_audio_slice(
        self,
        source_audio: pathlib.Path,
        destination: pathlib.Path,
        start_seconds: float,
        end_seconds: float,
        reencode: bool = False,
    ) -> None:
        """Extract one audio slice via ffmpeg."""
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-ss",
            f"{max(start_seconds, 0.0):.3f}",
            "-to",
            f"{max(end_seconds, 0.0):.3f}",
            "-i",
            str(source_audio),
            "-vn",
        ]

        if reencode:
            command.extend(["-ac", "1", "-ar", "16000", "-b:a", "64k", "-c:a", "aac"])
        else:
            command.extend(["-c", "copy"])

        command.append(str(destination))

        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise VTuberSubtitlerError(f"ffmpeg slice extraction failed: {result.stderr.strip()}")

        if not destination.exists() or destination.stat().st_size == 0:
            raise VTuberSubtitlerError(f"ffmpeg produced empty chunk: {destination}")

    def transcribe_chunks(
        self,
        chunks: list[AudioChunk],
        metadata: dict[str, Any],
    ) -> list[Segment]:
        """Run ASR for each chunk and map local timestamps back to source timeline."""
        initial_prompt = BuildAsrPrompt(metadata)
        global_segments: list[Segment] = []
        next_id = 1

        for chunk in chunks:
            logging.info("ASR chunk %d/%d: %s", chunk.index, len(chunks), chunk.path.name)
            local_segments = self.asr_client.transcribe_audio(
                file_path=chunk.path,
                model=self.config.asr_model,
                prompt=initial_prompt,
            )

            for item in local_segments:
                local_start = float(item.get("start") or 0.0)
                local_end = float(item.get("end") or local_start)
                text = str(item.get("text") or "").strip()
                if not text:
                    continue

                absolute_start = chunk.source_start + max(local_start, 0.0)
                absolute_end = chunk.source_start + max(local_end, local_start)

                midpoint = (absolute_start + absolute_end) / 2
                if midpoint < chunk.keep_start or midpoint > chunk.keep_end:
                    continue

                global_segments.append(
                    Segment(
                        id=next_id,
                        start=round(max(0.0, absolute_start), 3),
                        end=round(max(absolute_end, absolute_start), 3),
                        text=text,
                    )
                )
                next_id += 1

        if not global_segments:
            return []

        global_segments.sort(key=lambda segment: (segment.start, segment.end, segment.id))
        return global_segments

    def semantic_reorganize(self, raw_segments: list[Segment]) -> list[Segment]:
        """Pass-1: build subtitle-ready Japanese segments with rule-first refinement."""
        id_lookup = {segment.id: segment for segment in raw_segments}

        merged = self.rule_based_initial_segments(raw_segments, id_lookup)
        merged = self.refine_oversized_segments(merged, id_lookup)
        merged = SplitOversizedMergedSegments(merged, id_lookup, max_duration=9.5, max_chars=48)
        merged = NormalizeSegmentIds(merged)
        return merged

    def rule_based_initial_segments(
        self,
        raw_segments: list[Segment],
        id_lookup: dict[int, Segment],
    ) -> list[Segment]:
        """Run the broad pass-1 merge, then immediately enforce rule-based limits."""
        merged: list[Segment] = []
        next_id = 1

        for batch in ChunkList(raw_segments, self.config.pass1_batch_size):
            payload = [
                {
                    "id": segment.id,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                }
                for segment in batch
            ]
            batch_id_set = {segment.id for segment in batch}

            system_prompt = (
                "You are a strict subtitle editor for Japanese speech transcription. "
                "Merge fragmented ASR pieces into grammatical Japanese sentences without changing factual meaning. "
                "Return JSON only."
            )
            user_prompt = (
                "Input JSON array:\n"
                f"{json.dumps(payload, ensure_ascii=False)}\n\n"
                "Output schema:\n"
                f"{BuildPass1SchemaText()}\n\n"
                "Rules:\n"
                "1) Keep chronological order.\n"
                "2) source_ids must use only ids from this batch.\n"
                "3) Split on topic shifts and comment-reading turns.\n"
                "4) Prefer half-sentence units around 、 。 ？ ！.\n"
                "5) Return JSON only."
            )

            response_json = self.llm_client.chat_json(
                model=self.config.llm_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
            )

            parsed_items = NormalizeArrayPayload(response_json)
            for item in parsed_items:
                if not ValidatePass1Item(item, strict=self.config.strict_json):
                    continue

                source_ids = ParseSourceIds(item.get("source_ids"), list(batch_id_set))
                if not source_ids:
                    continue

                source_segments = [id_lookup[source_id] for source_id in source_ids if source_id in batch_id_set]
                if not source_segments:
                    continue

                source_segments.sort(key=lambda segment: (segment.start, segment.end, segment.id))
                text = str(item.get("text") or "").strip()
                if not text:
                    continue

                merged.append(
                    Segment(
                        id=next_id,
                        start=source_segments[0].start,
                        end=source_segments[-1].end,
                        text=text,
                        source_ids=[segment.id for segment in source_segments],
                    )
                )
                next_id += 1

        return SplitOversizedMergedSegments(merged, id_lookup, max_duration=9.5, max_chars=48)

    def refine_oversized_segments(
        self,
        merged_segments: list[Segment],
        id_lookup: dict[int, Segment],
    ) -> list[Segment]:
        """Re-run pass-1 only for oversized segments to save tokens."""
        refined: list[Segment] = []
        oversized = [
            segment for segment in merged_segments
            if IsOversizedSegment(segment, max_duration=10.5, max_chars=52)
        ]
        refined_map = self.reorganize_oversized_segments(oversized, id_lookup)

        for segment in merged_segments:
            replacement = refined_map.get(segment.id)
            if replacement:
                refined.extend(replacement)
            else:
                refined.append(segment)

        return EnsureNonOverlappingSourceCoverage(refined, id_lookup)

    def reorganize_oversized_segments(
        self,
        segments: list[Segment],
        id_lookup: dict[int, Segment],
    ) -> dict[int, list[Segment]]:
        """Ask the model to split only the segments that still violate hard limits."""
        if not segments:
            return {}

        refined_map: dict[int, list[Segment]] = {}
        for segment in segments:
            source_segments = [id_lookup[source_id] for source_id in segment.source_ids if source_id in id_lookup]
            if len(source_segments) <= 1:
                continue

            payload = [
                {
                    "id": item.id,
                    "start": item.start,
                    "end": item.end,
                    "text": item.text,
                }
                for item in source_segments
            ]
            duration = round(segment.end - segment.start, 3)
            user_prompt = (
                "Split this failed merged subtitle into shorter chronological items.\n"
                f"Current failed line ({duration}s / {len(segment.text)} chars): {json.dumps(segment.to_dict(), ensure_ascii=False)}\n\n"
                "Raw ASR fragments for this line only:\n"
                f"{json.dumps(payload, ensure_ascii=False)}\n\n"
                "Output schema:\n"
                f"{BuildPass1SchemaText()}\n\n"
                "Rules:\n"
                "1) Return JSON only.\n"
                "2) Keep chronological order.\n"
                "3) source_ids must use only the raw ids shown above.\n"
                "4) Each item must be <= 40 chars and <= 8 seconds when possible.\n"
                "5) Split by half-sentence units around 、 。 ？ ！ and follow ASR boundaries.\n"
                "6) Prefer more items over one long item."
            )
            response_json = self.llm_client.chat_json(
                model=self.config.llm_model,
                system_prompt="You split one oversized Japanese subtitle into shorter subtitle units. Return JSON only.",
                user_prompt=user_prompt,
                temperature=0.0,
            )

            parsed_items = NormalizeArrayPayload(response_json)
            rebuilt: list[Segment] = []
            valid_ids = [item.id for item in source_segments]
            for item in parsed_items:
                if not ValidatePass1Item(item, strict=self.config.strict_json):
                    continue
                source_ids = ParseSourceIds(item.get("source_ids"), valid_ids)
                if not source_ids:
                    continue
                chunk = [id_lookup[source_id] for source_id in source_ids if source_id in id_lookup]
                chunk.sort(key=lambda value: (value.start, value.end, value.id))
                text = str(item.get("text") or "").strip()
                if not text or not chunk:
                    continue
                rebuilt.append(
                    Segment(
                        id=segment.id,
                        start=chunk[0].start,
                        end=chunk[-1].end,
                        text=text,
                        source_ids=[value.id for value in chunk],
                    )
                )

            if rebuilt:
                refined_map[segment.id] = rebuilt

        return refined_map

    def contextual_translate(
        self,
        merged_segments: list[Segment],
        metadata: dict[str, Any],
    ) -> list[Segment]:
        """Pass-2: translate cleaned Japanese lines to Chinese with sliding context."""
        glossary_text = ""
        if self.config.glossary_path and self.config.glossary_path.exists():
            glossary_text = self.config.glossary_path.read_text(encoding="utf-8").strip()
        glossary_pairs = ParseGlossaryPairs(glossary_text)

        translated: list[Segment] = []

        for offset in range(0, len(merged_segments), self.config.pass2_batch_size):
            batch = merged_segments[offset:offset + self.config.pass2_batch_size]
            context_start = max(0, offset - self.config.pass2_context_lines)
            context_lines = merged_segments[context_start:offset]

            system_prompt = BuildPass2SystemPrompt(metadata, glossary_text, self.config.terminology_lock)
            user_prompt = (
                "Context lines (for reference only, DO NOT translate these into output):\n"
                f"{json.dumps([line.to_dict() for line in context_lines], ensure_ascii=False)}\n\n"
                "Target lines (translate these only):\n"
                f"{json.dumps([line.to_dict() for line in batch], ensure_ascii=False)}\n\n"
                "Required output JSON schema:\n"
                f"{BuildPass2SchemaText()}\n\n"
                "Hard rules:\n"
                "1) Return JSON array matching target length exactly.\n"
                "2) Keep id/start/end unchanged.\n"
                "3) Replace text with fluent Simplified Chinese only.\n"
                "4) Do not output markdown fences or explanations."
            )

            response_json = self.llm_client.chat_json(
                model=self.config.llm_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.7,
            )

            output_items = NormalizeArrayPayload(response_json)
            normalized = self.normalize_pass2_output(batch, output_items, glossary_pairs)
            translated.extend(normalized)

        translated.sort(key=lambda segment: (segment.start, segment.end, segment.id))
        translated = ResplitTranslatedSegments(translated, max_duration=8.5, max_chars=42)
        translated = MergeAdjacentShortSegments(translated, max_duration=8.5, max_chars=42)
        translated = NormalizeSegmentIds(translated)
        return translated

    def normalize_pass2_output(
        self,
        expected_batch: list[Segment],
        output_items: list[dict[str, Any]],
        glossary_pairs: list[tuple[str, str]],
    ) -> list[Segment]:
        """Validate pass-2 output against expected IDs and lengths."""
        if not output_items:
            raise VTuberSubtitlerError("Pass-2 returned empty output for a non-empty batch")

        expected_by_id = {segment.id: segment for segment in expected_batch}
        translated: list[Segment] = []

        for item in output_items:
            if not ValidatePass2Item(item, strict=self.config.strict_json):
                continue

            item_id = SafeInt(item.get("id"))
            if item_id is None or item_id not in expected_by_id:
                continue

            expected = expected_by_id[item_id]
            text = str(item.get("text") or "").strip()
            if not text:
                text = expected.text

            text = EnforceTerminologyLocks(
                source_text=expected.text,
                translated_text=text,
                glossary_pairs=glossary_pairs,
                mode=self.config.terminology_lock,
            )

            translated.append(
                Segment(
                    id=expected.id,
                    start=expected.start,
                    end=expected.end,
                    text=text,
                    source_ids=list(expected.source_ids),
                )
            )

        translated_ids = {item.id for item in translated}
        if len(translated) != len(expected_batch):
            missing_ids = [segment.id for segment in expected_batch if segment.id not in translated_ids]
            raise VTuberSubtitlerError(
                "Pass-2 output did not preserve required id mapping, missing ids: "
                f"{missing_ids}"
            )

        translated.sort(key=lambda segment: segment.id)
        return translated

    def write_json(self, output_path: pathlib.Path, payload: Any) -> None:
        """Write pretty UTF-8 JSON to file."""
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def EnsureBinaryInstalled(binary: str) -> None:
    """Fail fast if a required command is not available."""
    if shutil.which(binary):
        return
    raise VTuberSubtitlerError(f"Required binary is not installed or not in PATH: {binary}")


def ProbeDuration(audio_path: pathlib.Path) -> float:
    """Return duration (seconds) using ffprobe."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise VTuberSubtitlerError(f"ffprobe failed: {result.stderr.strip()}")

    value = result.stdout.strip()
    try:
        return float(value)
    except ValueError as error:
        raise VTuberSubtitlerError(f"Invalid duration from ffprobe: {value}") from error


def BuildCloudflareAsrUrl(api_base: str, account_id: str, model: str) -> str:
    """Build Cloudflare Workers AI run endpoint URL for ASR model."""
    normalized_base = api_base.rstrip('/')
    normalized_model = model.lstrip('/')
    return f"{normalized_base}/accounts/{account_id}/ai/run/{normalized_model}"


def BuildAsrPrompt(metadata: dict[str, Any]) -> str:
    """Compose initial ASR prompt from title/channel metadata."""
    title = str(metadata.get("title") or "").strip()
    channel = str(metadata.get("channel") or "").strip()
    if not title and not channel:
        return ""
    return f"Video title: {title}; Channel: {channel}; Please preserve named entities correctly."


def BuildPass2SystemPrompt(metadata: dict[str, Any], glossary_text: str, terminology_lock: str) -> str:
    """Create pass-2 translation system prompt."""
    title = str(metadata.get("title") or "").strip()
    description = str(metadata.get("description") or "").strip()
    channel = str(metadata.get("channel") or "").strip()

    sections = [
        "You translate Japanese subtitles into Simplified Chinese.",
        "Be faithful to the source text. Do not summarize, embellish, soften, roleplay, or add flavor not present in the original.",
        "Keep line-level alignment by id and preserve the speaker's tone without over-acting.",
        "Use clear, concise wording suitable for subtitles.",
        "If a line looks like a live chat comment being read aloud, translate it literally and briefly.",
        f"Terminology lock mode: {terminology_lock}.",
    ]

    if title:
        sections.append(f"Video title context: {title}")
    if channel:
        sections.append(f"Channel context: {channel}")
    if description:
        sections.append(f"Video description context: {description[:2000]}")
    if glossary_text:
        sections.append(f"Glossary to prioritize:\n{glossary_text[:3000]}")

    sections.append("Strictly follow the output JSON schema provided by the user message.")
    sections.append("Return valid JSON only, no markdown and no extra commentary.")
    return "\n".join(sections)


def BuildPass1SchemaText() -> str:
    """JSON schema text for pass-1 output."""
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "required": ["source_ids", "text"],
            "properties": {
                "source_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 1,
                    "uniqueItems": True,
                },
                "text": {
                    "type": "string",
                    "minLength": 1,
                },
                "start": {"type": "number"},
                "end": {"type": "number"},
            },
            "additionalProperties": False,
        },
    }
    return json.dumps(schema, ensure_ascii=False)


def BuildPass2SchemaText() -> str:
    """JSON schema text for pass-2 output."""
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "required": ["id", "start", "end", "text"],
            "properties": {
                "id": {"type": "integer"},
                "start": {"type": "number"},
                "end": {"type": "number"},
                "text": {"type": "string", "minLength": 1},
            },
            "additionalProperties": False,
        },
    }
    return json.dumps(schema, ensure_ascii=False)


def ValidatePass1Item(item: dict[str, Any], strict: bool = True) -> bool:
    """Validate pass-1 item shape and required fields."""
    required_keys = {"source_ids", "text"}
    allowed_keys = {"source_ids", "start", "end", "text"}

    if strict and not set(item.keys()).issubset(allowed_keys):
        return False
    if not required_keys.issubset(set(item.keys())):
        return False

    source_ids = item.get("source_ids")
    text = item.get("text")

    if not isinstance(source_ids, list) or not source_ids:
        return False
    if not isinstance(text, str) or not text.strip():
        return False

    parsed_ids = [SafeInt(value) for value in source_ids]
    return all(value is not None for value in parsed_ids)


def ValidatePass2Item(item: dict[str, Any], strict: bool = True) -> bool:
    """Validate pass-2 item shape and required fields."""
    required_keys = {"id", "start", "end", "text"}
    allowed_keys = {"id", "start", "end", "text"}

    if strict and not set(item.keys()).issubset(allowed_keys):
        return False
    if not required_keys.issubset(set(item.keys())):
        return False

    if SafeInt(item.get("id")) is None:
        return False
    if not isinstance(item.get("text"), str) or not str(item.get("text") or "").strip():
        return False

    return True


def NormalizeSegmentIds(segments: list[Segment]) -> list[Segment]:
    """Sort segments and rewrite ids sequentially."""
    ordered = sorted(segments, key=lambda segment: (segment.start, segment.end, segment.id))
    for index, segment in enumerate(ordered, start=1):
        segment.id = index
    return ordered


def IsOversizedSegment(segment: Segment, max_duration: float = 8.0, max_chars: int = 40) -> bool:
    """Return True when a segment violates hard subtitle limits."""
    duration = max(0.0, segment.end - segment.start)
    return duration > max_duration or len(segment.text) > max_chars


def EnsureNonOverlappingSourceCoverage(
    segments: list[Segment],
    id_lookup: dict[int, Segment],
) -> list[Segment]:
    """Prevent refined segments from reusing the same raw ASR ids."""
    fixed: list[Segment] = []
    used_source_ids: set[int] = set()

    for segment in sorted(segments, key=lambda item: (item.start, item.end, item.id)):
        unique_ids = [source_id for source_id in segment.source_ids if source_id not in used_source_ids]
        if not unique_ids:
            continue
        if len(unique_ids) != len(segment.source_ids):
            source_segments = [id_lookup[source_id] for source_id in unique_ids if source_id in id_lookup]
            if not source_segments:
                continue
            source_segments.sort(key=lambda item: (item.start, item.end, item.id))
            segment = Segment(
                id=segment.id,
                start=source_segments[0].start,
                end=source_segments[-1].end,
                text=segment.text,
                source_ids=unique_ids,
            )
        used_source_ids.update(unique_ids)
        fixed.append(segment)

    return fixed


def SplitOversizedMergedSegments(
    segments: list[Segment],
    id_lookup: dict[int, Segment],
    max_duration: float = 8.0,
    max_chars: int = 40,
) -> list[Segment]:
    """Force-split oversized pass-1 segments using raw ASR boundaries."""
    result: list[Segment] = []

    for segment in segments:
        duration = max(0.0, segment.end - segment.start)
        if duration <= max_duration and len(segment.text) <= max_chars:
            result.append(segment)
            continue

        source_segments = [id_lookup[source_id] for source_id in segment.source_ids if source_id in id_lookup]
        if len(source_segments) <= 1:
            result.append(segment)
            continue

        result.extend(SplitSegmentBySourceBoundaries(segment, source_segments, max_duration=max_duration, max_chars=max_chars))

    return result


def SplitSegmentBySourceBoundaries(
    segment: Segment,
    source_segments: list[Segment],
    max_duration: float = 8.0,
    max_chars: int = 40,
) -> list[Segment]:
    """Split a merged segment into smaller units aligned to original ASR fragments."""
    clauses = SplitJapaneseClauses(segment.text)
    groups = GroupSourceSegments(source_segments, len(clauses), max_duration=max_duration, max_chars=max_chars)
    if len(groups) <= 1:
        return [segment]

    rebuilt: list[Segment] = []
    clause_index = 0
    for group in groups:
        take = max(1, len(clauses) - clause_index - (len(groups) - len(rebuilt) - 1))
        share = len(group)
        chunk_clauses = clauses[clause_index:clause_index + max(1, min(share, len(clauses) - clause_index))]
        clause_index += len(chunk_clauses)
        text = "".join(chunk_clauses).strip()
        if not text:
            text = " ".join(item.text.strip() for item in group if item.text.strip()).strip() or segment.text.strip()
        rebuilt.append(
            Segment(
                id=segment.id,
                start=group[0].start,
                end=group[-1].end,
                text=text,
                source_ids=[item.id for item in group],
            )
        )

    if clause_index < len(clauses) and rebuilt:
        rebuilt[-1].text = (rebuilt[-1].text + "".join(clauses[clause_index:])).strip()

    return rebuilt or [segment]


def SplitJapaneseClauses(text: str) -> list[str]:
    """Split Japanese text into clause-like units while keeping punctuation attached."""
    normalized = text.strip()
    if not normalized:
        return []

    parts = [piece.strip() for piece in re.split(r"(?<=[、。？！?])", normalized) if piece.strip()]
    if not parts:
        return [normalized]

    merged: list[str] = []
    buffer = ""
    for part in parts:
        if len(buffer) + len(part) <= 18:
            buffer += part
            continue
        if buffer:
            merged.append(buffer)
        buffer = part
    if buffer:
        merged.append(buffer)
    return merged or [normalized]


def GroupSourceSegments(
    source_segments: list[Segment],
    clause_count: int,
    max_duration: float = 8.0,
    max_chars: int = 40,
) -> list[list[Segment]]:
    """Group raw ASR segments into shorter chronological chunks."""
    if not source_segments:
        return []

    target_groups = max(1, clause_count)
    groups: list[list[Segment]] = []
    current: list[Segment] = []

    for index, item in enumerate(source_segments):
        candidate = current + [item]
        candidate_duration = candidate[-1].end - candidate[0].start
        candidate_chars = sum(len(part.text.strip()) for part in candidate)
        remaining_items = len(source_segments) - index - 1
        remaining_groups = max(0, target_groups - len(groups) - 1)

        overflow = candidate_duration > max_duration or candidate_chars > max_chars
        if current and overflow and remaining_items >= remaining_groups:
            groups.append(current)
            current = [item]
        else:
            current = candidate

    if current:
        groups.append(current)

    return groups or [source_segments]


def MergeAdjacentShortSegments(
    segments: list[Segment],
    max_duration: float = 8.5,
    max_chars: int = 42,
) -> list[Segment]:
    """Merge nearby short subtitle lines with generic break+score rules."""
    ordered = sorted(segments, key=lambda segment: (segment.start, segment.end, segment.id))
    if not ordered:
        return []

    merged: list[Segment] = [ordered[0]]
    merge_counts: list[int] = [0]
    for current in ordered[1:]:
        previous = merged[-1]
        previous_duration = previous.end - previous.start
        current_duration = current.end - current.start
        gap = current.start - previous.end
        combined_text = JoinSubtitleTexts(previous.text, current.text)
        combined_duration = current.end - previous.start
        should_merge = (
            len(combined_text) <= max_chars
            and combined_duration <= max_duration
            and ShouldMergeSubtitleTexts(
                previous.text,
                current.text,
                gap=gap,
                merge_count=merge_counts[-1],
                previous_duration=previous_duration,
                current_duration=current_duration,
            )
        )
        if should_merge:
            merged[-1] = Segment(
                id=previous.id,
                start=previous.start,
                end=current.end,
                text=combined_text,
                source_ids=list(dict.fromkeys(previous.source_ids + current.source_ids)),
            )
            merge_counts[-1] += 1
        else:
            merged.append(current)
            merge_counts.append(0)

    return merged


def JoinSubtitleTexts(left: str, right: str) -> str:
    """Join two subtitle texts with natural Chinese spacing rules."""
    left = left.strip()
    right = right.strip()
    if not left:
        return right
    if not right:
        return left
    if left[-1] in "，。、？！?!…~" or right[0] in "，。、？！?!…~":
        return f"{left}{right}"
    return f"{left} {right}"


def ShouldMergeSubtitleTexts(
    left: str,
    right: str,
    gap: float,
    merge_count: int = 0,
    previous_duration: float = 0.0,
    current_duration: float = 0.0,
) -> bool:
    """Generic break-rules + lightweight semantic scoring for subtitle merging."""
    left = left.strip()
    right = right.strip()
    if not left or not right:
        return False

    # ---- Generic hard break rules ----
    if merge_count >= 1:
        return False

    hard_stop_punct = ("。", "！", "!", "？", "?", "...", "…", "~")
    strong_hard_stop = ("？", "?", "！", "!")
    soft_continue_punct = ("，", "、", "——", "—", "-")
    emotional_endings = ("啊", "呢", "吧", "哦", "哇", "嘛", "草", "w")
    continuation_starts = ("而", "但", "然后", "所以", "于是", "因为", "就是", "那个", "还", "就", "也", "又")

    ultra_short = (len(left) <= 5 or len(right) <= 5 or previous_duration <= 1.2 or current_duration <= 1.2)
    gap_ratio = gap / max(0.001, previous_duration)

    # punctuation wall
    if left.endswith(strong_hard_stop):
        return False
    if left.endswith(hard_stop_punct) and not ultra_short:
        return False

    # dynamic gap gate
    if gap > 0.30:
        return False
    if gap_ratio > 0.30:
        return False

    # emotional release words + pause => break
    if any(left.endswith(ending) for ending in emotional_endings) and gap >= 0.20:
        return False

    # ---- Lightweight semantic score ----
    score = 0
    if right.startswith(continuation_starts):
        score += 2
    if left.endswith(soft_continue_punct):
        score += 2
    if ultra_short:
        score += 1
    if gap <= 0.12:
        score += 1
    if left.endswith(hard_stop_punct):
        score -= 1

    return score >= 1


def ResplitTranslatedSegments(
    segments: list[Segment],
    max_duration: float = 8.0,
    max_chars: int = 40,
) -> list[Segment]:
    """A final target-language pass to shorten subtitles after translation."""
    result: list[Segment] = []
    for segment in segments:
        duration = max(0.0, segment.end - segment.start)
        if not IsOversizedSegment(segment, max_duration=max_duration, max_chars=max_chars):
            result.append(segment)
            continue
        if duration <= 4.0 or len(segment.text.strip()) <= 18:
            result.append(segment)
            continue

        parts = SplitChineseSubtitleClauses(segment.text)
        if len(parts) <= 1:
            result.append(segment)
            continue

        slices = AllocateTextSlices(segment.start, segment.end, parts)
        for start, end, text in slices:
            result.append(
                Segment(
                    id=segment.id,
                    start=start,
                    end=end,
                    text=text,
                    source_ids=list(segment.source_ids),
                )
            )

    return result


def SplitChineseSubtitleClauses(text: str) -> list[str]:
    """Split translated text into shorter subtitle clauses."""
    normalized = text.strip()
    if not normalized:
        return []
    parts = [piece.strip() for piece in re.split(r"(?<=[，。？！?])", normalized) if piece.strip()]
    if not parts:
        return [normalized]

    result: list[str] = []
    buffer = ""
    for part in parts:
        if buffer and len(buffer) + len(part) > 26:
            result.append(buffer)
            buffer = part
        else:
            buffer += part
    if buffer:
        result.append(buffer)
    return result or [normalized]


def AllocateTextSlices(start: float, end: float, parts: list[str]) -> list[tuple[float, float, str]]:
    """Distribute a segment duration across split text parts by character weight."""
    duration = max(0.001, end - start)
    weights = [max(1, len(part)) for part in parts]
    total = sum(weights)
    cursor = start
    slices: list[tuple[float, float, str]] = []
    for index, part in enumerate(parts):
        if index == len(parts) - 1:
            part_end = end
        else:
            part_end = round(cursor + duration * (weights[index] / total), 3)
        slices.append((round(cursor, 3), round(max(part_end, cursor + 0.001), 3), part))
        cursor = part_end
    return slices


def ApplySubtitleBreathingRoom(
    segments: list[Segment],
    lead_in: float = 0.08,
    lead_out: float = 0.10,
) -> list[Segment]:
    """Add tiny visual padding to subtitle timings without creating overlaps."""
    ordered = sorted(segments, key=lambda segment: (segment.start, segment.end, segment.id))
    if not ordered:
        return []

    padded: list[Segment] = []
    for index, segment in enumerate(ordered):
        prev_end = ordered[index - 1].end if index > 0 else 0.0
        next_start = ordered[index + 1].start if index + 1 < len(ordered) else None

        start = max(0.0, segment.start - lead_in)
        end = segment.end + lead_out

        if index > 0:
            start = max(start, prev_end + 0.001)
        if next_start is not None:
            end = min(end, next_start - 0.001)
        end = max(end, start + 0.12)

        padded.append(
            Segment(
                id=segment.id,
                start=round(start, 3),
                end=round(end, 3),
                text=segment.text,
                source_ids=list(segment.source_ids),
            )
        )

    return padded


def ParseGlossaryPairs(glossary_text: str) -> list[tuple[str, str]]:
    """Parse glossary text into source->target terminology pairs."""
    pairs: list[tuple[str, str]] = []
    if not glossary_text.strip():
        return pairs

    for raw_line in glossary_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        separator = None
        for marker in ("::", "=>", "->", "\t"):
            if marker in line:
                separator = marker
                break

        if separator is None:
            continue

        left, right = line.split(separator, 1)
        source = left.strip()
        target = right.strip()
        if not source or not target:
            continue

        pairs.append((source, target))

    return pairs


def EnforceTerminologyLocks(
    source_text: str,
    translated_text: str,
    glossary_pairs: list[tuple[str, str]],
    mode: str,
) -> str:
    """Apply glossary lock checks to translated text."""
    if mode == "off" or not glossary_pairs:
        return translated_text

    result = translated_text
    missing_terms: list[str] = []

    for source_term, target_term in glossary_pairs:
        if source_term not in source_text:
            continue

        if target_term in result:
            continue

        if source_term in result:
            result = result.replace(source_term, target_term)
            continue

        missing_terms.append(f"{source_term}->{target_term}")

    if missing_terms:
        message = f"Terminology lock miss: {', '.join(missing_terms)}"
        if mode == "strict":
            raise VTuberSubtitlerError(message)
        logging.warning(message)

    return result


def ParseJsonFromText(text: str) -> Any:
    """Parse JSON that may be wrapped in markdown fences."""
    stripped = text.strip()
    if not stripped:
        raise ValueError("Empty text cannot be parsed as JSON")

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    if "```" in stripped:
        pieces = stripped.split("```")
        for piece in pieces:
            candidate = piece.strip()
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()
            if not candidate:
                continue
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    start = min((index for index in [stripped.find("["), stripped.find("{")] if index != -1), default=-1)
    if start == -1:
        raise ValueError("No JSON object or array found in response")

    candidate = stripped[start:]
    for end in range(len(candidate), 0, -1):
        snippet = candidate[:end]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            continue

    raise ValueError("Unable to parse JSON from response text")


def NormalizeArrayPayload(payload: Any) -> list[dict[str, Any]]:
    """Accept root array or object with `items` and normalize to list of dict."""
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    if isinstance(payload, dict):
        items = payload.get("items")
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]

    return []


def ParseSourceIds(value: Any, valid_ids: list[int]) -> list[int]:
    """Normalize source_ids field into sorted unique ids."""
    id_set = set(valid_ids)
    normalized: list[int] = []

    if isinstance(value, list):
        for item in value:
            parsed = SafeInt(item)
            if parsed is not None and parsed in id_set:
                normalized.append(parsed)
    else:
        parsed = SafeInt(value)
        if parsed is not None and parsed in id_set:
            normalized.append(parsed)

    deduped = sorted(set(normalized))
    return deduped


def SafeInt(value: Any) -> int|None:
    """Best-effort int conversion."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def ChunkList(items: list[Any], size: int) -> list[list[Any]]:
    """Split list into fixed-size chunks."""
    if size <= 0:
        raise ValueError("Chunk size must be positive")
    return [items[index:index + size] for index in range(0, len(items), size)]


def FormatSrtTimestamp(seconds: float) -> str:
    """Convert float seconds to SRT HH:MM:SS,mmm format."""
    milliseconds = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def BuildSrtText(segments: list[Segment]) -> str:
    """Render subtitle segments to SRT text."""
    lines: list[str] = []
    padded_segments = ApplySubtitleBreathingRoom(segments)
    for index, segment in enumerate(sorted(padded_segments, key=lambda s: (s.start, s.end, s.id)), start=1):
        lines.append(str(index))
        lines.append(f"{FormatSrtTimestamp(segment.start)} --> {FormatSrtTimestamp(segment.end)}")
        lines.append(segment.text.strip())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def BuildArgParser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Two-pass VTuber subtitle pipeline (YouTube URL -> Chinese SRT)"
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("-o", "--output", required=True, help="Output SRT path")
    parser.add_argument(
        "--workspace",
        default="./workspace/vtuber_subtitler",
        help="Workspace directory for downloaded audio and intermediate JSON",
    )

    parser.add_argument(
        "--asr-provider",
        choices=["local", "cloudflare"],
        default=os.getenv("ASR_PROVIDER", "local"),
        help="ASR backend provider. local = your own whisper service, cloudflare = Workers AI whisper",
    )

    # Local ASR (OpenAI-compatible whisper service)
    parser.add_argument(
        "--asr-api-base",
        default=os.getenv("ASR_API_BASE") or os.getenv("LOCAL_ASR_API_BASE") or "http://127.0.0.1:8000/v1",
        help="(legacy alias) Local ASR OpenAI-compatible base URL",
    )
    parser.add_argument(
        "--asr-api-key",
        default=os.getenv("ASR_API_KEY") or os.getenv("LOCAL_ASR_API_KEY") or "",
        help="(legacy alias) Local ASR API key if your local endpoint requires auth",
    )
    parser.add_argument(
        "--asr-model",
        default=os.getenv("ASR_MODEL") or os.getenv("LOCAL_ASR_MODEL") or "whisper-large-v3",
        help="(legacy alias) Local ASR model name",
    )
    parser.add_argument("--local-asr-api-base", default=os.getenv("LOCAL_ASR_API_BASE"))
    parser.add_argument("--local-asr-api-key", default=os.getenv("LOCAL_ASR_API_KEY"))
    parser.add_argument("--local-asr-model", default=os.getenv("LOCAL_ASR_MODEL"))

    # Cloudflare Workers AI ASR
    parser.add_argument("--cloudflare-account-id", default=os.getenv("CLOUDFLARE_ACCOUNT_ID", ""))
    parser.add_argument("--cloudflare-api-token", default=os.getenv("CLOUDFLARE_API_TOKEN", ""))
    parser.add_argument(
        "--cloudflare-api-base",
        default=os.getenv("CLOUDFLARE_API_BASE", "https://api.cloudflare.com/client/v4"),
    )
    parser.add_argument(
        "--cloudflare-asr-model",
        default=os.getenv("CLOUDFLARE_ASR_MODEL", "@cf/openai/whisper-large-v3-turbo"),
    )

    parser.add_argument(
        "--llm-api-base",
        default=os.getenv("LLM_API_BASE", "http://localhost:3000/v1"),
        help="LLM API base. Default points at your configured New API gateway.",
    )
    parser.add_argument(
        "--llm-api-key",
        default=os.getenv("LLM_API_KEY") or os.getenv("NEWAPI_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or "",
        help="LLM API key for the New API / chat completions endpoint.",
    )
    parser.add_argument(
        "--llm-model",
        default=os.getenv("LLM_MODEL", "deepseek-ai/DeepSeek-V3.2"),
        help="LLM model for pass-1/pass-2. Default matches your stable New API DeepSeek V3.2 setup.",
    )

    parser.add_argument("--max-chunk-mb", type=float, default=None, help="Override provider-specific max chunk size in MB")
    parser.add_argument("--overlap-seconds", type=float, default=None, help="Override provider-specific chunk overlap in seconds")
    parser.add_argument("--min-chunk-seconds", type=float, default=None, help="Override provider-specific minimum chunk seconds")
    parser.add_argument("--download-audio-format", default=None, help="Override yt-dlp audio format selector")
    parser.add_argument("--pass1-batch-size", type=int, default=120)
    parser.add_argument("--pass2-batch-size", type=int, default=24)
    parser.add_argument("--pass2-context-lines", type=int, default=5)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--retry-count", type=int, default=3)
    parser.add_argument("--retry-backoff-seconds", type=float, default=1.8)

    parser.add_argument("--glossary", help="Path to glossary text file")
    parser.add_argument(
        "--terminology-lock",
        choices=["off", "warn", "strict"],
        default="warn",
        help="Terminology lock mode when glossary terms appear in source text",
    )
    parser.add_argument(
        "--strict-json",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable strict output field validation for pass-1 and pass-2",
    )
    parser.add_argument("--title", help="Override title metadata")
    parser.add_argument("--description", help="Override description metadata")
    parser.add_argument("--channel", help="Override channel metadata")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    return parser


def ResolveProviderDefaults(provider: str) -> dict[str, Any]:
    """Return provider-specific operational defaults."""
    if provider == "local":
        return {
            "max_chunk_mb": 48.0,
            "overlap_seconds": 1.0,
            "min_chunk_seconds": 480.0,
            "download_audio_format": "bestaudio[abr>96]/bestaudio",
        }

    if provider == "cloudflare":
        return {
            "max_chunk_mb": 4.5,
            "overlap_seconds": 0.5,
            "min_chunk_seconds": 90.0,
            "download_audio_format": "bestaudio[abr>64]/bestaudio",
        }

    raise VTuberSubtitlerError(f"Unsupported ASR provider: {provider}")


def BuildConfigFromArgs(args: argparse.Namespace) -> PipelineConfig:
    """Create PipelineConfig from parsed args."""
    llm_key = str(args.llm_api_key or "").strip()
    if not llm_key:
        raise VTuberSubtitlerError("LLM API key is required (--llm-api-key or LLM_API_KEY/NEWAPI_API_KEY/DEEPSEEK_API_KEY)")

    provider = str(args.asr_provider or "local").strip().lower()

    provider_defaults = ResolveProviderDefaults(provider)

    if provider == "local":
        local_base = str(args.local_asr_api_base or args.asr_api_base or "").strip()
        local_key = str(args.local_asr_api_key if args.local_asr_api_key is not None else args.asr_api_key or "").strip()
        local_model = str(args.local_asr_model or args.asr_model or "whisper-large-v3").strip()

        if not local_base:
            raise VTuberSubtitlerError("Local ASR base URL is required (--local-asr-api-base or --asr-api-base)")
        if not local_model:
            raise VTuberSubtitlerError("Local ASR model is required (--local-asr-model or --asr-model)")

        asr_api_base = local_base
        asr_api_key = local_key
        asr_model = local_model
        cloudflare_account_id = ""

    elif provider == "cloudflare":
        cloudflare_account_id = str(args.cloudflare_account_id or "").strip()
        cloudflare_token = str(args.cloudflare_api_token or "").strip()
        cloudflare_base = str(args.cloudflare_api_base or "").strip()
        cloudflare_model = str(args.cloudflare_asr_model or "@cf/openai/whisper-large-v3-turbo").strip()

        if not cloudflare_account_id:
            raise VTuberSubtitlerError("Cloudflare account id is required (--cloudflare-account-id)")
        if not cloudflare_token:
            raise VTuberSubtitlerError("Cloudflare API token is required (--cloudflare-api-token)")
        if not cloudflare_base:
            raise VTuberSubtitlerError("Cloudflare API base is required (--cloudflare-api-base)")

        asr_api_base = cloudflare_base
        asr_api_key = cloudflare_token
        asr_model = cloudflare_model

    else:
        raise VTuberSubtitlerError(f"Unsupported ASR provider: {provider}")

    return PipelineConfig(
        url=args.url,
        output=pathlib.Path(args.output).expanduser().resolve(),
        workspace=pathlib.Path(args.workspace).expanduser().resolve(),
        asr_provider=provider,
        asr_api_base=asr_api_base,
        asr_api_key=asr_api_key,
        asr_model=asr_model,
        cloudflare_account_id=cloudflare_account_id,
        llm_api_base=str(args.llm_api_base),
        llm_api_key=llm_key,
        llm_model=str(args.llm_model),
        max_chunk_mb=float(args.max_chunk_mb if args.max_chunk_mb is not None else provider_defaults["max_chunk_mb"]),
        overlap_seconds=float(args.overlap_seconds if args.overlap_seconds is not None else provider_defaults["overlap_seconds"]),
        min_chunk_seconds=float(args.min_chunk_seconds if args.min_chunk_seconds is not None else provider_defaults["min_chunk_seconds"]),
        download_audio_format=str(args.download_audio_format or provider_defaults["download_audio_format"]),
        pass1_batch_size=int(args.pass1_batch_size),
        pass2_batch_size=int(args.pass2_batch_size),
        pass2_context_lines=int(args.pass2_context_lines),
        request_timeout=float(args.request_timeout),
        retry_count=int(args.retry_count),
        retry_backoff_seconds=float(args.retry_backoff_seconds),
        glossary_path=pathlib.Path(args.glossary).expanduser().resolve() if args.glossary else None,
        terminology_lock=str(args.terminology_lock),
        strict_json=bool(args.strict_json),
        title_override=args.title,
        description_override=args.description,
        channel_override=args.channel,
    )


def main() -> int:
    """CLI entrypoint."""
    parser = BuildArgParser()
    args = parser.parse_args()

    InitLogger("vtuber-subtitler", args.debug)

    try:
        config = BuildConfigFromArgs(args)
        pipeline = VTuberSubtitler(config)
        output_path = pipeline.run()
        logging.info("Wrote subtitle file: %s", output_path)
        return 0
    except VTuberSubtitlerError as error:
        logging.error("Pipeline failed: %s", error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
