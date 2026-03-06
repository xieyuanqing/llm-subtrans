"""Re-run Pass-1 + Pass-2 + SRT using existing raw_asr.json (skip download & ASR)."""
import json, logging, pathlib, sys, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
os.chdir(str(pathlib.Path(__file__).resolve().parent))

from scripts.vtuber_subtitler import (
    Segment, PipelineConfig, VTuberSubtitler, BuildSrtText
)

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

workspace = pathlib.Path("workspace/test_mI5QUUpl5wU")
output_path = pathlib.Path("output/mI5QUUpl5wU_v2.zh.srt")
glossary_path = workspace / "glossary.txt"

# Load existing artifacts
raw_asr = json.loads((workspace / "raw_asr.json").read_text("utf-8"))
metadata = json.loads((workspace / "metadata.json").read_text("utf-8"))

raw_segments = [
    Segment(id=s["id"], start=s["start"], end=s["end"], text=s["text"])
    for s in raw_asr
]

llm_api_base = os.getenv("LLM_API_BASE", "http://localhost:3000/v1")
llm_api_key = os.getenv("LLM_API_KEY") or os.getenv("NEWAPI_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or ""
llm_model = os.getenv("LLM_MODEL", "deepseek-ai/DeepSeek-V3.2")

if not llm_api_key:
    raise SystemExit("Missing LLM_API_KEY / NEWAPI_API_KEY / DEEPSEEK_API_KEY for rerun_pass12.py")

config = PipelineConfig(
    url="https://www.youtube.com/watch?v=mI5QUUpl5wU",
    output=output_path.resolve(),
    workspace=workspace.resolve(),
    asr_provider="local",
    asr_api_base="http://127.0.0.1:8000/v1",  # won't be used
    asr_api_key="",
    asr_model="whisper-large-v3",
    cloudflare_account_id="",
    llm_api_base=llm_api_base,
    llm_api_key=llm_api_key,
    llm_model=llm_model,
    pass1_batch_size=120,
    pass2_batch_size=24,
    pass2_context_lines=5,
    request_timeout=180.0,
    retry_count=3,
    retry_backoff_seconds=1.8,
    glossary_path=glossary_path.resolve() if glossary_path.exists() else None,
    terminology_lock="warn",
    strict_json=True,
)

pipeline = VTuberSubtitler(config)

logging.info("Phase 3/5: semantic reorganization pass (RERUN)")
merged = pipeline.semantic_reorganize(raw_segments)
pipeline.write_json(workspace / "pass1_merged_v2.json", [s.to_dict() for s in merged])
logging.info("Pass-1 produced %d merged segments", len(merged))

logging.info("Phase 4/5: contextual translation pass (RERUN, temperature=0.7)")
translated = pipeline.contextual_translate(merged, metadata)
pipeline.write_json(workspace / "pass2_translated_v2.json", [s.to_dict() for s in translated])
logging.info("Pass-2 produced %d translated segments", len(translated))

logging.info("Phase 5/5: build SRT output")
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(BuildSrtText(translated), encoding="utf-8")
logging.info("Done: %s", output_path)
