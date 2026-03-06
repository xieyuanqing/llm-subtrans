# LLM-Subtrans（中文说明）

LLM-Subtrans 是一个开源字幕翻译工具，使用大语言模型（LLM）作为翻译服务，可在模型支持的语言对之间进行字幕翻译。

- 支持格式：`.srt`、`.ssa/.ass`、`.vtt`
- 支持方式：GUI + 命令行
- 网络要求：需要联网，字幕会发送到对应服务商，需遵循其隐私政策

## 快速开始

### 1) 安装

普通用户建议直接下载 Releases 包：
- https://github.com/machinewrapped/llm-subtrans/releases

开发者或命令行用户可从源码安装（见英文文档完整说明）：
- [README.en.md](./README.en.md)

### 2) 命令行示例

```sh
# OpenRouter 自动选模型
llm-subtrans --auto -l <目标语言> <字幕文件路径>

# 指定模型
llm-subtrans --model google/gemini-2.5-flash -l <目标语言> <字幕文件路径>

# 翻译并转换格式（ASS -> SRT）
llm-subtrans -l <目标语言> -o output.srt input.ass
```

### 3) VTuber 两阶段流程（实验）

```sh
# 本地 Whisper（优先）
python scripts/vtuber_subtitler.py "https://www.youtube.com/watch?v=<VIDEO_ID>" \
  --output ./output/demo.zh.srt \
  --asr-provider local \
  --local-asr-api-base "http://100.74.157.37:8000/v1" \
  --local-asr-api-key "$LOCAL_ASR_API_KEY" \
  --llm-api-base http://localhost:3000/v1 \
  --llm-api-key "$NEWAPI_API_KEY" \
  --llm-model deepseek-ai/DeepSeek-V3.2 \
  --terminology-lock warn \
  --strict-json
```

## 文档结构

- 中文说明（当前文件）：`README.zh.md`
- 英文完整文档：`README.en.md`

> 注：英文文档包含最完整的 Provider、参数、安装与高级配置细节。
