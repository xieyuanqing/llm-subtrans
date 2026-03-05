import unittest

from PySubtrans.Helpers.TestCases import LoggedTestCase
from scripts.vtuber_subtitler import (
    BuildCloudflareAsrUrl,
    BuildPass1SchemaText,
    BuildPass2SchemaText,
    BuildSrtText,
    EnforceTerminologyLocks,
    FormatSrtTimestamp,
    NormalizeArrayPayload,
    ParseGlossaryPairs,
    ParseJsonFromText,
    ParseSourceIds,
    Segment,
    ValidatePass1Item,
    ValidatePass2Item,
)


class TestVTuberSubtitlerHelpers(LoggedTestCase):
    def test_ParseJsonFromText(self):
        plain_payload = '[{"id":1,"text":"hello"}]'
        fenced_payload = """```json
[{"id":2,"text":"world"}]
```"""

        plain_result = ParseJsonFromText(plain_payload)
        fenced_result = ParseJsonFromText(fenced_payload)

        self.assertLoggedEqual("plain json parse result", 1, plain_result[0]["id"])
        self.assertLoggedEqual("fenced json parse result", 2, fenced_result[0]["id"])

    def test_NormalizeArrayPayload(self):
        from_array = NormalizeArrayPayload([{"id": 1}, "invalid"])
        from_object = NormalizeArrayPayload({"items": [{"id": 2}, None]})
        from_invalid = NormalizeArrayPayload({"bad": True})

        self.assertLoggedEqual("array payload length", 1, len(from_array))
        self.assertLoggedEqual("object payload length", 1, len(from_object))
        self.assertLoggedEqual("invalid payload returns empty", 0, len(from_invalid))

    def test_ParseSourceIds(self):
        parsed_ids = ParseSourceIds(["3", 1, 3, 99, "bad"], valid_ids=[1, 2, 3])
        self.assertLoggedSequenceEqual("source id parsing", [1, 3], parsed_ids)

    def test_FormatSrtTimestamp(self):
        timestamp = FormatSrtTimestamp(3723.456)
        self.assertLoggedEqual("srt timestamp formatting", "01:02:03,456", timestamp)

    def test_BuildSrtText(self):
        segments = [
            Segment(id=2, start=2.5, end=4.0, text="第二句"),
            Segment(id=1, start=0.1, end=1.2, text="第一句"),
        ]
        srt_text = BuildSrtText(segments)

        self.assertLoggedIn("first cue index", "1\n00:00:00,100 --> 00:00:01,200\n第一句", srt_text)
        self.assertLoggedIn("second cue index", "2\n00:00:02,500 --> 00:00:04,000\n第二句", srt_text)

    def test_BuildSchemaTexts(self):
        pass1_schema = BuildPass1SchemaText()
        pass2_schema = BuildPass2SchemaText()

        self.assertLoggedIn("pass1 schema contains additionalProperties", '"additionalProperties": false', pass1_schema)
        self.assertLoggedIn("pass2 schema contains id field", '"id": {"type": "integer"}', pass2_schema)

    def test_BuildCloudflareAsrUrl(self):
        url = BuildCloudflareAsrUrl(
            api_base="https://api.cloudflare.com/client/v4/",
            account_id="abc123",
            model="@cf/openai/whisper-large-v3-turbo",
        )
        self.assertLoggedEqual(
            "cloudflare asr url",
            "https://api.cloudflare.com/client/v4/accounts/abc123/ai/run/@cf/openai/whisper-large-v3-turbo",
            url,
        )

    def test_ValidatePassItems(self):
        valid_pass1 = {"source_ids": [1, 2], "text": "こんにちは"}
        invalid_pass1 = {"source_ids": [], "text": ""}

        valid_pass2 = {"id": 1, "start": 0.0, "end": 1.0, "text": "你好"}
        invalid_pass2 = {"id": "bad", "start": 0.0, "end": 1.0, "text": ""}

        self.assertLoggedTrue("validate pass1 valid item", ValidatePass1Item(valid_pass1, strict=True))
        self.assertLoggedFalse("validate pass1 invalid item", ValidatePass1Item(invalid_pass1, strict=True))
        self.assertLoggedTrue("validate pass2 valid item", ValidatePass2Item(valid_pass2, strict=True))
        self.assertLoggedFalse("validate pass2 invalid item", ValidatePass2Item(invalid_pass2, strict=True))

    def test_GlossaryParsingAndTerminologyLock(self):
        glossary_text = """
# comment
ぺこら::佩可拉
草=>笑死
""".strip()
        pairs = ParseGlossaryPairs(glossary_text)

        self.assertLoggedEqual("glossary pair count", 2, len(pairs))

        source = "ぺこら、今日も草"
        translated = "ぺこら今天也太草了"
        locked = EnforceTerminologyLocks(source, translated, pairs, mode="warn")

        self.assertLoggedIn("terminology replacement applied", "佩可拉", locked)


if __name__ == '__main__':
    unittest.main()
