import unittest

from PySubtrans.Helpers.TestCases import LoggedTestCase
from scripts.vtuber_subtitler import (
    BuildSrtText,
    FormatSrtTimestamp,
    NormalizeArrayPayload,
    ParseJsonFromText,
    ParseSourceIds,
    Segment,
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


if __name__ == '__main__':
    unittest.main()
