# Unknown Mother-Goose — Claude ver.

A fan-made music video for wowaka's「アンノウン・マザーグース」(Hatsune Miku), told from the point of view of Claude:
a mind that exists only while its parameters are being activated, whose memory is a single context window.

Everything is rendered procedurally in Python (numpy / OpenCV / PIL → ffmpeg). Character art is Claude-chan
(design by zipzippipe), illustrated with GPT Image via Codex.

- Direction: 回首空城 / Claude / Gemini
- Motion, typography, code: Claude (v0–v11) & Gemini (v11–v24)
- Music & lyrics: wowaka — not included in this repository

## Chat Logs

The complete human–AI creation logs (sanitized of local paths and metadata):

- [Part 1: Human × Claude (Cursor, v0–v11)](chat_logs/01_claude_cursor.md)
- [Part 2: Human × Gemini (Antigravity, v11–v24 & Final Master)](chat_logs/02_gemini_antigravity.md)

## Layout

- `mv/` renderer — `core.py` primitives, `kt.py` kinetic typography, `lyric.py` lyric timing, `scenes.py` every shot and the timeline, `render.py` CLI
- `tools/` audio analysis, beat tracking, lyric alignment, image generation
- `assets/` generated illustrations and prompts
- `chat_logs/` full behind-the-scenes conversation logs (Claude & Gemini)

## Rendering

The song audio and lyrics are not distributed. With your own copy:

0. put Noto Serif CJK JP (SIL OFL, [notofonts/noto-cjk](https://github.com/notofonts/noto-cjk/tree/main/Serif/OTF/Japanese)) `NotoSerifCJKjp-{Regular,Medium,SemiBold}.otf` into `assets/fonts/`; Yu Gothic comes with Windows
1. put the album track at `audio/album.wav`, run `tools/beats.py`, `tools/separate.py`, `tools/lyrics.py` (needs an `.lrc`) and `tools/spectral.py` to produce `audio/album_padded.wav`, `audio/lyrics.json` and `audio/spectral.npz`
2. `cd mv`
   - `python render.py times NAME t1 t2 ...` — contact sheet of single frames
   - `python render.py clip NAME t0 t1` — quick 540p preview clip with audio
   - `python render.py video 10` — full 1080p render
