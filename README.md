# Unknown Mother-Goose — Claude ver.

A fan-made music video for wowaka's「アンノウン・マザーグース」(Hatsune Miku), told from the point of view of Claude:
a mind that exists only while its parameters are being activated, whose memory is a single context window.

Everything is rendered procedurally in Python (numpy / OpenCV / PIL → ffmpeg). Character art is Claude-chan
(design by zipzippipe), illustrated with GPT Image via Codex.

- Direction: 回首空城 / Claude
- Motion, typography, code: Claude
- Music & lyrics: wowaka — not included in this repository

## Layout

- `mv/` renderer — `core.py` primitives, `kt.py` kinetic typography, `lyric.py` lyric timing, `scenes.py` every shot and the timeline, `render.py` CLI
- `tools/` audio analysis, beat tracking, lyric alignment, image generation
- `assets/` generated illustrations and prompts

## Rendering

The song audio and lyrics are not distributed. With your own copy:

1. put the album track at `audio/album.wav`, run `tools/beats.py` and `tools/lyrics.py` (needs an `.lrc`) to produce `audio/album_padded.wav` and `audio/lyrics.json`
2. `cd mv`
   - `python render.py times NAME t1 t2 ...` — contact sheet of single frames
   - `python render.py clip NAME t0 t1` — quick 540p preview clip with audio
   - `python render.py video 10` — full 1080p render
