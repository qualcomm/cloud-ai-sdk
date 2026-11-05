# Demo Configuration and Font Handling

## Contents

- [Write demo_config.json](#write-demo_configjson)
- [Font handling notes](#font-handling-notes)

## Write demo_config.json

After the smoke test passes, write `demo_config.json` alongside `predict.py`.
`input_type` must match the template selected during routing: `"image"`, `"text"`, `"audio"`, or `"tts"`.

Make `predict.py` relocatable: derive model-artifact paths from the directory containing the script, not the current working directory or a hard-coded workspace path.

```python
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent
QPC_DIR = WORKSPACE / "qpc"
```

Omit `runtime` when `predict.py` uses `python3`. If it requires a Python interpreter stored inside the model directory, declare that interpreter with a relative path:

```json
{
  "input_type": "audio",
  "runtime": {"interpreter": "venv_runtime/bin/python"}
}
```

Do not use an absolute interpreter path or `..` components. Commands in this skill use `python3` for the default case; when `runtime.interpreter` is declared, use that interpreter for `predict.py` syntax checks and smoke tests.

If `predict.py` opens sessions on multiple QIDs, set `runtime.qaic_devices` to `4`; otherwise omit it (default `1`). The only valid values are `1` and `4`. Redesign any pipeline that cannot fit within 4 QIDs.

```json
{"input_type":"tts","runtime":{"qaic_devices":4}}
```

For TTS, also write `reference_voices` — the selectable reference voices for this model. Each `id` must be a voice ID `predict.py` accepts (see the linked TTS and ASR contracts).

| Field | Meaning |
|---|---|
| `id` | Value passed as `--voice-id` |
| `label` | Short human-readable display name |
| `filename` | Path relative to the model dir for a playable preview, or `null` if no playable file exists |
| `kind` | `"file"` or `"builtin"` |
| `preview_available` | `false` when `filename` is `null` |
| `sample_rate`, `duration_s`, `transcript` | Optional metadata; include when known |

Which voices to list:
- Always include the default voice `predict.py` uses. Use `filename: null`, `kind: "builtin"`, `preview_available: false` when it is baked into weights or conditioning (for example `conds.pt`) and no playable WAV exists.
- Add file-backed voices only when they are model-native, short, deduped by file hash, and fit the compiled shape budget (see the linked audio and generative reference - a longer reference clip shrinks the generated-speech budget). For models that condition on a reference transcript, list a voice only if its transcript is known and matches the audio.
- Exclude `venv`, `site-packages`, `tests/data`, packaged test fixtures, duplicate files, and any clip whose transcript is unknown when the model needs one.

```bash
python3 -c "
import json
cfg = {'input_type': 'image'}   # set to 'text', 'audio', or 'tts' as appropriate
# TTS example:
# cfg = {
#   'input_type': 'tts',
#   'reference_voices': [
#     {'id': 'default', 'label': 'Default voice', 'filename': 'ref_audio_default.wav', 'kind': 'file'},
#     {'id': 'builtin', 'label': 'Built-in voice', 'filename': None, 'kind': 'builtin', 'preview_available': False},
#   ],
# }
open('/workspace/<model>/demo_config.json', 'w').write(json.dumps(cfg) + '\n')
print('wrote demo_config.json:', cfg)
"
```

---

## Font handling notes

**`_has_cjk` ranges** — copy exactly as written in the template (do not add ranges):
```python
return any('　' <= c <= '鿿' or '豈' <= c <= '﫿' for c in text)
```
Adding `' ' <= c <= '~'` would match every ASCII character and force CJK font
for all English labels.

**Text measurement** — always use `draw.textbbox`, never `font.getbbox`:
`font.getbbox(text)` returns ink-cell offsets where `top` is typically negative (ascender
above draw origin), causing label pills to be misaligned. `draw.textbbox((x,y), text, font=font)`
returns image-space pixel coordinates at the actual draw position — use those for the
background rectangle.
