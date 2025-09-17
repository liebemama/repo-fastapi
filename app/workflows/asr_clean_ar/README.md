# arabic_asr_clean

**Version:** 1.0.0
**Tags:** arabic, asr, clean, spellcheck
**Description:** Arabic ASR → normalize → spellcheck (returns final cleaned text)

## Overview
This workflow is defined by:
- **manifest:** `app/workflows/asr_clean_ar/manifest.json`
- **sequence:** `app/workflows/asr_clean_ar/workflow.json`

**Steps:** 3

## How it runs

### Option A — by preset name (if you expose it)
POST `/workflow/run`
```json
{
  "preset": "arabic_asr_clean",
  "inputs": {}
}
```

### Option B — by explicit sequence
POST `/workflow/run`
```json
{
  "name": "arabic_asr_clean",
  "sequence": [
    {
      "name": "asr",
      "plugin": "whisper",
      "task": "speech-to-text",
      "payload": {
        "audio_url": "{audio_url}",
        "language": "ar"
      }
    },
    {
      "name": "norm",
      "plugin": "text_tools",
      "task": "arabic_normalize",
      "payload": {
        "text": "{asr.text}"
      }
    },
    {
      "name": "spell",
      "plugin": "text_tools",
      "task": "spellcheck_ar",
      "payload": {
        "text": "{norm.text}"
      }
    }
  ],
  "return": "spell"
}
```

> The API base for workflows is `/workflow` (e.g. `POST /workflow/run`).

## Notes
- Placeholders like `{{ asr.text }}` or `{{audio_url}}` are supported by the router.
- Make sure any referenced plugins are installed/available.
