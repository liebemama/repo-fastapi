# 🔗 Plugins Index

_Auto-generated. Do not edit manually._


| Plugin | Provider | Tasks | README | manifest.json | plugin.py |
|-------:|:---------|:------|:------:|:-------------:|:---------:|
| **dummy** | dummy | `ping` | [README](../app/plugins/dummy/README.md) | [manifest](../app/plugins/dummy/manifest.json) | [plugin](../app/plugins/dummy/plugin.py) |
| **neu_server** | neu_server | `summarize, classify` | [README](../app/plugins/neu_server/README.md) | [manifest](../app/plugins/neu_server/manifest.json) | [plugin](../app/plugins/neu_server/plugin.py) |
| **text_tools** | — | `postprocess` | [README](../app/plugins/text_tools/README.md) | [manifest](../app/plugins/text_tools/manifest.json) | [plugin](../app/plugins/text_tools/plugin.py) |
| **whisper** | whisper | `speech-to-text` | [README](../app/plugins/whisper/README.md) | [manifest](../app/plugins/whisper/manifest.json) | [plugin](../app/plugins/whisper/plugin.py) |

## How to generate
```bash
python tools/build_plugins_index.py
```

## Force-refresh all plugin READMEs
```bash
python tools/build_plugins_index.py --force-readme
```
