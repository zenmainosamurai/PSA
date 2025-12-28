# common/ - 共通定義

プロジェクト全体で使用する定数・パス・翻訳辞書を定義。

## ファイル一覧

| ファイル | 説明 |
|---------|------|
| `constants.py` | 物理定数（気体定数、標準圧力）、単位変換係数（MPa→Pa等） |
| `paths.py` | ディレクトリパス定義（CONDITIONS_DIR, OUTPUT_DIR, DATA_DIR） |
| `translations.py` | 日本語変数名と単位の辞書（CSV/Excel出力時のヘッダー用） |
| `unit_conversion.py` | 単位変換関数 |

## 主な定数（constants.py）

```
GAS_CONSTANT = 8.314          # 気体定数 [J/(mol·K)]
STANDARD_PRESSURE = 101325    # 標準圧力 [Pa]
CELSIUS_TO_KELVIN_OFFSET = 273.15
MPA_TO_PA = 1e6               # MPa → Pa
MINUTE_TO_SECOND = 60         # min → s
```
