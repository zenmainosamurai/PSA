# tests/ - テストコード

シミュレーションの動作確認・回帰テスト。

## ファイル一覧

| ファイル | 説明 |
|---------|------|
| `test_simulation.py` | 各運転モードの動作確認テスト |
| `test_full_simulation.py` | 全工程シミュレーションテスト |
| `test_index_comparison.py` | インデックス修正前後の比較テスト |
| `test_index_quick.py` | クイックテスト |
| `test_minimal.py` | 最小限のテスト |

## テスト実行方法

```bash
# 個別テスト
python tests/test_simulation.py

# 全工程テスト
python tests/test_full_simulation.py
```

## test_simulation.py

各運転モードが正しく動作するかを確認。

- 停止モード
- 流通吸着モード
- バッチ吸着モード
- 均圧モード
- 真空脱着モード

## test_index_comparison.py

リファクタリング前後で計算結果が一致するかを確認。
`test_results_before.json`と比較。
