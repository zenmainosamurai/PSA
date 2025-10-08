## EXEファイル作成方法

```powershell
.\build_exe.ps1
```

## 何が実行されるか

1. 仮想環境のアクティベート
2. 古いファイルのクリーンアップ
3. PyInstallerでEXE作成
4. 配布用パッケージの作成
5. 必要なファイルをコピー

## 結果

- `distribution/` フォルダに配布用ファイルが作成されます

## 使用方法

1. `distribution/` フォルダを配布先に渡す
2. `CatReactorSimulator.exe` をダブルクリック
