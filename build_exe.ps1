# ===================================================================
# CatReactorSimulator EXE作成配布パッケージ作成スクリプト
# 使用方法: .\build_exe.ps1
# ===================================================================

Write-Host "=== CatReactorSimulator EXE作成開始 ===" -ForegroundColor Green

# エラー時に停止
$ErrorActionPreference = "Stop"

try {
    # 1. 仮想環境のアクティベート
    Write-Host "1. 仮想環境をアクティベート中..." -ForegroundColor Yellow
    if (-not (Test-Path ".venv")) {
        Write-Host "エラー: .venvフォルダが見つかりません。" -ForegroundColor Red
        exit 1
    }
    
    # 2. 古いファイルのクリーンアップ
    Write-Host "2. 古いファイルをクリーンアップ中..." -ForegroundColor Yellow
    if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }
    if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
    if (Test-Path "distribution") { Remove-Item -Recurse -Force "distribution" }

    # 3. PyInstallerでEXE作成
    Write-Host "3. PyInstallerでEXE作成中（数分かかります）..." -ForegroundColor Yellow
    & ".\.venv\Scripts\python.exe" -m PyInstaller CatReactorSimulator.spec --clean --noconfirm

    if ($LASTEXITCODE -ne 0) {
        throw "PyInstallerの実行に失敗しました"
    }

    # 4. 配布パッケージの作成
    Write-Host "4. 配布パッケージを作成中..." -ForegroundColor Yellow
    
    # 配布フォルダを作成
    $distributionDir = "distribution"
    New-Item -ItemType Directory -Path $distributionDir -Force | Out-Null
    
    # EXEファイルとライブラリをコピー
    Copy-Item -Recurse -Path "dist\CatReactorSimulator\*" -Destination $distributionDir
    
    # main_cond.ymlをコピー（必要な設定ファイル）
    if (Test-Path "main_cond.yml") {
        Copy-Item "main_cond.yml" "$distributionDir\main_cond.yml"
        Write-Host "   main_cond.ymlをコピーしました" -ForegroundColor Cyan
    }
    
    # dataフォルダをコピー（高速化のため必要）
    if (Test-Path "data") {
        Copy-Item -Recurse -Path "data" -Destination $distributionDir
        Write-Host "   dataフォルダをコピーしました" -ForegroundColor Cyan
    }
    
    # 必要な条件ファイルをコピー（サンプルのみ）
    if (Test-Path "conditions") {
        $conditionsTarget = "$distributionDir\conditions"
        New-Item -ItemType Directory -Path $conditionsTarget -Force | Out-Null
        
        # 5_08_mod_logging2 をサンプルとしてコピー
        if (Test-Path "conditions\5_08_mod_logging2") {
            Copy-Item -Recurse -Path "conditions\5_08_mod_logging2" -Destination $conditionsTarget
            Write-Host "   サンプル条件をコピーしました" -ForegroundColor Cyan
        }
    }
    
    
    # 出力フォルダ構造を作成
    New-Item -ItemType Directory -Path "$distributionDir\output\logs" -Force | Out-Null

    # 5. 結果表示
    Write-Host "`n=== 完了 ===" -ForegroundColor Green
    

} catch {
    Write-Host "`nエラーが発生しました: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
