# COW ID + ERPAnalyzer: установка при первом запуске и запуск обоих сайтов.
#
#   powershell -ExecutionPolicy Bypass -File start.ps1           запуск
#   powershell -ExecutionPolicy Bypass -File start.ps1 -Phone    + камера телефона (https :8443)
#
# Первый раз: ставит пакеты Python (с видеокартой NVIDIA или без), скачивает модели
# и демо-ролик с GitHub (1,6 ГБ), проверяет контрольную сумму, распаковывает.
# Дальше — сразу запускает. Нужен Python 3.10–3.13 и интернет только для первого раза.

param([switch]$Phone)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$CV = Join-Path $Root "01-computer-vision-cow-id"
$ERP = Join-Path $Root "02-erp-data-to-profit"

$DataUrl = "https://github.com/DiyasKBTU/AgriTech-Hackathon/releases/download/data-v1/cowid-data.zip"
$DataSha256 = "450a74963bf92de9a3a0561c3f09c48874a9f5aeae6ef1708848708609c9f3d8"
# По этим файлам видно, что архив уже распакован.
$DataMarkers = @("models\reid_mmcows\encoder.pt", "var\gallery_mmcows.json", "data\real\mmcows_0725_from14.mp4")

function Say($text) { Write-Host "`n== $text" -ForegroundColor Cyan }

function Run($exe, [string[]]$arguments) {
    & $exe @arguments
    if ($LASTEXITCODE -ne 0) { throw "Команда завершилась с ошибкой: $exe $($arguments -join ' ')" }
}

function Find-Python {
    foreach ($v in "3.13", "3.12", "3.11", "3.10") {
        try {
            $exe = & py "-$v" -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $exe) { return $exe.Trim() }
        } catch {}
    }
    try {
        $info = & python -c "import sys; print(sys.version_info[0] * 100 + sys.version_info[1], sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $info) {
            $ver, $exe = $info.Trim() -split " ", 2
            if ([int]$ver -ge 310 -and [int]$ver -le 313) { return $exe }
        }
    } catch {}
    return $null
}

function Install-Venv($dir, [scriptblock]$install) {
    $marker = Join-Path $dir ".venv\installed.txt"
    if (Test-Path $marker) { return }
    $py = Find-Python
    if (-not $py) {
        throw "Не найден Python 3.10–3.13. Установите Python 3.13 с https://www.python.org/downloads/ " +
              "(в установщике отметьте «Add python.exe to PATH») или командой: " +
              "winget install -e --id Python.Python.3.13 — и запустите start.ps1 ещё раз."
    }
    Push-Location $dir
    try {
        if (-not (Test-Path ".venv\Scripts\python.exe")) { Run $py @("-m", "venv", ".venv") }
        $vpy = (Resolve-Path ".venv\Scripts\python.exe").Path
        & $install $vpy
        Set-Content -Path $marker -Value (Get-Date -Format s) -Encoding ascii
    } finally { Pop-Location }
}

function Wait-Url($url, $seconds) {
    $deadline = (Get-Date).AddSeconds($seconds)
    while ((Get-Date) -lt $deadline) {
        try { Invoke-WebRequest $url -UseBasicParsing -TimeoutSec 2 | Out-Null; return $true } catch { Start-Sleep 1 }
    }
    return $false
}

# ---------------------------------------------------------------- установка

$free = (Get-PSDrive ((Get-Item $Root).PSDrive.Name)).Free / 1GB
if (-not (Test-Path (Join-Path $CV ".venv\installed.txt")) -and $free -lt 10) {
    Write-Host ("На диске свободно {0:N1} ГБ, для первой установки нужно около 10 ГБ." -f $free) -ForegroundColor Yellow
}

Say "COW ID: пакеты Python"
Install-Venv $CV {
    param($vpy)
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        Write-Host "Видеокарта NVIDIA найдена — ставлю torch с CUDA (около 3 ГБ)."
        $index = "https://download.pytorch.org/whl/cu128"
    } else {
        Write-Host "Видеокарты NVIDIA нет — ставлю torch для процессора (всё работает, но медленнее)."
        $index = "https://download.pytorch.org/whl/cpu"
    }
    Run $vpy @("-m", "pip", "install", "torch", "torchvision", "--index-url", $index)
    Run $vpy @("-m", "pip", "install", "-r", "requirements.txt")
    Run $vpy @("-m", "pip", "install", "--no-deps", "-e", ".")
}

Say "ERPAnalyzer: пакеты Python"
Install-Venv $ERP {
    param($vpy)
    Run $vpy @("-m", "pip", "install", "-e", ".[dev]")
}

Say "COW ID: модели и демо-ролик"
$missing = $DataMarkers | Where-Object { -not (Test-Path (Join-Path $CV $_)) }
if (-not $missing) {
    Write-Host "Уже на месте."
} else {
    $zip = Join-Path $Root "cowid-data.zip"
    $ok = (Test-Path $zip) -and ((Get-FileHash $zip -Algorithm SHA256).Hash -eq $DataSha256)
    for ($try = 1; -not $ok -and $try -le 2; $try++) {
        Write-Host "Скачиваю 1,6 ГБ с GitHub (если прервётся — запустите снова, докачает)."
        # Имя файла без пути: curl не открывает пути с русскими буквами (C:\Users\Диас\...).
        Push-Location $Root
        try { & curl.exe -L --fail --retry 5 -C - -o "cowid-data.zip" $DataUrl } finally { Pop-Location }
        $ok = (Test-Path $zip) -and ((Get-FileHash $zip -Algorithm SHA256).Hash -eq $DataSha256)
        if (-not $ok -and (Test-Path $zip)) {
            Write-Host "Контрольная сумма не совпала — скачиваю заново." -ForegroundColor Yellow
            Remove-Item $zip
        }
    }
    if (-not $ok) { throw "Не удалось скачать $DataUrl" }
    Write-Host "Контрольная сумма совпала. Распаковываю."
    Run (Join-Path $CV ".venv\Scripts\python.exe") @("-c",
        "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])", $zip, $CV)
    Remove-Item $zip
}

# ---------------------------------------------------------------- запуск

function Start-Site($title, $dir, $command, $url) {
    if (Wait-Url $url 1) { Write-Host "$title уже запущен: $url"; return }
    $script = "`$Host.UI.RawUI.WindowTitle = '$title'; `$env:PYTHONIOENCODING = 'utf-8'; " +
              "& '.\.venv\Scripts\python.exe' $command"
    Start-Process powershell -WorkingDirectory $dir -ArgumentList @("-NoExit", "-Command", $script)
    if (Wait-Url $url 90) { Write-Host "$title — $url" -ForegroundColor Green }
    else { Write-Host "$title не ответил за 90 секунд — посмотрите его окно." -ForegroundColor Yellow }
}

Say "Запуск"
$cowid = "-m cowid.cli serve" + $(if ($Phone) { " --phone" } else { "" })
Start-Site "COW ID" $CV $cowid "http://127.0.0.1:8000/health"
Start-Site "ERPAnalyzer" $ERP "-m erpanalyzer.cli serve" "http://127.0.0.1:8010/"
Start-Process "http://127.0.0.1:8000"
Start-Process "http://127.0.0.1:8010"
Write-Host "`nОстановить — закрыть окна «COW ID» и «ERPAnalyzer» (или Ctrl+C в них)."
if ($Phone) { Write-Host "Адрес для телефона напечатан в окне «COW ID»." }
