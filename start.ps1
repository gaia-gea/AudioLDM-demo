$ErrorActionPreference = 'Stop'
$venvRoot = Join-Path $PSScriptRoot '.venv-audioldm'
$pythonExe = Join-Path $venvRoot 'Scripts\python.exe'
$legacyPython = Join-Path (Split-Path -Parent $PSScriptRoot) '.venv-audioldm\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe) -and (Test-Path -LiteralPath $legacyPython)) {
    $pythonExe = $legacyPython
}
if (-not (Test-Path -LiteralPath $pythonExe)) { throw 'Run install.ps1 first.' }
& $pythonExe (Join-Path $PSScriptRoot 'demo.py')
if ($LASTEXITCODE -ne 0) { throw 'The demo could not start. Check the preceding error.' }
