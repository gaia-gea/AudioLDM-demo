param([ValidateSet('GPU','CPU')][string]$Device = 'GPU')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $projectRoot '.venv-audioldm\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    py -3.12 -m venv (Join-Path $projectRoot '.venv-audioldm')
    if ($LASTEXITCODE -ne 0) { throw 'Install 64-bit Python 3.12 and try again.' }
}
$index = if ($Device -eq 'GPU') { 'https://download.pytorch.org/whl/cu124' } else { 'https://download.pytorch.org/whl/cpu' }
& $pythonExe -m pip install --no-cache-dir torch==2.6.0 --index-url $index
if ($LASTEXITCODE -ne 0) { throw 'Could not install PyTorch.' }
& $pythonExe -m pip install --no-cache-dir -r (Join-Path $PSScriptRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Could not install the dependencies.' }
& $pythonExe (Join-Path $PSScriptRoot 'engine.py') --diagnose
if ($LASTEXITCODE -ne 0) { throw 'Hardware diagnostics failed.' }
