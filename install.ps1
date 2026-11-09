param([ValidateSet('GPU','CPU')][string]$Device = 'GPU')
$ErrorActionPreference = 'Stop'
$venvRoot = Join-Path $PSScriptRoot '.venv-audioldm'
$pythonExe = Join-Path $venvRoot 'Scripts\python.exe'
$legacyPython = Join-Path (Split-Path -Parent $PSScriptRoot) '.venv-audioldm\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe) -and (Test-Path -LiteralPath $legacyPython)) {
    $pythonExe = $legacyPython
}
if (-not (Test-Path -LiteralPath $pythonExe)) {
    $pythonOnPath = Get-Command python -CommandType Application -ErrorAction SilentlyContinue
    $pythonCheck = 'import struct, sys; sys.exit(not (sys.version_info[:2] == (3, 12) and struct.calcsize("P") == 8))'
    $usePythonOnPath = $false
    if ($pythonOnPath) {
        & $pythonOnPath.Source -c $pythonCheck
        $usePythonOnPath = $LASTEXITCODE -eq 0
    }
    if ($usePythonOnPath) {
        & $pythonOnPath.Source -m venv $venvRoot
    } elseif (Get-Command py -CommandType Application -ErrorAction SilentlyContinue) {
        py -3.12-64 -m venv $venvRoot
    } else {
        throw 'Install 64-bit Python 3.12 and try again.'
    }
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the virtual environment with 64-bit Python 3.12.' }
}
$index = if ($Device -eq 'GPU') { 'https://download.pytorch.org/whl/cu124' } else { 'https://download.pytorch.org/whl/cpu' }
& $pythonExe -m pip install --no-cache-dir torch==2.6.0 --index-url $index
if ($LASTEXITCODE -ne 0) { throw 'Could not install PyTorch.' }
& $pythonExe -m pip install --no-cache-dir -r (Join-Path $PSScriptRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Could not install the dependencies.' }
& $pythonExe (Join-Path $PSScriptRoot 'engine.py') --diagnose
if ($LASTEXITCODE -ne 0) { throw 'Hardware diagnostics failed.' }
