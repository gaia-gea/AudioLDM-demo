$ErrorActionPreference = 'Stop'

$runtime = Join-Path $PSScriptRoot '.cache\runtime'
$managedPython = Join-Path $runtime 'venv\Scripts\python.exe'
$ready = Join-Path $runtime 'ready'
$legacyPythons = @(
    (Join-Path $PSScriptRoot '.venv-audioldm\Scripts\python.exe'),
    (Join-Path (Split-Path -Parent $PSScriptRoot) '.venv-audioldm\Scripts\python.exe')
)
$pythonExe = $null
if ((Test-Path -LiteralPath $ready) -and (Test-Path -LiteralPath $managedPython)) {
    $pythonExe = $managedPython
} else {
    foreach ($candidate in $legacyPythons) {
        if (Test-Path -LiteralPath $candidate) {
            & $candidate -c 'import sys' 2>$null
            if ($LASTEXITCODE -eq 0) {
                $pythonExe = $candidate
                break
            }
        }
    }
}
if (-not $pythonExe) {
    & (Join-Path $PSScriptRoot 'install.ps1')
    $pythonExe = $managedPython
}

& $pythonExe (Join-Path $PSScriptRoot 'demo.py')
if ($LASTEXITCODE -ne 0) { throw 'The demo could not start. Check the preceding error.' }

