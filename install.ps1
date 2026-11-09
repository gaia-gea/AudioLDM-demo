param([ValidateSet('Auto', 'GPU', 'CPU')][string]$Device = 'Auto')
$ErrorActionPreference = 'Stop'

if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64' -and $env:PROCESSOR_ARCHITEW6432 -ne 'AMD64') {
    throw 'This installer requires x64 Windows.'
}

$runtime = Join-Path $PSScriptRoot '.cache\runtime'
$uvVersion = '0.9.26'
$uvExe = Join-Path $runtime "uv-$uvVersion.exe"
$venv = Join-Path $runtime 'venv'
$pythonExe = Join-Path $venv 'Scripts\python.exe'
$ready = Join-Path $runtime 'ready'
New-Item -ItemType Directory -Path $runtime -Force | Out-Null
if (-not (Test-Path -LiteralPath $uvExe -PathType Leaf)) {
    $archive = Join-Path $runtime ("uv-" + [guid]::NewGuid() + '.zip')
    $partialExe = "$archive.exe"
    $url = "https://github.com/astral-sh/uv/releases/download/$uvVersion/uv-x86_64-pc-windows-msvc.zip"
    try {
        Write-Host 'Downloading the local Python installer (uv)...'
        curl.exe --fail --location --retry 3 --connect-timeout 15 --output $archive $url
        if ($LASTEXITCODE -ne 0) { throw 'Could not download uv from GitHub. Check your connection and retry.' }
        $expectedHash = 'eb02fd95d8e0eed462b4a67ecdd320d865b38c560bffcda9a0b87ec944bdf036'
        if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $expectedHash) {
            throw 'The uv download did not match its expected checksum.'
        }
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        $zip = [IO.Compression.ZipFile]::OpenRead($archive)
        try {
            $entry = $zip.Entries | Where-Object { $_.Name -eq 'uv.exe' } | Select-Object -First 1
            if (-not $entry) { throw 'uv.exe is missing from the downloaded archive.' }
            [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $partialExe)
        } finally {
            $zip.Dispose()
        }
        Move-Item -LiteralPath $partialExe -Destination $uvExe
    } finally {
        foreach ($temporary in @($archive, $partialExe)) {
            if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary }
        }
    }
}

$env:UV_PYTHON_INSTALL_DIR = Join-Path $runtime 'python'
$env:UV_CACHE_DIR = Join-Path $runtime 'uv-cache'

function Install-Torch([string]$Backend) {
    Write-Host "Installing PyTorch ($Backend)..."
    & $uvExe --no-config --quiet pip install --python $pythonExe --index-url "https://download.pytorch.org/whl/$Backend" "torch==2.6.0+$Backend"
    if ($LASTEXITCODE -ne 0) { throw "Could not install PyTorch ($Backend)." }
}

$backend = 'cpu'
if ($Device -eq 'GPU') {
    $backend = 'cu124'
} elseif ($Device -eq 'Auto') {
    $smi = Get-Command nvidia-smi -CommandType Application -ErrorAction SilentlyContinue
    if ($smi) {
        $gpuRows = & $smi.Source --query-gpu=driver_version,compute_cap --format=csv,noheader,nounits
        if ($LASTEXITCODE -eq 0) {
            foreach ($row in $gpuRows) {
                if ($row -match '^\s*(\d+\.\d+)\s*,\s*(\d+)\.\d+\s*$') {
                    if ([version]$Matches[1] -ge [version]'551.61' -and [int]$Matches[2] -in @(5,6,7,8,9)) {
                        $backend = 'cu124'
                        break
                    }
                }
            }
        }
    }
}

if (Test-Path -LiteralPath $ready) { Remove-Item -LiteralPath $ready }
Write-Host 'Preparing local Python 3.12...'
& $uvExe --no-config venv --python 3.12 --managed-python --allow-existing $venv
if ($LASTEXITCODE -ne 0) { throw 'Could not prepare local Python 3.12. Check your connection and retry.' }

Install-Torch $backend
Write-Host 'Installing the demo dependencies...'
& $uvExe --no-config --quiet pip install --python $pythonExe --index-url 'https://pypi.org/simple' -r (Join-Path $PSScriptRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Could not install the demo dependencies.' }
if ($backend -eq 'cu124') {
    & $pythonExe -c 'import torch; torch.ones(1).cuda().item()'
    if ($LASTEXITCODE -ne 0) {
        if ($Device -eq 'GPU') { throw 'CUDA could not run on this computer. Retry with -Device CPU.' }
        Write-Warning 'CUDA could not run on this computer. Installing the CPU version.'
        Install-Torch 'cpu'
    }
}
& $pythonExe (Join-Path $PSScriptRoot 'engine.py') --diagnose
if ($LASTEXITCODE -ne 0) { throw 'Hardware diagnostics failed.' }
Set-Content -LiteralPath $ready -Value 'ready' -Encoding ASCII
Write-Host 'Setup complete. Run start.ps1 to open the demo.'
