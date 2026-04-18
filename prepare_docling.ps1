# Setup Directories
$basePath = "C:\docling_dist"
$wheelsPath = Join-Path $basePath "wheels"
$modelsPath = Join-Path $basePath "models_cache"
$depsPath = Join-Path $basePath "system_deps"

Write-Host "--- Resuming Download (Checking Existing Files) ---" -ForegroundColor Cyan
# We NO LONGER delete the folder. We only create missing ones.
if (!(Test-Path $wheelsPath)) { New-Item -ItemType Directory -Path $wheelsPath -Force }
if (!(Test-Path $modelsPath)) { New-Item -ItemType Directory -Path $modelsPath -Force }
if (!(Test-Path $depsPath)) { New-Item -ItemType Directory -Path $depsPath -Force }

# 1. Install EasyOCR (Required to fix the crash)
Write-Host "--- Installing Missing Tooling (Online Machine) ---" -ForegroundColor Cyan
pip install -U easyocr docling

# 2. Sync Missing Wheels
Write-Host "--- Syncing Missing Python Wheels ---" -ForegroundColor Cyan
pip download docling[rapidocr] easyocr `
    --dest $wheelsPath `
    --platform win_amd64 `
    --python-version 313 `
    --only-binary=:all:

# 3. Sync Missing Models
Write-Host "--- Syncing Missing Models (Will skip already downloaded ones) ---" -ForegroundColor Cyan
docling-tools models download --all -o $modelsPath

# 4. Check for VC Redist
if (!(Test-Path (Join-Path $depsPath "vc_redist.x64.exe"))) {
    Write-Host "--- Downloading VC Redist ---" -ForegroundColor Cyan
    $vcUrl = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
    Invoke-WebRequest -Uri $vcUrl -OutFile (Join-Path $depsPath "vc_redist.x64.exe")
}

# 5. Final Verification
Write-Host "`n--- VERIFICATION REPORT ---" -ForegroundColor Yellow
$checks = @(
    @{ Name="Formula Model"; Path=(Join-Path $modelsPath "docling-project--CodeFormulaV2") },
    @{ Name="Table Model"; Path=(Join-Path $modelsPath "docling-project--TableFormerV2") },
    @{ Name="OCR Models (Rapid)"; Path=(Join-Path $modelsPath "RapidOcr") },
    @{ Name="Layout Engine"; Path=(Join-Path $modelsPath "ds4sd--docling-models") },
    @{ Name="VC Redist"; Path=(Join-Path $depsPath "vc_redist.x64.exe") }
)

foreach ($c in $checks) {
    if (Test-Path $c.Path) {
        Write-Host "[OK] $($c.Name) found." -ForegroundColor Green
    } else {
        Write-Host "[!!] $($c.Name) MISSING!" -ForegroundColor Red
    }
}

Write-Host "`nCheck complete. Ready to continue!" -ForegroundColor Cyan
