$HF_USERNAME = "ayushthecaringnihilist"
$SPACE_NAME  = "rag-ai-teaching"
$PROJECT_DIR = "C:\Project- RAG Based Al Teaching"
$CLONE_DIR   = "C:\hf-deploy\$SPACE_NAME"

Write-Host "=== Pushing fixes to HuggingFace Space ===" -ForegroundColor Cyan

$HF_TOKEN = Read-Host "Paste your HuggingFace WRITE token"

# Go to existing clone
Set-Location $CLONE_DIR

# Update changed files
Write-Host "Updating files..." -ForegroundColor Yellow
Copy-Item "$PROJECT_DIR\requirements-prod.txt" . -Force
Copy-Item "$PROJECT_DIR\Dockerfile.spaces"     "Dockerfile" -Force
Copy-Item "$PROJECT_DIR\main.py"               . -Force

# Check if models folder exists and has files
if (-not (Test-Path "models\faiss_with_titles.index")) {
    Write-Host "Models folder missing - copying from project..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path "models" | Out-Null
    Copy-Item "$PROJECT_DIR\models\faiss_with_titles.index"   "models\" -Force
    Copy-Item "$PROJECT_DIR\models\faiss_metadata_clean.json" "models\" -Force
    Write-Host "Models copied." -ForegroundColor Green
} else {
    Write-Host "Models already present - skipping." -ForegroundColor Green
}

# Push
Write-Host "Pushing..." -ForegroundColor Yellow
git add .
git commit -m "Fix: add einops, pin numpy<2, copy models in Dockerfile"

$pushUrl = "https://" + $HF_USERNAME + ":" + $HF_TOKEN + "@huggingface.co/spaces/" + $HF_USERNAME + "/" + $SPACE_NAME
git push $pushUrl main

Write-Host ""
Write-Host "Done! Space is rebuilding (~8 min):" -ForegroundColor Green
Write-Host "  https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME" -ForegroundColor Cyan
