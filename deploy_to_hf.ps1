$HF_USERNAME = "ayushthecaringnihilist"
$SPACE_NAME  = "rag-ai-teaching"
$PROJECT_DIR = "C:\Project- RAG Based Al Teaching"
$CLONE_DIR   = "C:\hf-deploy\$SPACE_NAME"

Write-Host ""
Write-Host "=== RAG Teaching Assistant - HuggingFace Deploy ===" -ForegroundColor Cyan
Write-Host ""

$HF_TOKEN = Read-Host "Paste your HuggingFace WRITE token (hf_xxx...)"

if (-not $HF_TOKEN.StartsWith("hf_")) {
    Write-Host "Token should start with hf_ - please check and rerun." -ForegroundColor Red
    exit 1
}

Write-Host "Token accepted." -ForegroundColor Green

# Clone the Space repo
Write-Host ""
Write-Host "Cloning Space repo..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "C:\hf-deploy" | Out-Null

if (Test-Path $CLONE_DIR) {
    Write-Host "  Removing old clone..." -ForegroundColor Gray
    Remove-Item -Recurse -Force $CLONE_DIR
}

$cloneUrl = "https://" + $HF_USERNAME + ":" + $HF_TOKEN + "@huggingface.co/spaces/" + $HF_USERNAME + "/" + $SPACE_NAME
git clone $cloneUrl $CLONE_DIR

if (-not (Test-Path $CLONE_DIR)) {
    Write-Host "Clone failed - check your token and Space name." -ForegroundColor Red
    exit 1
}

Set-Location $CLONE_DIR

# Configure git-lfs for large model files
git lfs install
git lfs track "*.index"
git lfs track "*.json"

# Copy project files
Write-Host ""
Write-Host "Copying project files..." -ForegroundColor Yellow

Copy-Item "$PROJECT_DIR\main.py"               . -Force
Copy-Item "$PROJECT_DIR\evaluate.py"           . -Force
Copy-Item "$PROJECT_DIR\requirements-prod.txt" . -Force
Copy-Item "$PROJECT_DIR\Dockerfile.spaces"     "Dockerfile" -Force

if (Test-Path "templates") { Remove-Item -Recurse -Force "templates" }
if (Test-Path "static")    { Remove-Item -Recurse -Force "static" }
Copy-Item "$PROJECT_DIR\templates" "templates" -Recurse -Force
Copy-Item "$PROJECT_DIR\static"    "static"    -Recurse -Force

# Copy FAISS model files
Write-Host "Copying FAISS model files..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "models" | Out-Null
Copy-Item "$PROJECT_DIR\models\faiss_with_titles.index"   "models\" -Force
Copy-Item "$PROJECT_DIR\models\faiss_metadata_clean.json" "models\" -Force

# Commit and push
Write-Host ""
Write-Host "Pushing to HuggingFace..." -ForegroundColor Yellow

git add .
git commit -m "Deploy RAG AI Teaching Assistant"

$pushUrl = "https://" + $HF_USERNAME + ":" + $HF_TOKEN + "@huggingface.co/spaces/" + $HF_USERNAME + "/" + $SPACE_NAME
git push $pushUrl main

Write-Host ""
Write-Host "=== DONE ===" -ForegroundColor Green
Write-Host ""
Write-Host "Space is building at:" -ForegroundColor Green
Write-Host "  https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME" -ForegroundColor Cyan
Write-Host ""
Write-Host "NEXT STEP - Add your Groq key as a secret:" -ForegroundColor Yellow
Write-Host "  https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME/settings" -ForegroundColor Cyan
Write-Host "  Secret name:  GROQ_API_KEY" -ForegroundColor White
Write-Host "  Secret value: your Groq API key from console.groq.com" -ForegroundColor White
Write-Host ""
