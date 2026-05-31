# ============================================================
# Deploy to HuggingFace Spaces
# Run this in a normal PowerShell window (not Claude Code)
# ============================================================

$HF_USERNAME = "ayushthecaringnihilist"
$SPACE_NAME  = "rag-ai-teaching"
$PROJECT_DIR = "C:\Project- RAG Based Al Teaching"
$CLONE_DIR   = "C:\hf-deploy\$SPACE_NAME"

Write-Host ""
Write-Host "=== RAG Teaching Assistant — HuggingFace Deploy ===" -ForegroundColor Cyan
Write-Host ""

# Step 1: Get token
$HF_TOKEN = Read-Host "Paste your HuggingFace WRITE token (hf_xxx...)"
if (-not $HF_TOKEN.StartsWith("hf_")) {
    Write-Host "Token should start with hf_ — please check and rerun." -ForegroundColor Red
    exit 1
}

# Step 2: Clone the Space repo
Write-Host ""
Write-Host "Cloning Space repo..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "C:\hf-deploy" | Out-Null
Set-Location "C:\hf-deploy"

# Set git credentials for this session
git config --global credential.helper store
"https://ayushthecaringnihilist:$HF_TOKEN@huggingface.co" | Out-File -FilePath "$env:USERPROFILE\.git-credentials" -Encoding ascii -Append

if (Test-Path $CLONE_DIR) {
    Write-Host "  (removing old clone)" -ForegroundColor Gray
    Remove-Item -Recurse -Force $CLONE_DIR
}
git clone "https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME" $CLONE_DIR
Set-Location $CLONE_DIR

# Step 3: Configure git-lfs for large model files
git lfs install
git lfs track "*.index"
git lfs track "*.json"

# Step 4: Copy project files
Write-Host ""
Write-Host "Copying project files..." -ForegroundColor Yellow

Copy-Item "$PROJECT_DIR\main.py"              . -Force
Copy-Item "$PROJECT_DIR\evaluate.py"          . -Force
Copy-Item "$PROJECT_DIR\requirements-prod.txt" . -Force
Copy-Item "$PROJECT_DIR\Dockerfile.spaces"    "Dockerfile" -Force  # HF needs it named Dockerfile

# Templates and static
if (Test-Path "templates") { Remove-Item -Recurse -Force "templates" }
if (Test-Path "static")    { Remove-Item -Recurse -Force "static" }
Copy-Item "$PROJECT_DIR\templates" "templates" -Recurse -Force
Copy-Item "$PROJECT_DIR\static"    "static"    -Recurse -Force

# Step 5: Copy FAISS model files (tracked by git-lfs)
Write-Host "Copying FAISS model files (this uses git-lfs)..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "models" | Out-Null
Copy-Item "$PROJECT_DIR\models\faiss_with_titles.index"   "models\" -Force
Copy-Item "$PROJECT_DIR\models\faiss_metadata_clean.json" "models\" -Force

# Step 6: Add .gitattributes for lfs
Add-Content ".gitattributes" "models/*.index filter=lfs diff=lfs merge=lfs -text"
Add-Content ".gitattributes" "models/*.json  filter=lfs diff=lfs merge=lfs -text"

# Step 7: Commit and push
Write-Host ""
Write-Host "Pushing to HuggingFace..." -ForegroundColor Yellow
git add .
git commit -m "Deploy RAG AI Teaching Assistant"
git push https://$HF_USERNAME`:$HF_TOKEN@huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME main

Write-Host ""
Write-Host "=== DONE ===" -ForegroundColor Green
Write-Host "Your Space is building at:" -ForegroundColor Green
Write-Host "  https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next: Set your GROQ_API_KEY secret in the Space settings:" -ForegroundColor Yellow
Write-Host "  https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME/settings" -ForegroundColor Cyan
Write-Host ""
