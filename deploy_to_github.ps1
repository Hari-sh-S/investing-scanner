# Quick GitHub Setup Script
# Run this to push your code to GitHub

Write-Host "🚀 Investing Scanner - GitHub Deployment Script" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""

# Get GitHub username and repo name
$username = Read-Host "Enter your GitHub username"
$reponame = Read-Host "Enter repository name (default: investing-scanner)"

if ([string]::IsNullOrWhiteSpace($reponame)) {
    $reponame = "investing-scanner"
}

$remote_url = "https://github.com/$username/$reponame.git"

Write-Host ""
Write-Host "Setting up remote repository..." -ForegroundColor Yellow
Write-Host "URL: $remote_url" -ForegroundColor Cyan

# Add remote
git remote add origin $remote_url

# Rename branch to main
git branch -M main

Write-Host ""
Write-Host "Pushing code to GitHub..." -ForegroundColor Yellow

# Push to GitHub
git push -u origin main

Write-Host ""
Write-Host "✅ Done! Your code is now on GitHub!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Visit: https://github.com/$username/$reponame" -ForegroundColor Cyan
Write-Host "2. Go to https://streamlit.io/cloud to deploy" -ForegroundColor Cyan
Write-Host "3. Read DEPLOYMENT_GUIDE.md for detailed instructions" -ForegroundColor Cyan
Write-Host ""
