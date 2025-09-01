# Build and test script for local package development

Write-Host "🔨 Building package locally..." -ForegroundColor Green

# Clean previous builds
Write-Host "🧹 Cleaning previous builds..." -ForegroundColor Yellow
Remove-Item -Path "build", "dist", "*.egg-info" -Recurse -Force -ErrorAction SilentlyContinue

# Install build requirements
Write-Host "📦 Installing build requirements..." -ForegroundColor Yellow
pip install build twine

# Build the package
Write-Host "🏗️ Building package..." -ForegroundColor Yellow
python -m build

# Check the built package
Write-Host "✅ Checking package..." -ForegroundColor Yellow
python -m twine check dist/*

Write-Host "📁 Built files:" -ForegroundColor Cyan
Get-ChildItem -Path "dist" | Format-Table Name, Length, LastWriteTime

Write-Host ""
Write-Host "🧪 To test the package locally:" -ForegroundColor Green
Write-Host "   pip install dist/drr_toolkit-*.whl" -ForegroundColor White
Write-Host ""
Write-Host "📤 To upload to Test PyPI:" -ForegroundColor Green
Write-Host "   python -m twine upload --repository testpypi dist/*" -ForegroundColor White
Write-Host ""
Write-Host "🚀 To upload to PyPI:" -ForegroundColor Green
Write-Host "   python -m twine upload dist/*" -ForegroundColor White
Write-Host ""
Write-Host "✨ Or just create a GitHub release to trigger automatic publishing!" -ForegroundColor Magenta
