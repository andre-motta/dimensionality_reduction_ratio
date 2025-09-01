#!/bin/bash
# Build and test script for local package development

echo "🔨 Building package locally..."

# Clean previous builds
rm -rf build/ dist/ *.egg-info/

# Install build requirements
echo "📦 Installing build requirements..."
pip install build twine

# Build the package
echo "🏗️ Building package..."
python -m build

# Check the built package
echo "✅ Checking package..."
python -m twine check dist/*

echo "📁 Built files:"
ls -la dist/

echo ""
echo "🧪 To test the package locally:"
echo "   pip install dist/drr_toolkit-*.whl"
echo ""
echo "📤 To upload to Test PyPI:"
echo "   python -m twine upload --repository testpypi dist/*"
echo ""
echo "🚀 To upload to PyPI:"
echo "   python -m twine upload dist/*"
echo ""
echo "✨ Or just create a GitHub release to trigger automatic publishing!"
