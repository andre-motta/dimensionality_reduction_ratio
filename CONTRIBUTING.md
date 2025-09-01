# Contributing to Intrinsic Dimension Analysis with DRR Metrics

Thank you for your interest in contributing to this project! We welcome contributions of all kinds, from bug reports and feature requests to code improvements and documentation updates.

## 🚀 Quick Start for Contributors

1. **Fork the Repository**
   ```bash
   # Fork the repository on GitHub by clicking the "Fork" button
   # Then clone your fork (replace 'your-username' with your GitHub username)
   git clone https://github.com/your-username/dimensionality_reduction_ratio.git
   cd dimensionality_reduction_ratio
   
   # Add the original repository as upstream remote
   git remote add upstream https://github.com/andre-motta/dimensionality_reduction_ratio.git
   ```

2. **Set Up Development Environment**
   ```bash
   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies including development tools
   pip install -r requirements.txt
   ```

3. **Create a Feature Branch**
   ```bash
   # Make sure you're on main and up to date
   git checkout main
   git pull upstream main
   
   # Create your feature branch
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

## 📋 Development Guidelines

### Code Style and Quality

We maintain high code quality standards using automated tools:

#### **Formatting and Linting**
```bash
# Format code with Black
black src/

# Sort imports with isort
isort src/

# Check for issues with flake8
flake8 src/ --max-line-length=127

# Run all checks
black src/ && isort src/ && flake8 src/ --max-line-length=127
```

#### **Code Standards**
- **Line Length**: Maximum 127 characters
- **Import Sorting**: Use isort for consistent import organization
- **Code Formatting**: Use Black for consistent code formatting
- **Documentation**: All public functions must have docstrings
- **Type Hints**: Use type hints for function parameters and returns
- **Error Handling**: Use specific exception types, avoid bare `except`

### Testing

#### **Manual Testing**
```bash
# Test CLI functionality
cd src
python main.py --help
python main.py batch --help
python main.py single --help

# Test with sample data
python main.py single ../data/optimize/config/SS-A.csv
```

#### **Algorithm Validation**
When contributing algorithm changes:
- Test with small, medium, and large datasets
- Verify DRR calculations: `DRR = 1 - (I/R)`
- Check that intrinsic dimensions are reasonable (I ≤ R)
- Validate against known dataset characteristics

### Documentation

#### **Code Documentation**
- Use clear, descriptive docstrings for all public functions
- Include parameter types and descriptions
- Provide usage examples in docstrings
- Document any algorithmic assumptions or limitations

#### **README Updates**
When adding features, update relevant sections:
- Usage examples
- API reference
- Installation instructions
- Command-line options

## 🔧 Development Workflow

### 1. **Before Starting Development**
```bash
# Ensure you're on main and sync with upstream
git checkout main
git pull upstream main

# Push updates to your fork
git push origin main

# Create your feature branch
git checkout -b feature/descriptive-name
```

### 2. **During Development**
```bash
# Make your changes, then run quality checks
black src/
isort src/
flake8 src/ --max-line-length=127

# Test your changes
cd src
python main.py --help  # Basic functionality test
```

### 3. **Before Committing**
```bash
# Stage your changes
git add .

# Commit with descriptive message
git commit -s -m "Add feature: descriptive summary

Detailed description of what was changed and why.
Fixes #issue-number (if applicable)"
```

### 4. **Submitting Changes**
```bash
# Push your feature branch to your fork
git push origin feature/descriptive-name

# Create Pull Request on GitHub
# 1. Go to your fork on GitHub
# 2. Click "Compare & pull request" button
# 3. Ensure the base repository is andre-motta/dimensionality_reduction_ratio
# 4. Ensure the base branch is "main"
# 5. Provide clear title and description
# 6. Reference any related issues
# 7. Include testing notes
```

#### **Pull Request Guidelines**
- **Title**: Use clear, descriptive titles (e.g., "Add cosine distance metric support")
- **Description**: Include:
  - What changes were made and why
  - How to test the changes
  - Any breaking changes or migration notes
  - Screenshots/examples if relevant
- **References**: Link related issues using `Fixes #123` or `Addresses #456`
- **Ready for Review**: Mark as draft if still in progress

### 5. **Keeping Your Fork Updated**
```bash
# Regularly sync your fork with the upstream repository
git checkout main
git pull upstream main
git push origin main

# If you need to update a feature branch with latest changes
git checkout feature/your-feature-name
git rebase main  # or: git merge main
git push origin feature/your-feature-name --force-with-lease  # if rebased
```

#### **Handling Merge Conflicts**
```bash
# If there are conflicts during rebase/merge
git status  # See conflicted files
# Edit files to resolve conflicts
git add resolved-file.py
git rebase --continue  # if rebasing
# or git commit  # if merging
```
# - Reference any related issues
# - Include testing notes
```

## 🐛 Bug Reports

When reporting bugs, please include:

### **Required Information**
- **Python Version**: Output of `python --version`
- **Package Versions**: Output of `pip list`
- **Operating System**: Windows, macOS, Linux distribution
- **Error Messages**: Complete error traceback
- **Dataset Information**: Size, type, any special characteristics

### **Bug Report Template**
```markdown
## Bug Description
Brief description of the issue

## Steps to Reproduce
1. Step one
2. Step two
3. Step three

## Expected Behavior
What you expected to happen

## Actual Behavior
What actually happened (include error messages)

## Environment
- Python Version: 
- OS: 
- Package Versions: (pip list output)

## Additional Context
Any other relevant information
```

## ✨ Feature Requests

### **Feature Request Template**
```markdown
## Feature Description
Clear description of the proposed feature

## Use Case
Why is this feature needed? What problem does it solve?

## Proposed Implementation
Any ideas for how this could be implemented

## Alternative Solutions
Other approaches you've considered

## Additional Context
Any other relevant information
```

## 📊 Algorithm Contributions

### **For Algorithm Improvements**
When contributing to the core intrinsic dimension estimation:

1. **Research Basis**: Cite relevant papers or mathematical foundations
2. **Validation**: Test against known datasets with expected results
3. **Performance**: Consider computational complexity and memory usage
4. **Backward Compatibility**: Ensure existing functionality is preserved

### **For New Metrics**
When adding new dimensionality metrics:

1. **Mathematical Definition**: Provide clear mathematical formulation
2. **Interpretation**: Explain what the metric represents
3. **Validation**: Test with synthetic data of known dimensionality
4. **Documentation**: Add usage examples and interpretation guidelines

## 🧪 Testing Contributions

### **Test Data Guidelines**
- Include small test datasets for CI/CD
- Document expected results for validation
- Consider edge cases (empty data, single column, etc.)
- Test with different data types (numerical, categorical, mixed)

### **Performance Testing**
- Test with various dataset sizes (100, 1K, 10K, 100K+ rows)
- Monitor memory usage for large datasets
- Validate sampling strategies maintain statistical properties

## 📝 Documentation Contributions

### **README Improvements**
- Keep examples up-to-date with current API
- Add new usage patterns as features are added
- Improve clarity and organization

### **Code Comments**
- Explain complex algorithmic steps
- Document mathematical formulations
- Clarify design decisions and trade-offs

## 🤝 Community Guidelines

### **Code of Conduct**
- Be respectful and constructive in all interactions
- Welcome newcomers and help them get started
- Focus on the technical merit of contributions
- Provide helpful feedback on pull requests

### **Communication**
- Use GitHub Issues for bug reports and feature requests
- Use Pull Request comments for code-specific discussions
- Be specific and provide context in all communications

## 🔍 Review Process

### **Pull Request Review**
Pull requests will be reviewed for:

1. **Code Quality**: Adherence to style guidelines and best practices
2. **Functionality**: Correctness and completeness of implementation
3. **Testing**: Adequate testing of new features
4. **Documentation**: Clear documentation of changes
5. **Compatibility**: Backward compatibility and integration

### **Approval Process**
- All PRs require at least one approval from a maintainer
- CI/CD checks must pass
- Code style checks must pass
- Manual testing verification may be required
- Only maintainers can merge PRs into the main repository

### **After Your PR is Merged**
```bash
# Clean up your local branches after successful merge
git checkout main
git pull upstream main
git push origin main
git branch -d feature/your-feature-name  # Delete local branch
git push origin --delete feature/your-feature-name  # Delete remote branch
```

## 📚 Additional Resources

### **Useful Links**
- [Levina & Bickel (2005) Paper](https://example.com/paper) - Original algorithm reference
- [NumPy Documentation](https://numpy.org/doc/) - For numerical computations
- [Pandas Documentation](https://pandas.pydata.org/docs/) - For data manipulation
- [Click Documentation](https://click.palletsprojects.com/) - For CLI development

### **Learning Resources**
- **Intrinsic Dimensionality**: Understanding the mathematical concepts
- **Scientific Python**: Best practices for numerical computing
- **Software Engineering**: Clean code and testing practices

## 🎯 Getting Help

### **Where to Get Help**
- **GitHub Issues**: For project-specific questions and bug reports
- **GitHub Discussions**: For general discussions about usage and algorithms  
- **Pull Request Comments**: For questions about specific code changes
- **Stack Overflow**: For general Python/scientific computing questions (tag: intrinsic-dimension)

### **Mentorship**
New contributors are welcome! If you're new to:
- **Open Source**: We'll help you learn the fork/PR workflow and best practices
- **Scientific Computing**: We'll explain the algorithms and mathematics
- **Python Development**: We'll guide you through code quality standards
- **Git/GitHub**: We'll help you navigate version control and collaboration

### **First-Time Contributors**
Look for issues labeled:
- `good first issue`: Perfect for newcomers
- `help wanted`: Community input needed
- `documentation`: Non-code contributions welcome

---

## 🙏 Recognition

Contributors will be recognized in:
- **Contributors section** in README.md
- **Release notes** for significant contributions
- **Commit history** with proper attribution

Thank you for contributing to making intrinsic dimension analysis more accessible and robust!
