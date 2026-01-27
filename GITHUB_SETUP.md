# GitHub Setup Guide

Quick guide to get your project on GitHub and ready for Render deployment.

## Step 1: Initialize Git Repository

If you haven't already:

```bash
# Initialize git
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit - Financial RAG API ready for deployment"
```

## Step 2: Create GitHub Repository

1. Go to [GitHub.com](https://github.com)
2. Click **"New repository"** (or **"+"** → **"New repository"**)
3. Repository name: `financial-rag-api` (or your preferred name)
4. Description: "Financial RAG API with Azure OpenAI and Chroma Cloud"
5. Choose **Public** or **Private**
6. **DO NOT** initialize with README, .gitignore, or license (we already have these)
7. Click **"Create repository"**

## Step 3: Connect Local Repository to GitHub

```bash
# Add remote (replace YOUR_USERNAME and REPO_NAME)
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

## Step 4: Verify Files Are Pushed

Check your GitHub repository to ensure these files are present:

✅ **Required for Render:**
- `app.py` - Main FastAPI application
- `requirements.txt` - Python dependencies
- `runtime.txt` - Python version
- `gunicorn_config.py` - Gunicorn configuration
- `Procfile` - Process file for Render
- `render.yaml` - Render deployment configuration

✅ **Configuration:**
- `config/settings.py` - Configuration management
- `.gitignore` - Git ignore rules

✅ **Documentation:**
- `README.md` - Project documentation
- `RENDER_DEPLOYMENT.md` - Deployment guide

❌ **Should NOT be in GitHub:**
- `.env` - Environment variables (in .gitignore)
- `venv/` - Virtual environment (in .gitignore)
- `__pycache__/` - Python cache (in .gitignore)
- `*.log` - Log files (in .gitignore)

## Step 5: Test Repository

Verify everything works:

```bash
# Clone in a new directory to test
cd /tmp
git clone https://github.com/YOUR_USERNAME/REPO_NAME.git test-clone
cd test-clone

# Verify files are present
ls -la
cat render.yaml
cat Procfile
```

## Step 6: Ready for Render

Once your code is on GitHub, you can:

1. Go to [Render.com](https://render.com)
2. Connect your GitHub account
3. Select your repository
4. Render will auto-detect `render.yaml`
5. Add environment variables
6. Deploy!

## Common Issues

### "Repository not found"
- Check repository name is correct
- Verify you have access (if private repo)
- Ensure you're logged into GitHub

### "Permission denied"
- Use HTTPS with personal access token, or
- Set up SSH keys for GitHub

### Files missing after push
- Check `.gitignore` isn't excluding needed files
- Verify files are committed: `git status`
- Check files are tracked: `git ls-files`

## Next Steps

After GitHub setup:
1. ✅ Code is on GitHub
2. ✅ All required files are present
3. ✅ `.env` is NOT committed (security)
4. → Proceed to [RENDER_DEPLOYMENT.md](./RENDER_DEPLOYMENT.md)

## Security Checklist

Before pushing:
- [ ] `.env` is in `.gitignore`
- [ ] No API keys in code
- [ ] No passwords in code
- [ ] No secrets in commit history
- [ ] Repository is private (if contains sensitive info)

If you accidentally committed secrets:
```bash
# Remove from git history (use with caution)
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch .env" \
  --prune-empty --tag-name-filter cat -- --all

# Force push (destructive - be careful!)
git push origin --force --all
```

## Git Workflow

For future updates:

```bash
# Make changes
# ...

# Stage changes
git add .

# Commit
git commit -m "Description of changes"

# Push to GitHub
git push origin main

# Render will auto-deploy!
```

## Branch Strategy (Optional)

For production deployments:

```bash
# Create production branch
git checkout -b production

# Push production branch
git push origin production

# In Render, deploy from 'production' branch instead of 'main'
```

---

**You're all set!** Your code is on GitHub and ready for Render deployment. 🚀
