# 🚀 Deployment Guide

This guide will walk you through deploying your Investing Scanner to GitHub and hosting it on Streamlit Cloud (free).

## Part 1: Deploy to GitHub

### Step 1: Create a GitHub Repository

1. Go to [GitHub](https://github.com) and sign in
2. Click the **"+"** icon in the top right → **"New repository"**
3. Fill in the details:
   - **Repository name**: `investing-scanner` (or your preferred name)
   - **Description**: "Advanced Backtesting Platform for Indian Stock Market"
   - **Visibility**: Choose **Public** or **Private**
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
4. Click **"Create repository"**

### Step 2: Push Your Code to GitHub

GitHub will show you instructions, but here's what you need to run:

```bash
# Add the GitHub repository as remote
git remote add origin https://github.com/YOUR_USERNAME/investing-scanner.git

# Push your code
git branch -M main
git push -u origin main
```

**Replace `YOUR_USERNAME`** with your actual GitHub username!

#### Alternative: Using GitHub Desktop (Easier)

1. Download [GitHub Desktop](https://desktop.github.com/)
2. Open GitHub Desktop
3. Click **File → Add Local Repository**
4. Browse to `c:\Users\haris\OneDrive\Project Fetch\claude investing`
5. Click **Publish repository**
6. Choose repository name and visibility
7. Click **Publish**

---

## Part 2: Deploy to Streamlit Cloud (Free Hosting)

### Step 1: Prepare Your Streamlit Account

1. Go to [Streamlit Cloud](https://streamlit.io/cloud)
2. Click **"Sign up"** or **"Sign in"**
3. Sign in with your **GitHub account** (recommended)
4. Authorize Streamlit to access your repositories

### Step 2: Deploy Your App

1. After signing in, you'll see the Streamlit Cloud dashboard
2. Click **"New app"** button
3. Fill in the deployment form:

   **Repository**: `YOUR_USERNAME/investing-scanner`  
   **Branch**: `main`  
   **Main file path**: `app.py`  
   **App URL**: Choose a custom URL (e.g., `investing-scanner`)

4. Click **"Deploy!"**

### Step 3: Wait for Deployment

- Streamlit Cloud will:
  - Clone your repository
  - Install dependencies from `requirements.txt`
  - Start your app
- This takes **2-5 minutes** initially
- You'll see build logs in real-time

### Step 4: Access Your Live App

Once deployed, your app will be available at:
```
https://YOUR_APP_NAME.streamlit.app
```

Example: `https://investing-scanner.streamlit.app`

---

## ⚙️ Configuration & Settings

### Environment Variables (if needed)

If you need to add API keys or secrets:

1. In Streamlit Cloud dashboard, click on your app
2. Click **"Settings"** → **"Secrets"**
3. Add your secrets in TOML format:
   ```toml
   # Example
   api_key = "your-api-key-here"
   ```

### App Settings

In the Streamlit Cloud dashboard:
- **Sleep settings**: Free apps sleep after inactivity (can wake on visit)
- **Resources**: Free tier has 1GB RAM, 1 CPU core
- **Custom domain**: Available with paid plans

---

## 🔄 Updating Your Deployed App

Whenever you make changes:

```bash
# Make your changes locally
# ... edit files ...

# Commit changes
git add .
git commit -m "Your update message"

# Push to GitHub
git push
```

**Streamlit Cloud will automatically redeploy** when it detects changes on GitHub!

---

## 🐛 Troubleshooting

### Build Fails

**Check logs** in Streamlit Cloud dashboard to see what went wrong.

Common issues:
1. **Missing dependencies** → Add to `requirements.txt`
2. **Python version** → Check `runtime.txt` (currently set to Python 3.11)
3. **File paths** → Use relative paths, not absolute

### App is Slow

Free tier has limited resources. To optimize:
- Cache data with `@st.cache_data`
- Reduce universe size for demo
- Use smaller date ranges

### Data Cache Not Working

Streamlit Cloud has ephemeral storage. Consider:
- Using Streamlit's built-in caching
- External storage (AWS S3, Google Cloud Storage)
- Smaller default date ranges

---

## 📊 Alternative Hosting Options

### Render (Free Tier Available)
1. Go to [Render](https://render.com)
2. Create new Web Service
3. Connect GitHub repository
4. Set:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port $PORT`

### Railway (Free $5/month credit)
1. Go to [Railway](https://railway.app)
2. Create new project from GitHub
3. Railway auto-detects Streamlit apps

### Heroku (Paid plans only now)
1. Create `Procfile`:
   ```
   web: sh setup.sh && streamlit run app.py
   ```
2. Deploy via Heroku CLI

---

## 📝 Post-Deployment Checklist

- [ ] GitHub repository is public/accessible
- [ ] README.md has correct URLs
- [ ] App loads without errors
- [ ] Data fetching works
- [ ] Backtest runs successfully
- [ ] Download features work
- [ ] Mobile responsiveness checked

---

## 🎉 Congratulations!

Your Investing Scanner is now live and accessible to anyone with the URL!

### Share Your App:
- **GitHub**: `https://github.com/YOUR_USERNAME/investing-scanner`
- **Live App**: `https://YOUR_APP_NAME.streamlit.app`

### Next Steps:
1. ⭐ Add badges to README
2. 📸 Add screenshots/demo GIF
3. 🔗 Share on social media
4. 📊 Monitor usage in Streamlit Cloud analytics

---

## 📞 Need Help?

- **Streamlit Docs**: https://docs.streamlit.io/
- **Community Forum**: https://discuss.streamlit.io/
- **GitHub Issues**: Create an issue in your repository

---

Made with ❤️ for the Indian Stock Market community
