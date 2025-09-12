# NeethiAI Railway Deployment Guide

## 🚀 Deploy to Railway

### Prerequisites
1. GitHub account with your NeethiAI repository
2. Railway account (free at railway.app)
3. Google Gemini API key

### Step 1: Prepare Repository
1. Push all changes to your GitHub repository
2. Ensure all files are committed:
   - `neethi.py` (updated for Railway)
   - `requirements.txt`
   - `railway.toml`
   - `Procfile`
   - `static/manifest.json`
   - `static/sw.js`
   - `templates/` (all files)

### Step 2: Deploy to Railway
1. Go to [railway.app](https://railway.app) and sign up/login
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your NeethiAI repository
4. Railway will automatically detect it's a Python app

### Step 3: Add PostgreSQL Database
1. In your Railway project dashboard
2. Click "New" → "Database" → "PostgreSQL"
3. Railway will automatically provision a PostgreSQL database
4. The `DATABASE_URL` environment variable will be set automatically

### Step 4: Configure Environment Variables
In Railway project settings, add these environment variables:

```
SECRET_KEY=your-super-secret-key-here
GEMINI_API_KEY=your-gemini-api-key-here
GOOGLE_CLIENT_ID=your-google-oauth-client-id (optional)
GOOGLE_CLIENT_SECRET=your-google-oauth-secret (optional)
```

### Step 5: Deploy
1. Railway will automatically deploy when you push to GitHub
2. Your app will be available at: `https://your-app-name.railway.app`
3. The database will be automatically connected

## 📱 Mobile Web App Installation

### For Users:
1. Open the app in Chrome/Edge on mobile
2. Tap the browser menu (3 dots)
3. Select "Add to Home Screen" or "Install App"
4. The app will install with the NeethiAI logo
5. Launch from home screen like a native app

### Features:
- ✅ Works offline (cached)
- ✅ Full-screen experience
- ✅ Custom app icon
- ✅ Splash screen
- ✅ Native app feel

## 🔧 Railway Features Used

- **Free Tier**: Generous monthly credits
- **PostgreSQL**: Managed database with automatic connection
- **Auto-deploy**: Git push triggers deployment
- **Environment Variables**: Secure configuration
- **Custom Domain**: Available on paid plans
- **SSL**: Automatic HTTPS

## 📊 Monitoring

Railway provides:
- Real-time logs
- Performance metrics
- Database monitoring
- Automatic restarts on failure

## 🆘 Troubleshooting

### Common Issues:
1. **Database Connection**: Ensure `DATABASE_URL` is set
2. **API Keys**: Verify all environment variables are set
3. **Build Failures**: Check `requirements.txt` is complete
4. **Port Issues**: Railway sets `PORT` automatically

### Support:
- Railway Documentation: https://docs.railway.app
- Railway Discord: https://discord.gg/railway
