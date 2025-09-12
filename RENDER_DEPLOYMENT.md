# NeethiAI - Render Deployment Guide

This guide will help you deploy the NeethiAI Flask application to Render with PostgreSQL database and configure all necessary API keys.

## 🚀 Quick Deployment Steps

### 1. Prepare Your Repository
Make sure your code is pushed to a Git repository (GitHub, GitLab, or Bitbucket).

### 2. Create Render Account
1. Go to [render.com](https://render.com)
2. Sign up with your Git provider (GitHub recommended)
3. Connect your repository

### 3. Deploy PostgreSQL Database
1. In Render Dashboard, click **"New +"**
2. Select **"PostgreSQL"**
3. Configure:
   - **Name**: `neethi-postgres`
   - **Database**: `neethi_ai`
   - **User**: `neethi_user`
   - **Region**: Choose closest to your users
   - **Plan**: Starter (Free tier available)
4. Click **"Create Database"**
5. Wait for database to be ready (2-3 minutes)

### 4. Deploy Web Service
1. In Render Dashboard, click **"New +"**
2. Select **"Web Service"**
3. Connect your repository
4. Configure:
   - **Name**: `neethi-ai`
   - **Environment**: `Python 3`
   - **Region**: Same as database
   - **Branch**: `main` (or your default branch)
   - **Root Directory**: Leave empty
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python neethi.py`
   - **Plan**: Starter (Free tier available)

### 5. Configure Environment Variables
In your web service settings, add these environment variables:

#### Required Variables:
```
FLASK_ENV=production
FLASK_DEBUG=false
SECRET_KEY=your-super-secret-key-here
GEMINI_API_KEY=your-gemini-api-key
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
```

#### Database (Auto-configured):
```
DATABASE_URL=postgresql://username:password@host:port/database
```
*This is automatically set by Render when you connect the database.*

### 6. Connect Database to Web Service
1. In your web service settings
2. Go to **"Environment"** tab
3. Click **"Link Database"**
4. Select your `neethi-postgres` database
5. Render will automatically add the `DATABASE_URL` variable

### 7. Deploy
1. Click **"Create Web Service"**
2. Wait for deployment (5-10 minutes)
3. Your app will be available at: `https://neethi-ai.onrender.com`

## 🔑 API Keys Setup

### Google Gemini API
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key and add as `GEMINI_API_KEY`

### Google OAuth (Optional)
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable Google+ API
4. Create OAuth 2.0 credentials
5. Add authorized redirect URI: `https://your-app-name.onrender.com/auth/google/callback`
6. Copy Client ID and Secret

## 📁 Project Structure
```
neethi-ai/
├── neethi.py              # Main Flask application
├── requirements.txt       # Python dependencies
├── render.yaml           # Render configuration
├── env.example           # Environment variables template
├── templates/            # HTML templates
│   ├── base.html
│   ├── chat.html
│   ├── login.html
│   ├── register.html
│   ├── profile.html
│   └── dashboard.html
└── static/               # CSS/JS files (if any)
```

## 🔧 Configuration Files

### render.yaml
```yaml
services:
  - type: web
    name: neethi-ai
    env: python
    plan: starter
    buildCommand: pip install -r requirements.txt
    startCommand: python neethi.py
    envVars:
      - key: FLASK_ENV
        value: production
      - key: FLASK_DEBUG
        value: false
      - key: SECRET_KEY
        generateValue: true
      - key: GEMINI_API_KEY
        sync: false
      - key: GOOGLE_CLIENT_ID
        sync: false
      - key: GOOGLE_CLIENT_SECRET
        sync: false
      - key: DATABASE_URL
        fromDatabase:
          name: neethi-postgres
          property: connectionString
    healthCheckPath: /
    autoDeploy: true

  - type: pserv
    name: neethi-postgres
    env: postgresql
    plan: starter
    region: oregon
    postgresMajorVersion: 15
    autoDeploy: true
```

## 🚨 Troubleshooting

### Common Issues:

1. **Build Fails**
   - Check `requirements.txt` has all dependencies
   - Ensure Python version compatibility

2. **Database Connection Error**
   - Verify `DATABASE_URL` is set correctly
   - Check database is running and accessible

3. **API Key Errors**
   - Verify all API keys are correctly set
   - Check key permissions and quotas

4. **App Crashes on Startup**
   - Check logs in Render dashboard
   - Verify all environment variables are set

### Logs and Monitoring:
- View logs in Render dashboard
- Monitor database usage
- Check API key quotas

## 🔄 Updates and Maintenance

### Updating the App:
1. Push changes to your Git repository
2. Render will automatically redeploy
3. Check logs for any issues

### Database Backups:
- Render provides automatic backups
- Download backups from dashboard if needed

### Scaling:
- Upgrade to paid plans for better performance
- Add more resources as needed

## 📞 Support

- Render Documentation: [render.com/docs](https://render.com/docs)
- Flask Documentation: [flask.palletsprojects.com](https://flask.palletsprojects.com/)
- PostgreSQL Documentation: [postgresql.org/docs](https://www.postgresql.org/docs/)

---

**Note**: The free tier has limitations (sleep after inactivity, limited resources). Consider upgrading to paid plans for production use.
