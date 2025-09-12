# NeethiAI - Legal AI Assistant

<div align="center">
  <img src="static/icons/Logo.png" alt="NeethiAI Logo" width="100" height="100">
  <h3>உங்கள் சட்ட AI உதவியாளர்</h3>
  <p>AI-powered legal assistant for Tamil and English queries with document analysis</p>
  
  [![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)
  [![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
  [![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com)
  [![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-blue.svg)](https://postgresql.org)
</div>

## 🌟 Features

- **🤖 AI-Powered Legal Assistance**: Get instant legal guidance using Google Gemini AI
- **🌐 Bilingual Support**: Full Tamil and English language support
- **📄 Document Analysis**: Upload and analyze PDF/DOCX documents
- **🔍 Fake Notice Detection**: OCR-powered detection of fraudulent legal notices
- **🎤 Voice Chat**: Voice input and text-to-speech in both languages
- **📱 PWA Support**: Install as mobile app with custom logo
- **🔐 User Authentication**: Secure login with Google OAuth integration
- **💾 Chat History**: Persistent conversation storage with PostgreSQL
- **📊 Tax Advisory**: Specialized tax-related legal guidance

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PostgreSQL 12+
- Google Gemini API key

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/neethiai.git
   cd neethiai
   ```

2. **Create virtual environment**
   ```bash
   python -m venv neethi_env
   # Windows
   neethi_env\Scripts\activate
   # Linux/Mac
   source neethi_env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp env.example .env
   # Edit .env with your API keys
   ```

5. **Set up PostgreSQL database**
   ```sql
   CREATE DATABASE neethi_ai;
   CREATE USER neethi_user WITH PASSWORD 'your_password';
   GRANT ALL PRIVILEGES ON DATABASE neethi_ai TO neethi_user;
   ```

6. **Run the application**
   ```bash
   python neethi.py
   ```

7. **Open your browser**
   ```
   http://localhost:5000
   ```

## 🌐 Deployment on Render

### One-Click Deploy
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

### Manual Deployment
1. Fork this repository
2. Create a Render account
3. Create a PostgreSQL database
4. Create a Web Service
5. Configure environment variables
6. Deploy!

See [RENDER_DEPLOYMENT.md](RENDER_DEPLOYMENT.md) for detailed instructions.

## 🔧 Configuration

### Environment Variables
```bash
# Required
GEMINI_API_KEY=your-gemini-api-key
SECRET_KEY=your-secret-key

# Optional
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
DATABASE_URL=postgresql://user:pass@host:port/db
```

### API Keys Setup
1. **Google Gemini**: Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. **Google OAuth**: Get from [Google Cloud Console](https://console.cloud.google.com/)

## 📱 Mobile App Installation

NeethiAI is a Progressive Web App (PWA) that can be installed on mobile devices:

1. Open the app in your mobile browser
2. Look for "Add to Home Screen" option
3. Install with the custom NeethiAI logo
4. Access like a native app

## 🏗️ Project Structure

```
neethiai/
├── neethi.py              # Main Flask application
├── requirements.txt       # Python dependencies
├── render.yaml           # Render deployment config
├── Procfile              # Alternative deployment
├── static/               # Static assets
│   ├── icons/
│   │   └── Logo.png      # App logo
│   ├── css/
│   │   └── style.css     # Custom styles
│   └── manifest.json     # PWA manifest
├── templates/            # HTML templates
│   ├── base.html         # Base template
│   ├── chat.html         # Chat interface
│   ├── login.html        # Login page
│   ├── register.html     # Registration
│   ├── profile.html      # User profile
│   └── index.html        # Landing page
├── .gitignore           # Git ignore rules
├── env.example          # Environment template
└── README.md            # This file
```

## 🛠️ Technologies Used

- **Backend**: Flask, SQLAlchemy, PostgreSQL
- **AI**: Google Gemini API, EasyOCR
- **Frontend**: Bootstrap 5, JavaScript, PWA
- **Authentication**: Flask-Login, Google OAuth
- **Document Processing**: PyPDF2, python-docx, Pillow
- **Language Detection**: langdetect
- **Web Search**: DuckDuckGo Search API

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Website**: [neethiai.com](https://neethiai.com)
- **Issues**: [GitHub Issues](https://github.com/yourusername/neethiai/issues)
- **Documentation**: [RENDER_DEPLOYMENT.md](RENDER_DEPLOYMENT.md)

## 🙏 Acknowledgments

- Google Gemini AI for powerful language processing
- Flask community for the excellent framework
- Bootstrap for responsive UI components
- All contributors and users

---

<div align="center">
  <p>Made with ❤️ for the Indian legal community</p>
  <p><strong>NeethiAI</strong> - Your Legal AI Companion</p>
</div>
