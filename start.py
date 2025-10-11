#!/usr/bin/env python3
"""
Production startup script for NeethiAI
Handles database initialization and server startup
"""

import os
import sys

def check_dependencies():
    """Check if all required dependencies are available"""
    missing_deps = []
    
    try:
        import flask
        print("✓ Flask available")
    except ImportError:
        missing_deps.append("flask")
    
    try:
        import flask_sqlalchemy
        print("✓ Flask-SQLAlchemy available")
    except ImportError:
        missing_deps.append("flask-sqlalchemy")
    
    try:
        import flask_login
        print("✓ Flask-Login available")
    except ImportError:
        missing_deps.append("flask-login")
    
    try:
        import authlib
        print("✓ Authlib available")
    except ImportError:
        print("⚠ Authlib not available - OAuth disabled")
    
    try:
        import waitress
        print("✓ Waitress available")
    except ImportError:
        missing_deps.append("waitress")
    
    if missing_deps:
        print(f"✗ Missing dependencies: {', '.join(missing_deps)}")
        return False
    
    return True

def initialize_database():
    """Initialize database tables"""
    try:
        from neethi import app, db
        with app.app_context():
            db.create_all()
            print("✓ Database tables created successfully")
            return True
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        print("Continuing with limited functionality...")
        return False

def main():
    """Main startup function"""
    print("🚀 Starting NeethiAI Production Server...")
    
    # Check dependencies first
    if not check_dependencies():
        print("❌ Critical dependencies missing. Please check requirements-prod.txt")
        sys.exit(1)
    
    # Initialize database
    db_status = initialize_database()
    
    # Get port from environment
    port = int(os.getenv("PORT", 5000))
    
    if db_status:
        print("✓ Database connected - Full functionality available")
    else:
        print("⚠ Database disconnected - Limited functionality")
    
    print(f"🌐 Server will be available on port {port}")
    print("📱 Mobile-friendly interface ready")
    
    # The app will be served by Waitress via Procfile
    # This script just ensures proper initialization
    return True

if __name__ == "__main__":
    main()
