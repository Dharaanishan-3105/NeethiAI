#!/usr/bin/env python3
"""
Production startup script for NeethiAI
Handles database initialization and server startup
"""

import os
import sys
from neethi import app, db

def initialize_database():
    """Initialize database tables"""
    try:
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
    return app

if __name__ == "__main__":
    main()
