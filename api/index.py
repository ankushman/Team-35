"""
api/index.py - Vercel Serverless Entry Point

Exposes the Flask app from dashboard.py as a Vercel serverless function.
Forces simulation mode since real psutil metrics are not meaningful
on a serverless platform.
"""

import os
import sys

# Add the project root to sys.path so imports like
# 'from monitor import get_metrics' resolve correctly.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Force simulation mode for the serverless deployment
import dashboard
dashboard.SIMULATE_SPIKE = True

# Vercel looks for an `app` variable that is a WSGI/ASGI application
app = dashboard.app
