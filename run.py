#!/usr/bin/env python3
"""
AlgoTrader Pro - Quick Start
============================

Double-click this file or run: python run.py

This will start the trading application!
"""

import sys
import os

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_dependencies():
    """Check if required packages are installed"""
    required = ['tkinter', 'yaml', 'pandas', 'numpy']
    missing = []

    for package in required:
        try:
            if package == 'tkinter':
                import tkinter
            elif package == 'yaml':
                import yaml
            elif package == 'pandas':
                import pandas
            elif package == 'numpy':
                import numpy
        except ImportError:
            missing.append(package)

    if missing:
        print("[X] Missing packages:", ", ".join(missing))
        print("\n[*] Install them with:")
        print("   pip install pyyaml pandas numpy")
        return False

    return True

def main():
    """Start the application"""
    print("""
    ========================================
    |                                      |
    |   ALGOTRADER PRO v2.0                |
    |   Professional Trading Made Simple   |
    |                                      |
    ========================================
    """)

    print("[*] Checking dependencies...")

    if not check_dependencies():
        input("\nPress Enter to exit...")
        return

    print("[OK] All dependencies OK!")
    print("[*] Starting application...\n")

    try:
        from ui.app import AlgoTraderApp
        app = AlgoTraderApp()
        app.run()
    except Exception as e:
        print(f"\n[ERROR] Error starting app: {e}")
        import traceback
        traceback.print_exc()
        print("\n[*] Try running with: python -m ui.app")
        input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()
