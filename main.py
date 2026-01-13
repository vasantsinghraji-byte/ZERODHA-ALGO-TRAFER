#!/usr/bin/env python3
"""
Zerodha Algo Trader - Main Entry Point
=======================================

This is the main entry point for the Zerodha Algo Trading platform.

Entry Points:
    - GUI Application: Run AlgoTrader_Pro.py for the full desktop interface
    - Dashboard: Run dashboard.py for the web-based monitoring dashboard
    - CLI Trading: Use the src/zerodha_trader module for programmatic access

Usage:
    python main.py          # Shows this help and launches GUI
    python main.py --help   # Shows available options
    python main.py --gui    # Launch GUI application
    python main.py --dash   # Launch web dashboard
"""

import sys
import os

def show_help():
    """Display help information."""
    print(__doc__)
    print("\nAvailable commands:")
    print("  --help, -h     Show this help message")
    print("  --gui, -g      Launch AlgoTrader Pro GUI")
    print("  --dash, -d     Launch web dashboard")
    print("  --version, -v  Show version information")

def launch_gui():
    """Launch the AlgoTrader Pro GUI application."""
    try:
        import AlgoTrader_Pro
        AlgoTrader_Pro.main()
    except ImportError:
        print("Error: AlgoTrader_Pro.py not found.")
        print("Make sure you're running from the project root directory.")
        sys.exit(1)

def launch_dashboard():
    """Launch the web dashboard."""
    try:
        import dashboard
        dashboard.main()
    except ImportError:
        print("Error: dashboard.py not found.")
        print("Make sure you're running from the project root directory.")
        sys.exit(1)

def show_version():
    """Show version information."""
    print("Zerodha Algo Trader")
    print("Version: 1.0.0")
    print("Python:", sys.version)

def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        show_help()
        print("\nLaunching GUI application...")
        launch_gui()
        return

    arg = sys.argv[1].lower()

    if arg in ('--help', '-h'):
        show_help()
    elif arg in ('--gui', '-g'):
        launch_gui()
    elif arg in ('--dash', '-d'):
        launch_dashboard()
    elif arg in ('--version', '-v'):
        show_version()
    else:
        print(f"Unknown option: {arg}")
        show_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
