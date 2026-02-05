# -*- coding: utf-8 -*-
"""
Settings Panel - Customize Your Trading Bot!
=============================================
Configure all aspects of your trading bot.

Sections:
- API Settings (Zerodha credentials) + Token Generation
- Trading Settings (risk, position size)
- Notification Settings (alerts)
- Theme Settings (appearance)
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Dict, Any, Callable, Optional
import logging
import os
import json
import webbrowser
from pathlib import Path
from datetime import datetime

from .themes import get_theme, THEMES

logger = logging.getLogger(__name__)


class CredentialsManager:
    """
    Manage API credentials and tokens securely.

    SECURITY: Credentials are stored ONLY in the .env file, which should be
    in .gitignore. This prevents accidental credential exposure through:
    - Git commits
    - File sharing
    - Backup systems

    Usage:
        manager = CredentialsManager()
        manager.save_credentials(api_key, api_secret)
        creds = manager.get_credentials()
    """

    # Deprecated files that should not be used
    _DEPRECATED_FILES = ['credentials.json', 'ui_settings.json', 'access_token.json']

    def __init__(self):
        self.config_dir = Path("config")
        self.config_dir.mkdir(exist_ok=True)
        self.env_file = Path(".env")

        # Check for deprecated credential files and warn
        self._check_deprecated_files()

    def _check_deprecated_files(self):
        """Warn about deprecated credential files that should be deleted."""
        for filename in self._DEPRECATED_FILES:
            filepath = self.config_dir / filename
            if filepath.exists():
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    # Check if file contains actual credentials
                    sensitive_keys = ['api_key', 'api_secret', 'access_token']
                    has_creds = any(data.get(k) for k in sensitive_keys)
                    if has_creds:
                        logger.warning(
                            f"\n"
                            f"{'='*60}\n"
                            f"SECURITY WARNING: {filepath} contains credentials!\n"
                            f"{'='*60}\n"
                            f"This file is deprecated and poses a security risk.\n"
                            f"Credentials are now stored only in .env file.\n"
                            f"\n"
                            f"Please DELETE this file: {filepath.absolute()}\n"
                            f"{'='*60}\n"
                        )
                except (json.JSONDecodeError, IOError):
                    pass

    def get_credentials(self) -> dict:
        """
        Load credentials from .env file only.

        Returns:
            dict with keys: api_key, api_secret, user_id
        """
        creds = {"api_key": "", "api_secret": "", "user_id": ""}

        # Load ONLY from .env file (secure storage)
        if self.env_file.exists():
            try:
                with open(self.env_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if '=' in line and not line.startswith('#'):
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip().strip('"').strip("'")
                            if key == 'ZERODHA_API_KEY' and value:
                                creds['api_key'] = value
                            elif key == 'ZERODHA_API_SECRET' and value:
                                creds['api_secret'] = value
                            elif key == 'ZERODHA_USER_ID' and value:
                                creds['user_id'] = value
            except IOError as e:
                logger.error(f"Failed to read .env file: {e}")

        return creds

    def save_credentials(self, api_key: str, api_secret: str, user_id: str = ""):
        """
        Save credentials to .env file only.

        SECURITY: Credentials are NOT saved to JSON files to prevent
        accidental exposure through git commits or file sharing.
        """
        self._update_env_file({
            'ZERODHA_API_KEY': api_key,
            'ZERODHA_API_SECRET': api_secret,
            'ZERODHA_USER_ID': user_id
        })
        logger.info("Credentials saved to .env file")

    def _update_env_file(self, updates: dict):
        """Update .env file with new values, preserving other entries."""
        env_data = {}

        # Read existing .env
        if self.env_file.exists():
            try:
                with open(self.env_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if '=' in line and not line.startswith('#'):
                            key, value = line.split('=', 1)
                            env_data[key.strip()] = value.strip()
            except IOError:
                pass

        # Update with new values
        for key, value in updates.items():
            if value:  # Only update non-empty values
                env_data[key] = value

        # Write back with security header
        with open(self.env_file, 'w') as f:
            f.write("# Zerodha AlgoTrader Configuration\n")
            f.write("# SECURITY: This file contains sensitive credentials.\n")
            f.write("# Make sure this file is in .gitignore!\n")
            f.write(f"# Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for key, value in env_data.items():
                f.write(f"{key}={value}\n")

    def get_access_token(self) -> Optional[str]:
        """
        Load access token from .env file.

        Note: Access tokens are daily and need to be regenerated each day.
        """
        if self.env_file.exists():
            try:
                with open(self.env_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('ZERODHA_ACCESS_TOKEN='):
                            value = line.split('=', 1)[1].strip().strip('"').strip("'")
                            if value:
                                return value
            except IOError:
                pass
        return None

    def save_access_token(self, token: str):
        """
        Save access token to .env file only.

        SECURITY: Token is NOT saved to JSON files.
        """
        self._update_env_file({'ZERODHA_ACCESS_TOKEN': token})
        logger.info("Access token saved to .env file")


class SettingField:
    """
    A single setting field with label and input.

    Supports: text, number, dropdown, checkbox, password
    """

    def __init__(
        self,
        parent: tk.Widget,
        theme: dict,
        label: str,
        field_type: str = 'text',
        default: Any = '',
        options: list = None,
        help_text: str = '',
        on_change: Callable = None
    ):
        self.theme = theme
        self.field_type = field_type
        self.on_change = on_change

        self.frame = tk.Frame(parent, bg=theme['bg_card'])

        # Label row
        label_frame = tk.Frame(self.frame, bg=theme['bg_card'])
        label_frame.pack(fill=tk.X, pady=(0, 5))

        tk.Label(
            label_frame,
            text=label,
            bg=theme['bg_card'],
            fg=theme['text_primary'],
            font=('Segoe UI', 11, 'bold')
        ).pack(side=tk.LEFT)

        if help_text:
            tk.Label(
                label_frame,
                text=f"({help_text})",
                bg=theme['bg_card'],
                fg=theme['text_dim'],
                font=('Segoe UI', 9)
            ).pack(side=tk.LEFT, padx=(5, 0))

        # Input field based on type
        if field_type == 'checkbox':
            self.var = tk.BooleanVar(value=default)
            self.widget = tk.Checkbutton(
                self.frame,
                variable=self.var,
                bg=theme['bg_card'],
                activebackground=theme['bg_card'],
                selectcolor=theme['bg_secondary'],
                command=self._on_value_change
            )
        elif field_type == 'dropdown':
            self.var = tk.StringVar(value=default)
            self.widget = ttk.Combobox(
                self.frame,
                textvariable=self.var,
                values=options or [],
                state='readonly',
                width=30
            )
            self.widget.bind('<<ComboboxSelected>>', lambda e: self._on_value_change())
        elif field_type == 'password':
            self.var = tk.StringVar(value=default)
            self.widget = tk.Entry(
                self.frame,
                textvariable=self.var,
                show='*',
                bg=theme['bg_secondary'],
                fg=theme['text_primary'],
                font=('Segoe UI', 11),
                relief=tk.FLAT,
                width=35
            )
            self.widget.bind('<KeyRelease>', lambda e: self._on_value_change())
        elif field_type == 'number':
            self.var = tk.StringVar(value=str(default))
            self.widget = tk.Entry(
                self.frame,
                textvariable=self.var,
                bg=theme['bg_secondary'],
                fg=theme['text_primary'],
                font=('Segoe UI', 11),
                relief=tk.FLAT,
                width=15
            )
            self.widget.bind('<KeyRelease>', lambda e: self._on_value_change())
        else:  # text
            self.var = tk.StringVar(value=default)
            self.widget = tk.Entry(
                self.frame,
                textvariable=self.var,
                bg=theme['bg_secondary'],
                fg=theme['text_primary'],
                font=('Segoe UI', 11),
                relief=tk.FLAT,
                width=35
            )
            self.widget.bind('<KeyRelease>', lambda e: self._on_value_change())

        self.widget.pack(anchor=tk.W)

    def _on_value_change(self):
        """Handle value change"""
        if self.on_change:
            self.on_change(self.get_value())

    def get_value(self) -> Any:
        """Get current value"""
        if self.field_type == 'checkbox':
            return self.var.get()
        elif self.field_type == 'number':
            try:
                return float(self.var.get())
            except ValueError:
                return 0
        return self.var.get()

    def set_value(self, value: Any):
        """Set field value"""
        self.var.set(value)

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)


class SettingsSection:
    """
    A collapsible section of settings.
    """

    def __init__(
        self,
        parent: tk.Widget,
        theme: dict,
        title: str,
        emoji: str = '⚙️'
    ):
        self.theme = theme
        self.expanded = True
        self.fields: Dict[str, SettingField] = {}

        # Main frame
        self.frame = tk.Frame(parent, bg=theme['bg_card'])
        self.frame.configure(
            highlightbackground=theme['border'],
            highlightthickness=1
        )

        # Header (clickable to collapse/expand)
        header = tk.Frame(self.frame, bg=theme['bg_card'], cursor='hand2')
        header.pack(fill=tk.X)
        header.bind('<Button-1>', self._toggle)

        header_inner = tk.Frame(header, bg=theme['bg_card'])
        header_inner.pack(fill=tk.X, padx=15, pady=10)
        header_inner.bind('<Button-1>', self._toggle)

        tk.Label(
            header_inner,
            text=f"{emoji} {title}",
            bg=theme['bg_card'],
            fg=theme['text_primary'],
            font=('Segoe UI', 14, 'bold')
        ).pack(side=tk.LEFT)

        self.toggle_label = tk.Label(
            header_inner,
            text="▼",
            bg=theme['bg_card'],
            fg=theme['text_secondary'],
            font=('Segoe UI', 10)
        )
        self.toggle_label.pack(side=tk.RIGHT)
        self.toggle_label.bind('<Button-1>', self._toggle)

        # Content frame
        self.content = tk.Frame(self.frame, bg=theme['bg_card'])
        self.content.pack(fill=tk.X, padx=15, pady=(0, 15))

    def _toggle(self, event=None):
        """Toggle section expand/collapse"""
        if self.expanded:
            self.content.pack_forget()
            self.toggle_label.config(text="▶")
        else:
            self.content.pack(fill=tk.X, padx=15, pady=(0, 15))
            self.toggle_label.config(text="▼")
        self.expanded = not self.expanded

    def add_field(
        self,
        key: str,
        label: str,
        field_type: str = 'text',
        default: Any = '',
        options: list = None,
        help_text: str = ''
    ) -> SettingField:
        """Add a setting field to this section"""
        field = SettingField(
            self.content,
            self.theme,
            label,
            field_type,
            default,
            options,
            help_text
        )
        field.pack(fill=tk.X, pady=5)
        self.fields[key] = field
        return field

    def get_values(self) -> Dict[str, Any]:
        """Get all field values"""
        return {key: field.get_value() for key, field in self.fields.items()}

    def set_values(self, values: Dict[str, Any]):
        """Set multiple field values"""
        for key, value in values.items():
            if key in self.fields:
                self.fields[key].set_value(value)

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)


class SettingsPanel:
    """
    Complete settings panel with all trading configurations.
    """

    def __init__(
        self,
        parent: tk.Widget,
        theme_name: str = 'dark',
        on_save: Callable = None,
        initial_settings: Dict = None
    ):
        self.theme = get_theme(theme_name)
        self.parent = parent
        self.on_save = on_save
        self.initial_settings = initial_settings or {}

        self.frame = tk.Frame(parent, bg=self.theme['bg_primary'])
        self.sections: Dict[str, SettingsSection] = {}

        self._create_widgets()
        self._load_settings()

    def _create_widgets(self):
        """Build settings panel"""
        # Title
        title_frame = tk.Frame(self.frame, bg=self.theme['bg_primary'])
        title_frame.pack(fill=tk.X, pady=(0, 15))

        tk.Label(
            title_frame,
            text="⚙️ Settings",
            bg=self.theme['bg_primary'],
            fg=self.theme['text_primary'],
            font=('Segoe UI', 18, 'bold')
        ).pack(side=tk.LEFT)

        # Scrollable content
        canvas = tk.Canvas(
            self.frame,
            bg=self.theme['bg_primary'],
            highlightthickness=0
        )
        scrollbar = ttk.Scrollbar(
            self.frame,
            orient=tk.VERTICAL,
            command=canvas.yview
        )

        scrollable = tk.Frame(canvas, bg=self.theme['bg_primary'])
        scrollable.bind(
            '<Configure>',
            lambda e: canvas.configure(scrollregion=canvas.bbox('all'))
        )

        canvas.create_window((0, 0), window=scrollable, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)

        # Mouse wheel scrolling
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", on_mousewheel)

        # === API SETTINGS ===
        api_section = SettingsSection(scrollable, self.theme, "API Configuration", "🔐")
        api_section.add_field('api_key', 'API Key', 'text', '', help_text='Your Zerodha API key')
        api_section.add_field('api_secret', 'API Secret', 'password', '', help_text='Keep this secret!')
        api_section.add_field('user_id', 'User ID', 'text', '', help_text='Your Zerodha user ID')

        # API Action Buttons
        api_btn_frame = tk.Frame(api_section.content, bg=self.theme['bg_card'])
        api_btn_frame.pack(fill=tk.X, pady=(10, 5))

        tk.Button(
            api_btn_frame,
            text="💾 Save Credentials",
            bg=self.theme['btn_success'],
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            relief=tk.FLAT,
            cursor='hand2',
            command=self._save_api_credentials
        ).pack(side=tk.LEFT, padx=(0, 10), ipadx=10, ipady=5)

        tk.Button(
            api_btn_frame,
            text="🔑 Generate Token",
            bg=self.theme['btn_primary'],
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            relief=tk.FLAT,
            cursor='hand2',
            command=self._generate_token
        ).pack(side=tk.LEFT, padx=(0, 10), ipadx=10, ipady=5)

        tk.Button(
            api_btn_frame,
            text="📋 Get API Key",
            bg=self.theme['bg_secondary'],
            fg=self.theme['text_primary'],
            font=('Segoe UI', 10),
            relief=tk.FLAT,
            cursor='hand2',
            command=lambda: webbrowser.open("https://developers.kite.trade/")
        ).pack(side=tk.LEFT, ipadx=10, ipady=5)

        # Token status label
        self.token_status_label = tk.Label(
            api_section.content,
            text="Token: Not generated today",
            bg=self.theme['bg_card'],
            fg=self.theme['text_dim'],
            font=('Segoe UI', 9)
        )
        self.token_status_label.pack(anchor=tk.W, pady=(5, 0))

        # Check existing token
        self._credentials_manager = CredentialsManager()
        self._check_token_status()

        api_section.pack(fill=tk.X, pady=(0, 10))
        self.sections['api'] = api_section

        # === TRADING SETTINGS ===
        trading_section = SettingsSection(scrollable, self.theme, "Trading Configuration", "📊")
        trading_section.add_field(
            'paper_trading', 'Paper Trading Mode', 'checkbox', True,
            help_text='Practice without real money'
        )
        trading_section.add_field(
            'initial_capital', 'Initial Capital (Rs.)', 'number', 100000,
            help_text='Starting balance'
        )
        trading_section.add_field(
            'max_positions', 'Max Open Positions', 'number', 5,
            help_text='Maximum trades at once'
        )
        trading_section.add_field(
            'default_exchange', 'Default Exchange', 'dropdown', 'NSE',
            options=['NSE', 'BSE', 'NFO'], help_text='Primary exchange'
        )
        trading_section.pack(fill=tk.X, pady=(0, 10))
        self.sections['trading'] = trading_section

        # === RISK SETTINGS ===
        risk_section = SettingsSection(scrollable, self.theme, "Risk Management", "🛡️")
        risk_section.add_field(
            'risk_per_trade', 'Risk Per Trade (%)', 'number', 2.0,
            help_text='Max loss per trade'
        )
        risk_section.add_field(
            'max_daily_loss', 'Max Daily Loss (%)', 'number', 5.0,
            help_text='Stop trading after this loss'
        )
        risk_section.add_field(
            'default_stop_loss', 'Default Stop Loss (%)', 'number', 2.0,
            help_text='Automatic stop loss'
        )
        risk_section.add_field(
            'default_target', 'Default Target (%)', 'number', 4.0,
            help_text='Automatic profit target'
        )
        risk_section.add_field(
            'use_trailing_stop', 'Use Trailing Stop Loss', 'checkbox', False,
            help_text='Auto-adjust stop loss'
        )
        risk_section.pack(fill=tk.X, pady=(0, 10))
        self.sections['risk'] = risk_section

        # === INFRASTRUCTURE SETTINGS ===
        infra_section = SettingsSection(scrollable, self.theme, "Infrastructure", "🏗️")
        infra_section.add_field(
            'latency_warning_ms', 'Latency Warning (ms)', 'number', 200,
            help_text='Alert when latency exceeds this'
        )
        infra_section.add_field(
            'latency_critical_ms', 'Latency Critical (ms)', 'number', 500,
            help_text='Critical alert threshold'
        )
        infra_section.add_field(
            'tick_filter_enabled', 'Enable Tick Filter', 'checkbox', True,
            help_text='Filter bad/spike ticks'
        )
        infra_section.add_field(
            'tick_spike_threshold', 'Tick Spike Threshold (%)', 'number', 5.0,
            help_text='Max price change per tick'
        )
        infra_section.add_field(
            'rate_limit_buffer', 'Rate Limit Buffer (%)', 'number', 20,
            help_text='Reserve % of API quota'
        )
        infra_section.add_field(
            'kill_switch_keyboard', 'Kill Switch Hotkey', 'checkbox', True,
            help_text='Enable Ctrl+Shift+K hotkey'
        )
        infra_section.add_field(
            'max_drawdown_trigger', 'Auto Kill Switch Drawdown (%)', 'number', 10.0,
            help_text='Trigger kill switch on drawdown'
        )
        infra_section.pack(fill=tk.X, pady=(0, 10))
        self.sections['infrastructure'] = infra_section

        # === NOTIFICATION SETTINGS ===
        notif_section = SettingsSection(scrollable, self.theme, "Notifications", "🔔")
        notif_section.add_field(
            'telegram_enabled', 'Enable Telegram Alerts', 'checkbox', False,
            help_text='Get alerts on Telegram'
        )
        notif_section.add_field(
            'telegram_token', 'Telegram Bot Token', 'password', '',
            help_text='From @BotFather'
        )
        notif_section.add_field(
            'telegram_chat_id', 'Telegram Chat ID', 'text', '',
            help_text='Your chat ID'
        )
        notif_section.add_field(
            'alert_on_trade', 'Alert on Trade', 'checkbox', True,
            help_text='Notify on buy/sell'
        )
        notif_section.add_field(
            'alert_on_stop_loss', 'Alert on Stop Loss', 'checkbox', True,
            help_text='Notify on stop loss hit'
        )
        notif_section.pack(fill=tk.X, pady=(0, 10))
        self.sections['notifications'] = notif_section

        # === APPEARANCE SETTINGS ===
        appearance_section = SettingsSection(scrollable, self.theme, "Appearance", "🎨")
        appearance_section.add_field(
            'theme', 'Color Theme', 'dropdown', 'dark',
            options=list(THEMES.keys()), help_text='App color scheme'
        )
        appearance_section.add_field(
            'show_emojis', 'Show Emojis', 'checkbox', True,
            help_text='Display emoji icons'
        )
        appearance_section.add_field(
            'compact_mode', 'Compact Mode', 'checkbox', False,
            help_text='Smaller UI elements'
        )
        appearance_section.pack(fill=tk.X, pady=(0, 10))
        self.sections['appearance'] = appearance_section

        # === ADVANCED SETTINGS ===
        advanced_section = SettingsSection(scrollable, self.theme, "Advanced", "🔧")
        advanced_section.add_field(
            'log_level', 'Log Level', 'dropdown', 'INFO',
            options=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
            help_text='Logging verbosity'
        )
        advanced_section.add_field(
            'auto_start_bot', 'Auto-Start Bot on Login', 'checkbox', False,
            help_text='Start bot automatically'
        )
        advanced_section.add_field(
            'save_trades_to_csv', 'Save Trades to CSV', 'checkbox', True,
            help_text='Export trade history'
        )
        advanced_section.pack(fill=tk.X, pady=(0, 10))
        self.sections['advanced'] = advanced_section

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Bottom buttons
        button_frame = tk.Frame(self.frame, bg=self.theme['bg_primary'])
        button_frame.pack(fill=tk.X, pady=(15, 0))

        tk.Button(
            button_frame,
            text="💾 Save Settings",
            bg=self.theme['btn_success'],
            fg='white',
            font=('Segoe UI', 12, 'bold'),
            relief=tk.FLAT,
            cursor='hand2',
            command=self._save_settings
        ).pack(side=tk.RIGHT, ipadx=15, ipady=8)

        tk.Button(
            button_frame,
            text="↩️ Reset to Defaults",
            bg=self.theme['bg_secondary'],
            fg=self.theme['text_primary'],
            font=('Segoe UI', 11),
            relief=tk.FLAT,
            cursor='hand2',
            command=self._reset_settings
        ).pack(side=tk.RIGHT, padx=(0, 10), ipadx=10, ipady=8)

        tk.Button(
            button_frame,
            text="📤 Export",
            bg=self.theme['bg_secondary'],
            fg=self.theme['text_primary'],
            font=('Segoe UI', 11),
            relief=tk.FLAT,
            cursor='hand2',
            command=self._export_settings
        ).pack(side=tk.LEFT, ipadx=10, ipady=8)

        tk.Button(
            button_frame,
            text="📥 Import",
            bg=self.theme['bg_secondary'],
            fg=self.theme['text_primary'],
            font=('Segoe UI', 11),
            relief=tk.FLAT,
            cursor='hand2',
            command=self._import_settings
        ).pack(side=tk.LEFT, padx=(10, 0), ipadx=10, ipady=8)

    def _load_settings(self):
        """Load initial settings"""
        # Load from initial_settings
        for section_key, section_values in self.initial_settings.items():
            if section_key in self.sections:
                self.sections[section_key].set_values(section_values)

        # Load API credentials from saved files
        try:
            creds = self._credentials_manager.get_credentials()
            if 'api' in self.sections:
                self.sections['api'].set_values(creds)
        except Exception as e:
            logger.warning(f"Could not load saved credentials: {e}")

    def _save_api_credentials(self):
        """Save API credentials to .env file."""
        try:
            api_settings = self.sections['api'].get_values()
            api_key = api_settings.get('api_key', '').strip()
            api_secret = api_settings.get('api_secret', '').strip()
            user_id = api_settings.get('user_id', '').strip()

            if not api_key or not api_secret:
                messagebox.showerror("Error", "Please enter both API Key and API Secret")
                return

            self._credentials_manager.save_credentials(api_key, api_secret, user_id)

            messagebox.showinfo(
                "Credentials Saved",
                "Your API credentials have been saved to:\n\n"
                "- .env file\n"
                "- config/credentials.json\n\n"
                "You can now generate your daily token."
            )

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save credentials:\n{str(e)}")

    def _generate_token(self):
        """Open browser for Zerodha login and handle token generation."""
        try:
            api_settings = self.sections['api'].get_values()
            api_key = api_settings.get('api_key', '').strip()
            api_secret = api_settings.get('api_secret', '').strip()

            if not api_key:
                messagebox.showerror(
                    "API Key Required",
                    "Please enter your API Key first!\n\n"
                    "If you don't have one, click 'Get API Key' button."
                )
                return

            # Save credentials first
            self._credentials_manager.save_credentials(
                api_key,
                api_secret,
                api_settings.get('user_id', '')
            )

            # Open Zerodha login
            login_url = f"https://kite.zerodha.com/connect/login?v=3&api_key={api_key}"
            webbrowser.open(login_url)

            # Show token entry dialog
            self._show_token_dialog(api_key, api_secret)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start token generation:\n{str(e)}")

    def _show_token_dialog(self, api_key: str, api_secret: str):
        """Show dialog to enter request token after Zerodha login."""
        dialog = tk.Toplevel(self.parent)
        dialog.title("Complete Login - Enter Request Token")
        dialog.geometry("550x480")
        dialog.configure(bg=self.theme['bg_primary'])
        dialog.transient(self.parent)
        dialog.grab_set()
        dialog.resizable(False, False)

        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() - 550) // 2
        y = (dialog.winfo_screenheight() - 480) // 2
        dialog.geometry(f"+{x}+{y}")

        # Main content frame (to ensure buttons at bottom)
        content_frame = tk.Frame(dialog, bg=self.theme['bg_primary'])
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)

        # Title
        tk.Label(
            content_frame,
            text="🔑 Complete Token Generation",
            bg=self.theme['bg_primary'],
            fg=self.theme['text_primary'],
            font=('Segoe UI', 14, 'bold')
        ).pack(pady=(0, 10))

        # Instructions
        instructions = """1. Login to Zerodha in the browser window that opened

2. After login, you'll be redirected to a URL like:
   http://127.0.0.1/?request_token=XXXXXX&status=success

3. Copy the 'request_token' value from the URL
   (the part after 'request_token=' and before '&')

4. Paste it below and click 'Generate Token'"""

        tk.Label(
            content_frame,
            text=instructions,
            bg=self.theme['bg_primary'],
            fg=self.theme['text_secondary'],
            font=('Segoe UI', 10),
            justify=tk.LEFT,
            anchor=tk.W
        ).pack(fill=tk.X, pady=(0, 15))

        # Token entry
        entry_frame = tk.Frame(content_frame, bg=self.theme['bg_primary'])
        entry_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            entry_frame,
            text="Request Token:",
            bg=self.theme['bg_primary'],
            fg=self.theme['text_primary'],
            font=('Segoe UI', 11, 'bold')
        ).pack(anchor=tk.W)

        token_entry = tk.Entry(
            entry_frame,
            font=('Segoe UI', 12),
            width=45,
            bg=self.theme['bg_secondary'],
            fg=self.theme['text_primary'],
            relief=tk.FLAT
        )
        token_entry.pack(fill=tk.X, pady=5, ipady=8)
        token_entry.focus_set()  # Auto-focus for easy pasting

        # Status label
        status_label = tk.Label(
            content_frame,
            text="Paste the request_token from the redirect URL above",
            bg=self.theme['bg_primary'],
            fg=self.theme['text_dim'],
            font=('Segoe UI', 9)
        )
        status_label.pack(pady=(5, 0))

        def submit_token():
            request_token = token_entry.get().strip()
            if not request_token:
                status_label.config(text="Please enter the request token!", fg='red')
                return

            status_label.config(text="Generating access token...", fg=self.theme['text_dim'])
            dialog.update()

            try:
                from kiteconnect import KiteConnect

                kite = KiteConnect(api_key=api_key)
                data = kite.generate_session(request_token, api_secret=api_secret)
                access_token = data['access_token']

                # Save token
                self._credentials_manager.save_access_token(access_token)

                # Update status
                self._check_token_status()

                dialog.destroy()
                messagebox.showinfo(
                    "Success!",
                    "Access token generated and saved!\n\n"
                    "You're now ready to trade.\n"
                    "Token is valid until 6 AM tomorrow."
                )

            except ImportError:
                status_label.config(text="Error: kiteconnect not installed", fg='red')
                messagebox.showerror(
                    "Missing Package",
                    "Please install kiteconnect:\n\npip install kiteconnect"
                )
            except Exception as e:
                status_label.config(text=f"Error: {str(e)}", fg='red')
                logger.error(f"Token generation failed: {e}")

        # Button frame at bottom with clear separation
        btn_frame = tk.Frame(dialog, bg=self.theme['bg_card'])
        btn_frame.pack(fill=tk.X, side=tk.BOTTOM)

        # Separator line
        separator = tk.Frame(btn_frame, bg=self.theme['border'], height=1)
        separator.pack(fill=tk.X)

        btn_inner = tk.Frame(btn_frame, bg=self.theme['bg_card'])
        btn_inner.pack(fill=tk.X, padx=20, pady=15)

        tk.Button(
            btn_inner,
            text="Cancel",
            bg=self.theme['bg_secondary'],
            fg=self.theme['text_primary'],
            font=('Segoe UI', 11),
            relief=tk.FLAT,
            cursor='hand2',
            width=12,
            command=dialog.destroy
        ).pack(side=tk.LEFT, ipady=8)

        generate_btn = tk.Button(
            btn_inner,
            text="✓ GENERATE TOKEN",
            bg='#28a745',
            fg='white',
            font=('Segoe UI', 12, 'bold'),
            relief=tk.FLAT,
            cursor='hand2',
            width=20,
            command=submit_token
        )
        generate_btn.pack(side=tk.RIGHT, ipady=8)

        # Bind Enter key to submit
        token_entry.bind('<Return>', lambda e: submit_token())

    def _check_token_status(self):
        """Check and display current token status."""
        try:
            token = self._credentials_manager.get_access_token()
            if token:
                self.token_status_label.config(
                    text=f"Token: Valid for today",
                    fg='green'
                )
            else:
                self.token_status_label.config(
                    text="Token: Not generated today - Click 'Generate Token'",
                    fg='orange'
                )
        except Exception:
            pass

    def _save_settings(self):
        """Save all settings"""
        settings = self.get_all_settings()

        if self.on_save:
            self.on_save(settings)

        messagebox.showinfo(
            "Settings Saved",
            "Your settings have been saved successfully!\n\n"
            "Some changes may require a restart."
        )

    def _reset_settings(self):
        """Reset to default settings"""
        if messagebox.askyesno("Reset Settings", "Reset all settings to defaults?"):
            defaults = {
                'api': {'api_key': '', 'api_secret': '', 'user_id': ''},
                'trading': {
                    'paper_trading': True,
                    'initial_capital': 100000,
                    'max_positions': 5,
                    'default_exchange': 'NSE'
                },
                'risk': {
                    'risk_per_trade': 2.0,
                    'max_daily_loss': 5.0,
                    'default_stop_loss': 2.0,
                    'default_target': 4.0,
                    'use_trailing_stop': False
                },
                'infrastructure': {
                    'latency_warning_ms': 200,
                    'latency_critical_ms': 500,
                    'tick_filter_enabled': True,
                    'tick_spike_threshold': 5.0,
                    'rate_limit_buffer': 20,
                    'kill_switch_keyboard': True,
                    'max_drawdown_trigger': 10.0
                },
                'notifications': {
                    'telegram_enabled': False,
                    'telegram_token': '',
                    'telegram_chat_id': '',
                    'alert_on_trade': True,
                    'alert_on_stop_loss': True
                },
                'appearance': {
                    'theme': 'dark',
                    'show_emojis': True,
                    'compact_mode': False
                },
                'advanced': {
                    'log_level': 'INFO',
                    'auto_start_bot': False,
                    'save_trades_to_csv': True
                }
            }

            for section_key, section_values in defaults.items():
                if section_key in self.sections:
                    self.sections[section_key].set_values(section_values)

            messagebox.showinfo("Reset Complete", "Settings reset to defaults!")

    def _export_settings(self):
        """Export settings to file"""
        import json

        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            title="Export Settings"
        )

        if filename:
            settings = self.get_all_settings()
            # SECURITY: Remove ALL sensitive data before export
            if 'api' in settings:
                settings['api'] = {
                    'api_key': '',
                    'api_secret': '',
                    'user_id': settings.get('api', {}).get('user_id', '')
                }
            # Also remove notification tokens
            if 'notifications' in settings:
                settings['notifications']['telegram_token'] = ''

            with open(filename, 'w') as f:
                json.dump(settings, f, indent=2)

            messagebox.showinfo(
                "Export Complete",
                f"Settings exported to:\n{filename}\n\n"
                "Note: Credentials were excluded for security."
            )

    def _import_settings(self):
        """Import settings from file"""
        import json

        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json")],
            title="Import Settings"
        )

        if filename:
            try:
                with open(filename, 'r') as f:
                    settings = json.load(f)

                for section_key, section_values in settings.items():
                    if section_key in self.sections:
                        self.sections[section_key].set_values(section_values)

                messagebox.showinfo("Import Complete", "Settings imported successfully!")

            except Exception as e:
                messagebox.showerror("Import Failed", f"Could not import settings:\n{str(e)}")

    def get_all_settings(self) -> Dict[str, Dict[str, Any]]:
        """Get all settings from all sections"""
        return {
            key: section.get_values()
            for key, section in self.sections.items()
        }

    def get_section_settings(self, section_key: str) -> Dict[str, Any]:
        """Get settings for a specific section"""
        if section_key in self.sections:
            return self.sections[section_key].get_values()
        return {}

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)


# ============== SETTINGS DIALOG ==============

class SettingsDialog:
    """
    Modal dialog for settings.
    """

    def __init__(
        self,
        parent: tk.Widget,
        theme_name: str = 'dark',
        on_save: Callable = None,
        initial_settings: Dict = None
    ):
        self.theme = get_theme(theme_name)
        self.result = None

        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("⚙️ Settings")
        self.dialog.geometry("700x600")
        self.dialog.configure(bg=self.theme['bg_primary'])
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # Center dialog
        self.dialog.update_idletasks()
        x = (parent.winfo_width() - 700) // 2 + parent.winfo_x()
        y = (parent.winfo_height() - 600) // 2 + parent.winfo_y()
        self.dialog.geometry(f"+{x}+{y}")

        # Create settings panel
        def save_and_close(settings):
            self.result = settings
            if on_save:
                on_save(settings)
            self.dialog.destroy()

        self.panel = SettingsPanel(
            self.dialog,
            theme_name,
            on_save=save_and_close,
            initial_settings=initial_settings
        )
        self.panel.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

    def show(self) -> Optional[Dict]:
        """Show dialog and return result"""
        self.dialog.wait_window()
        return self.result


# ============== TEST ==============

if __name__ == "__main__":
    print("=" * 50)
    print("SETTINGS PANEL - Test")
    print("=" * 50)

    def on_save(settings):
        print("Settings saved:")
        for section, values in settings.items():
            print(f"\n[{section}]")
            for key, value in values.items():
                # Hide sensitive data
                if 'secret' in key.lower() or 'token' in key.lower():
                    value = '***' if value else ''
                print(f"  {key}: {value}")

    # Create test window
    root = tk.Tk()
    root.title("Settings Panel Test")
    root.geometry("800x700")
    root.configure(bg='#1a1a2e')

    panel = SettingsPanel(root, 'dark', on_save=on_save)
    panel.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

    root.mainloop()

    print("\n" + "=" * 50)
    print("Settings Panel ready!")
    print("=" * 50)
