# -*- coding: utf-8 -*-
"""
Settings Panel - Customize Your Trading Bot!
=============================================
Configure all aspects of your trading bot.

Sections:
- API Settings (Zerodha credentials)
- Trading Settings (risk, position size)
- Notification Settings (alerts)
- Theme Settings (appearance)
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Dict, Any, Callable, Optional
import logging
import os

from .themes import get_theme, THEMES

logger = logging.getLogger(__name__)


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
        for section_key, section_values in self.initial_settings.items():
            if section_key in self.sections:
                self.sections[section_key].set_values(section_values)

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
            # Remove sensitive data
            if 'api' in settings:
                settings['api']['api_secret'] = ''

            with open(filename, 'w') as f:
                json.dump(settings, f, indent=2)

            messagebox.showinfo("Export Complete", f"Settings exported to:\n{filename}")

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
