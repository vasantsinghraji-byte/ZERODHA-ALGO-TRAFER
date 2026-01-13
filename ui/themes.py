"""
Color Themes
Beautiful colors that make the app easy to use!
"""

THEMES = {
    'dark': {
        # Background colors
        'bg_primary': '#1a1a2e',      # Deep blue-black
        'bg_secondary': '#16213e',    # Navy blue
        'bg_card': '#0f3460',         # Card background
        'bg_hover': '#1f4068',        # Hover state

        # Text colors
        'text_primary': '#ffffff',    # White
        'text_secondary': '#a0a0a0',  # Gray
        'text_dim': '#606060',        # Dim gray

        # Accent colors
        'accent': '#e94560',          # Red-pink accent
        'success': '#00ff88',         # Bright green
        'danger': '#ff4444',          # Red
        'warning': '#ffaa00',         # Orange
        'info': '#00d4ff',            # Cyan

        # Button colors
        'btn_primary': '#e94560',
        'btn_success': '#00c853',
        'btn_danger': '#ff1744',

        # Border
        'border': '#2a2a4a',
    },

    'light': {
        # Background colors
        'bg_primary': '#f5f5f5',
        'bg_secondary': '#ffffff',
        'bg_card': '#ffffff',
        'bg_hover': '#e8e8e8',

        # Text colors
        'text_primary': '#1a1a1a',
        'text_secondary': '#666666',
        'text_dim': '#999999',

        # Accent colors
        'accent': '#6200ea',
        'success': '#00c853',
        'danger': '#ff1744',
        'warning': '#ff9100',
        'info': '#2979ff',

        # Button colors
        'btn_primary': '#6200ea',
        'btn_success': '#00c853',
        'btn_danger': '#ff1744',

        # Border
        'border': '#e0e0e0',
    },

    'neon': {
        # Background colors
        'bg_primary': '#0d0d0d',
        'bg_secondary': '#1a1a1a',
        'bg_card': '#262626',
        'bg_hover': '#333333',

        # Text colors
        'text_primary': '#ffffff',
        'text_secondary': '#b0b0b0',
        'text_dim': '#707070',

        # Accent colors (NEON!)
        'accent': '#ff00ff',          # Magenta
        'success': '#00ff00',         # Lime green
        'danger': '#ff0000',          # Red
        'warning': '#ffff00',         # Yellow
        'info': '#00ffff',            # Cyan

        # Button colors
        'btn_primary': '#ff00ff',
        'btn_success': '#00ff00',
        'btn_danger': '#ff0000',

        # Border
        'border': '#404040',
    }
}


def get_theme(theme_name: str = 'dark') -> dict:
    """Get theme colors by name"""
    return THEMES.get(theme_name, THEMES['dark'])
