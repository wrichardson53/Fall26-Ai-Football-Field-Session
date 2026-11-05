from PySide6.QtGui import QPalette, QColor
from PySide6.QtCore import Qt


def _build_palette(colors, disabled_text):
    palette = QPalette()
    for role, color in colors.items():
        palette.setColor(role, QColor(color))
    # Greyed-out text for disabled widgets so they stay readable but distinct
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        palette.setColor(QPalette.Disabled, role, QColor(disabled_text))
    return palette


def get_light_palette():
    return _build_palette({
        QPalette.Window: "#f5f5f5",
        QPalette.WindowText: "#000000",
        QPalette.Base: "#ffffff",
        QPalette.AlternateBase: "#eeeeee",
        QPalette.ToolTipBase: "#ffffdc",
        QPalette.ToolTipText: "#000000",
        QPalette.PlaceholderText: "#808080",
        QPalette.Text: "#000000",
        QPalette.Button: "#e0e0e0",
        QPalette.ButtonText: "#000000",
        QPalette.BrightText: "#ff0000",
        QPalette.Light: "#ffffff",
        QPalette.Midlight: "#e8e8e8",
        QPalette.Mid: "#b0b0b0",
        QPalette.Dark: "#a0a0a0",
        QPalette.Shadow: "#707070",
        QPalette.Highlight: "#0078d4",
        QPalette.HighlightedText: "#ffffff",
        QPalette.Link: "#0062b1",
    }, disabled_text="#a0a0a0")


def get_dark_palette():
    return _build_palette({
        QPalette.Window: "#2f2f2f",
        QPalette.WindowText: "#ffffff",
        QPalette.Base: "#3c3c3c",
        QPalette.AlternateBase: "#444444",
        QPalette.ToolTipBase: "#3c3c3c",
        QPalette.ToolTipText: "#ffffff",
        QPalette.PlaceholderText: "#9a9a9a",
        QPalette.Text: "#ffffff",
        QPalette.Button: "#3c3c3c",
        QPalette.ButtonText: "#ffffff",
        QPalette.BrightText: "#ff5555",
        QPalette.Light: "#505050",
        QPalette.Midlight: "#444444",
        QPalette.Mid: "#2a2a2a",
        QPalette.Dark: "#202020",
        QPalette.Shadow: "#000000",
        QPalette.Highlight: "#0078d4",
        QPalette.HighlightedText: "#ffffff",
        QPalette.Link: "#58a6ff",
    }, disabled_text="#808080")
