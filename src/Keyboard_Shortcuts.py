from PyQt5.QtWidgets import QShortcut
from PyQt5.QtGui import QKeySequence

class ShortcutManager:
    def __init__(self, main_window):
        self.main_window = main_window
        self.shortcuts = {}

    def add_shortcut(self, key_sequence, callback):
        shortcut = QShortcut(QKeySequence(key_sequence), self.main_window)
        shortcut.activated.connect(callback)
        self.shortcuts[key_sequence] = shortcut

    def remove_shortcut(self, key_sequence):
        if key_sequence in self.shortcuts:
            self.shortcuts[key_sequence].setEnabled(False)
            del self.shortcuts[key_sequence]

