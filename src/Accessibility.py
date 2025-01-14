from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QObject, pyqtSignal
import pyttsx3

class AccessibilityManager(QObject):
    speak_signal = pyqtSignal(str)

    def __init__(self, app):
        super().__init__()
        self.app = app
        self.tts_engine = pyttsx3.init()
        self.speak_signal.connect(self.speak)
        self.setup_screen_reader()

    def setup_screen_reader(self):
        self.app.installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() == Qt.KeyPress and event.modifiers() & Qt.ControlModifier:
            if event.key() == Qt.Key_R:  # Ctrl+R to read focused widget
                focused_widget = QApplication.focusWidget()
                if focused_widget:
                    self.speak_signal.emit(focused_widget.accessibleName() or focused_widget.toolTip())
        return False

    def speak(self, text):
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def set_accessible_name(self, widget, name):
        widget.setAccessibleName(name)

    def set_tab_order(self, *widgets):
        for i in range(len(widgets) - 1):
            QWidget.setTabOrder(widgets[i], widgets[i + 1])

    def create_keyboard_shortcut(self, key_sequence, callback):
        shortcut = QShortcut(QKeySequence(key_sequence), self.app.activeWindow())
        shortcut.activated.connect(callback)


