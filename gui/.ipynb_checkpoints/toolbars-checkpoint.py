from PyQt5.QtWidgets import QToolBar, QAction, QComboBox, QSpinBox, QDoubleSpinBox, QLabel
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSignal

class MainToolBar(QToolBar):
    tool_changed = pyqtSignal(str)
    brush_size_changed = pyqtSignal(int)
    opacity_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        self.setMovable(False)
        self.setIconSize(QSize(32, 32))

        # Add tools
        self.add_tool_action("Select", "icons/select.png", "Select tool")
        self.add_tool_action("Pan", "icons/pan.png", "Pan tool")
        self.add_tool_action("Zoom", "icons/zoom.png", "Zoom tool")
        self.add_tool_action("Brush", "icons/brush.png", "Brush tool")
        self.add_tool_action("Eraser", "icons/eraser.png", "Eraser tool")

        self.addSeparator()

        # Brush size
        self.brush_size_spin = QSpinBox()
        self.brush_size_spin.setRange(1, 100)
        self.brush_size_spin.setValue(10)
        self.brush_size_spin.valueChanged.connect(self.brush_size_changed.emit)
        self.addWidget(QLabel("Brush Size:"))
        self.addWidget(self.brush_size_spin)

        # Opacity
        self.opacity_spin = QDoubleSpinBox()
        self.opacity_spin.setRange(0.0, 1.0)
        self.opacity_spin.setSingleStep(0.1)
        self.opacity_spin.setValue(1.0)
        self.opacity_spin.valueChanged.connect(self.opacity_changed.emit)
        self.addWidget(QLabel("Opacity:"))
        self.addWidget(self.opacity_spin)

    def add_tool_action(self, name, icon_path, tooltip):
        action = QAction(QIcon(icon_path), name, self)
        action.setCheckable(True)
        action.setToolTip(tooltip)
        action.triggered.connect(lambda: self.tool_changed.emit(name))
        self.addAction(action)

class FilterToolBar(QToolBar):
    filter_applied = pyqtSignal(str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        self.setMovable(False)

        # Filter selection
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["None", "Gaussian Blur", "Sharpen", "Edge Detection"])
        self.addWidget(QLabel("Filter:"))
        self.addWidget(self.filter_combo)

        # Filter strength
        self.strength_spin = QSpinBox()
        self.strength_spin.setRange(1, 100)
        self.strength_spin.setValue(50)
        self.addWidget(QLabel("Strength:"))
        self.addWidget(self.strength_spin)

        # Apply button
        apply_action = QAction("Apply Filter", self)
        apply_action.triggered.connect(self.apply_filter)
        self.addAction(apply_action)

    def apply_filter(self):
        filter_name = self.filter_combo.currentText()
        strength = self.strength_spin.value()
        self.filter_applied.emit(filter_name, {"strength": strength})
