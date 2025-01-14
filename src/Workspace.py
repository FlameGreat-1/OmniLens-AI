from PyQt5.QtWidgets import QDockWidget, QListWidget

class WorkspaceManager:
    def __init__(self, main_window):
        self.main_window = main_window
        self.dock_widgets = {}

    def add_dock_widget(self, name, widget, area):
        dock = QDockWidget(name, self.main_window)
        dock.setWidget(widget)
        self.main_window.addDockWidget(area, dock)
        self.dock_widgets[name] = dock

    def remove_dock_widget(self, name):
        if name in self.dock_widgets:
            self.main_window.removeDockWidget(self.dock_widgets[name])
            del self.dock_widgets[name]

    def save_layout(self):
        return self.main_window.saveState()

    def load_layout(self, state):
        self.main_window.restoreState(state)

# Save and load layouts
layout_state = workspace_manager.save_layout()
workspace_manager.load_layout(layout_state)
