import json

class PresetManager:
    def __init__(self):
        self.presets = {}

    def add_preset(self, name, settings):
        self.presets[name] = settings

    def get_preset(self, name):
        return self.presets.get(name)

    def save_presets(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.presets, f)

    def load_presets(self, filename):
        with open(filename, 'r') as f:
            self.presets = json.load(f)

class MacroRecorder:
    def __init__(self):
        self.current_macro = []
        self.macros = {}

    def start_recording(self):
        self.current_macro = []

    def add_action(self, action, *args, **kwargs):
        self.current_macro.append((action, args, kwargs))

    def stop_recording(self, name):
        self.macros[name] = self.current_macro
        self.current_macro = []

    def play_macro(self, name):
        if name in self.macros:
            for action, args, kwargs in self.macros[name]:
                action(*args, **kwargs)

