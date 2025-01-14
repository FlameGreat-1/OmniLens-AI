import gettext
import os

class LocalizationManager:
    def __init__(self):
        self.translations = {}

    def load_translations(self, locale_dir):
        for lang in os.listdir(locale_dir):
            lang_path = os.path.join(locale_dir, lang, 'LC_MESSAGES', 'messages.mo')
            if os.path.exists(lang_path):
                translation = gettext.translation('messages', localedir=locale_dir, languages=[lang])
                self.translations[lang] = translation

    def set_language(self, lang):
        if lang in self.translations:
            self.translations[lang].install()

    def get_text(self, text):
        return gettext.gettext(text)


