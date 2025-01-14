import sys
import os
import logging
from PyQt5.QtWidgets import QApplication, QSplashScreen
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QTimer

# Import custom modules
from src.gui.main_window import MainWindow
from src.utils.translator import translator, update_ui_language
from src.utils.file_utils import load_json, create_directory
from src.image_processor import ImageProcessor
from src.object_detector import ObjectDetector
from src.face_detector import FaceDetector
from src.auto_enhancer import AutoEnhancer
from src.cloud_manager import CloudManager
from src.social_media_manager import SocialMediaManager
from src.collaboration_manager import CollaborationManager
from src.version_control import VersionControl

# Setup logging
logging.basicConfig(filename='omnilens_ai.log', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OmniLensAI:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.load_config()
        self.init_translator()
        self.init_managers()
        self.init_ui()

    def load_config(self):
        try:
            self.config = load_json('configs/app_config.json')
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            self.config = {}  # Use default config if loading fails

    def init_translator(self):
        default_language = self.config.get('default_language', 'en')
        translator.set_language(default_language)

    def init_managers(self):
        self.image_processor = ImageProcessor()
        self.object_detector = ObjectDetector()
        self.face_detector = FaceDetector()
        self.auto_enhancer = AutoEnhancer()
        self.cloud_manager = CloudManager(self.config.get('cloud_config', {}))
        self.social_media_manager = SocialMediaManager(self.config.get('social_media_config', {}))
        self.collaboration_manager = CollaborationManager()
        self.version_control = VersionControl()

    def init_ui(self):
        # Show splash screen
        splash_pix = QPixmap('resources/images/splash.png')
        splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
        splash.show()
        self.app.processEvents()

        # Initialize main window
        self.main_window = MainWindow(
            image_processor=self.image_processor,
            object_detector=self.object_detector,
            face_detector=self.face_detector,
            auto_enhancer=self.auto_enhancer,
            cloud_manager=self.cloud_manager,
            social_media_manager=self.social_media_manager,
            collaboration_manager=self.collaboration_manager,
            version_control=self.version_control
        )

        # Close splash screen and show main window
        QTimer.singleShot(3000, splash.close)
        QTimer.singleShot(3000, self.main_window.show)

    def run(self):
        try:
            sys.exit(self.app.exec_())
        except Exception as e:
            logger.critical(f"Application crashed: {e}")

def check_dependencies():
    required_dirs = ['configs', 'resources', 'models']
    for dir in required_dirs:
        if not os.path.exists(dir):
            logger.warning(f"Required directory '{dir}' not found. Creating it.")
            create_directory(dir)

    # Check for other critical dependencies
    try:
        import cv2
        import numpy
        import tensorflow
    except ImportError as e:
        logger.critical(f"Critical dependency not found: {e}")
        sys.exit(1)

if __name__ == "__main__":
    check_dependencies()
    omnilens_ai = OmniLensAI()
    omnilens_ai.run()
