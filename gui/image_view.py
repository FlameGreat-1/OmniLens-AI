import cv2
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtGui import QImage, QPixmap, QPainter
from PyQt5.QtCore import Qt, pyqtSignal, QRectF

class ImageView(QGraphicsView):
    image_changed = pyqtSignal(np.ndarray)
    zoom_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setInteractive(True)
        self.zoom_factor = 1.0
        self.current_image = None

    def set_image(self, image):
        self.current_image = image
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)
        self.pixmap_item.setPixmap(pixmap)
        self.setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
        self.image_changed.emit(image)

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            factor = 1.25
            self.zoom_factor *= factor
        else:
            factor = 0.8
            self.zoom_factor *= factor
        self.scale(factor, factor)
        self.zoom_changed.emit(self.zoom_factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.original_event = event
            self.viewport().setCursor(Qt.ClosedHandCursor)
            self.original_event = None
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.setDragMode(QGraphicsView.NoDrag)
            self.viewport().setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)

    def get_visible_rect(self):
        visible_rect = self.mapToScene(self.viewport().rect()).boundingRect()
        return visible_rect

    def reset_view(self):
        self.setSceneRect(QRectF(self.pixmap_item.pixmap().rect()))
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
        self.zoom_factor = 1.0
        self.zoom_changed.emit(self.zoom_factor)

    def get_zoom_factor(self):
        return self.zoom_factor
