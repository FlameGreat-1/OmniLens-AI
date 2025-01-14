from PyQt5.QtCore import QRunnable, pyqtSlot, QObject, pyqtSignal

class WorkerSignals(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    error = pyqtSignal(str)

class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception as e:
            self.signals.error.emit(str(e))
        else:
            self.signals.finished.emit()

class BatchProcessor:
    def __init__(self, thread_pool):
        self.thread_pool = thread_pool
        self.queue = []

    def add_task(self, task):
        self.queue.append(task)

    def process_queue(self):
        for task in self.queue:
            worker = Worker(task)
            self.thread_pool.start(worker)
