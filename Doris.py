import sys
import os
from PyQt5.QtWidgets import QApplication
from ImageGridViewer import ImageGridViewer

if __name__ == "__main__":
    app = QApplication(sys.argv)

    viewer = ImageGridViewer(
        cols=5,
        thumb_size=384
    )   

    viewer.show()
    sys.exit(app.exec_())
