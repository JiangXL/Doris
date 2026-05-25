import sys
import os
from PyQt5.QtWidgets import QApplication
from ImageGridViewer import ImageGridViewer

def TUI():
    choice = 0
    EXIT = -1
    STEP1 = 1
    STEP2 = 2
    STEP3 = 3
    STEP4 = 4
    STEP5 = 5
    STEP6 = 6
    STEP7 = 6
    while( True ):
        choice = input()
        if choice == STEP1:
            print("Locate and Crop Fin") 
        elif( choice == STEP2 ):
            print("Filter out low quality and low confidence fin")
        elif( choice == STEP3 ):
            print("Automatic connect high similar fin")
        elif( choice == STEP4):
            print("Manual cofirm the same fin in similar fin image")
        elif( choice == STEP5):
            print("Fine Tune")
        elif( choice == STEP6):
            print("Find Relationship and move to NN folder")
            print("Please check and classify sortted group")
        elif( choice == STEP7):
            print("Check User Sorted Relationship folder")
        elif( choice == EXIT):
            print("Exiting now")
            break
        else:
            pass

if __name__ == "__main__":
    # TODO: TUI
    # Step 1.
    # Step 2.
    # Step 3.
    # Step 4.
    app = QApplication(sys.argv)

    viewer = ImageGridViewer(
        cols=5,
        thumb_size=384
    )   

    viewer.show()
    sys.exit(app.exec_())

