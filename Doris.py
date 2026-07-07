import sys
import os
import time
import subprocess
from util import select_folder

from Step4_sort_fin_automatic import FinFeatureSorter
from Step5_sort_fin_by_hand import FinInteractiveLabeler
from Step6_merge_preview import FinMergePreview
from Step7_pair_fin import pair_fin
from Step8_statistics import statistics
from  Step9_apply_change import convert_symlinks


def TUI(root_dir):
    choice = "0"
    EXIT = "-1"
    STEP1 = "1"
    STEP2 = "2"
    STEP3 = "3"
    STEP4 = "4"
    STEP5 = "5"
    STEP6 = "6"
    STEP7 = "7"
    STEP8 = "8"
    STEP9 = "9"
    while( True ):
        choice = input("------Dolphin Reidentify Script Toolkit-------------\n" +
                       "ROOT_DIR: %s\n"%(root_dir) +
                       " [ 0]: Select Image Root Folder\n" + 
                       " [ 1]: Locate and Crop Fin\n" +
                       " [ 2]: Filter quality fin image\n" +
                       " [ 3]: Compute fin feature fingerprint\n" +
                       " [ 4]: Auto Cluster High Similar FIN\n" +
                       " [ 5]: Select Image same with reference\n" + 
                       " [ 6]: Merge Preview\n" + 
                       " [ 7]: Pair Fin and Find Behavior Group\n" + 
                       " [ 8]: Plot Group Matching Review\n" + 
                       " [ 9]: Convert all soft link to regular file\n" +
                       " [-1]: Exit\n" +
                       "Type Step Number: ")
        if choice == "0":
            root_dir = select_folder()
            print("Working root diretory is ", root_dir)
        elif choice == STEP1:
            print("Locate and Crop Fin") 
            from Step1_crop_fin import FinCropper
            cropper = FinCropper()
            meta_df = cropper.crop(root_dir)
        elif( choice == STEP2 ):
            print("Filter out low quality and low confidence fin")
            from Step2_filter_low_quality import FinQualityFilter
            filter_obj = FinQualityFilter(root_dir=root_dir)
            filter_obj.auto_filter()
            ret = "N" # TODO: move choice inside filter module
            while (ret != "Y"):
                ret = input("Done auto filering, please the FIN folder and recorrect result\n"
                            +"Type Y to continue: ")
                if ret == "Y":
                    filter_obj.confirm_filter()
        elif( choice == STEP3 ):
            from Step3_extract_fin_feature import FinFeatureExtractor
            extractor = FinFeatureExtractor()
            extractor.extract(root_dir)
            print("Compute the fin fingerprint DONE")
        elif( choice == STEP4 ):
            print("Automatic connect high similar fin")
            sorter = FinFeatureSorter(root_dir=root_dir)
            sorter.run()
        elif( choice == STEP5):
            print("Manual cofirm the same fin in similar fin image")
            viewer_script = os.path.join(os.getcwd(), 'ImageGridViewer.py')
            viewer_proc = subprocess.Popen([sys.executable, viewer_script])
            time.sleep(1)
            labeler = FinInteractiveLabeler(root_dir=root_dir, threshold=0.5)
            try:
                labeler.run()
            finally:
                viewer_proc.terminate()
                try:
                    viewer_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    viewer_proc.kill()
        elif( choice == STEP6):
            print("Merge Preview")
            viewer_script = os.path.join(os.getcwd(), 'ImageGridViewer.py')
            viewer_proc = subprocess.Popen([sys.executable, viewer_script])
            time.sleep(1)
            merger = FinMergePreview(root_dir=root_dir)
            try:
                merger()
            finally:
                viewer_proc.terminate()
                try:
                    viewer_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    viewer_proc.kill()
        elif( choice == STEP7):
            print("Find Relationship and move to NN folder")
            pair_fin(root_dir)
        elif( choice == STEP8):
            print("Review Group Matching")
            statistics(root_dir)
        elif( choice == STEP9):
            print("Convert all soft link to regular file")
            convert_symlinks(root_dir)
        elif( choice == EXIT):
            print("Exiting now")
            break
        else:
            print("No correct step number was provide")
            pass

if __name__ == "__main__":
    import sys
    #root_dir = "/media/filming/2025-白海豚/20240825-JM_02-3//"
    if len(sys.argv) == 2:
        root_dir = sys.argv[1]
    else:
        root_dir = None
    TUI(root_dir)
