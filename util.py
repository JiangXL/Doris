import os
import pyexiv2


def select_folder(title="Select Folder") -> str | None:
    """弹出系统文件对话框，让用户选择一个文件夹。

    Args:
        title: 对话框标题

    Returns:
        选中的文件夹绝对路径；如果用户取消则返回 None
    """
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    root.attributes("-topmost", True)  # 确保对话框置顶
    folder_path = filedialog.askdirectory(title=title)
    root.destroy()

    if folder_path and os.path.isdir(folder_path):
        return os.path.abspath(folder_path)
    return None


def read_exif_FocusPosition2(image_path):
    # Return FocusPosition if exist, or -1
    FocusPosition2 = -1
    with pyexiv2.Image(image_path) as img:
        metadata = img.read_exif()
    
        # Sony FocusPosition is usually in the Makernote tag or FocusInfo 
        focus_tag_key = 'Exif.Sony2Fp.FocusPosition2' 
    
        if focus_tag_key in metadata:
            FocusPosition2 = int(metadata[focus_tag_key])
    return FocusPosition2
