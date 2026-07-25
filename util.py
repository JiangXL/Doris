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


def read_exif(image_path):
    # Return info from sony exif metadata assume they exit
    exif = {
        "FocusPosition2":-1,
        "DateTime":"",
        "SubSecTime":"",
        "AmbientTemperature":-1,
        "PixelXDimension":-1,
        "PixelYDimension":-1,
        "Exposure":"",
        "FocalLength35mm":-1,
    }
    with pyexiv2.Image(image_path) as img:
        metadata = img.read_exif()
    
        # Sony FocusPosition is usually in the Makernote tag or FocusInfo 
        exif["FocusPosition2"] = int(metadata["Exif.Sony2Fp.FocusPosition2"])
        exif["PixelXDimension"]= int(metadata["Exif.Photo.PixelXDimension"])
        exif["PixelYDimension"]= int(metadata["Exif.Photo.PixelYDimension"])
        exif["AmbientTemperature"] = int(metadata["Exif.Sony2Fp.AmbientTemperature"])
        exif["DateTime"] = metadata["Exif.Photo.DateTimeOriginal"]
        exif["SubSecTime"] = metadata["Exif.Photo.SubSecTimeOriginal"]
        exif["Exposure"] = metadata["Exif.Photo.ExposureTime"]
        exif["FocalLength35mm"] = int(metadata["Exif.Photo.FocalLengthIn35mmFilm"])
    return exif
