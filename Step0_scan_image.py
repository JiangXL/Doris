"""Scan and orangize orignal image information from exif
"""
import os
import glob
import pyexiv2
import pandas as pd

class ImageScanner:
    def __init__(self):
        self.root_dir = ""
        # initilize DataFrame
        self.img_df = pd.DataFrame(
            columns=["orig_name", "orig_img_w", "orig_img_h",
                "focusposition2", "temperature", "datetimesec",
                "exposure", "focuslength35mm"
            ]
        )
        self.meta_filename = "IMAGE_METAINFO.csv"
 
    def read_jpg_info(self):
        imageset_name = os.path.basename(self.root_dir)
        jpg_paths = sorted(glob.glob(
            os.path.join(glob.escape(self.root_dir), "*.JPG")))
        # treat speical characters in paths by glob.escape

        for jpg in jpg_paths:
            with pyexiv2.Image(jpg) as img:
                row= {}
                metadata = img.read_exif()
                row["orig_name"] =  os.path.basename(jpg)
                row["focusposition2"] = int(metadata["Exif.Sony2Fp.FocusPosition2"])
                row["orig_img_w"]= int(metadata["Exif.Photo.PixelXDimension"])
                row["orig_img_h"]= int(metadata["Exif.Photo.PixelYDimension"])
                row["temperature"] = int(metadata["Exif.Sony2Fp.AmbientTemperature"])
                datetime = metadata["Exif.Photo.DateTimeOriginal"]
                subsectime = metadata["Exif.Photo.SubSecTimeOriginal"]
                row["datetimesec"] = f"{datetime}.{subsectime}"
                row["exposure"] = metadata["Exif.Photo.ExposureTime"]
                row["focuslength35mm"] = int(metadata["Exif.Photo.FocalLengthIn35mmFilm"])
            self.img_df.loc[len(self.img_df)] = row

    def group_shot_by_timestamp(self, gap_threshold_s=0.15):
        """
        Sony A1 continuous drive max speed
        AUTO/Electronic Shutter: Continuous shooting: Hi+: 30fps, Hi: 20fps, Mid: 15fps, Lo: 5fps, 
        Mechanical Shutter: Continuous shooting: Hi+: 10fps, Hi: 8fps, Mid: 6fps, Lo: 3fps
        """
        df = self.img_df
        # Combine datetime and subsectime into a Unix timestamp in seconds
        ts = pd.to_datetime(df["datetimesec"].str.strip(), 
                    format="%Y:%m:%d %H:%M:%S.%f").astype("int64") / 1e6

        # Assign group IDs based on timestamp gaps larger than threshold
        df["shot_id"] = (ts.sort_values().diff() > gap_threshold_s).cumsum()
        df["shot_id"] = df["shot_id"].fillna(0).astype(int)
        shot_count = len(df["shot_id"].unique())
        print(f"Found {shot_count} shots")

    def save_to_csv(self, name):
        meta_dir = os.path.join(self.root_dir, "METAINFO")
        os.makedirs(meta_dir, exist_ok=True)
        path = os.path.join(meta_dir, name)
        self.img_df.to_csv(path, index=False)

    def scan(self, root_dir):
        self.root_dir = root_dir
        self.read_jpg_info()
        self.group_shot_by_timestamp()
        self.save_to_csv(self.meta_filename)

if __name__== "__main__":
    import sys
    root_dir = sys.argv[1]
    scanner = ImageScanner()
    scanner.scan(root_dir)
