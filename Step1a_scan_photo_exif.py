"""Scan and orangize orignal photo information from exif
"""
import os
import glob
import pyexiv2
import pandas as pd
from tqdm import tqdm

class ExifScanner:
    def __init__(self):
        self.root_dir = ""
        # initilize DataFrame
        self.photo_df = pd.DataFrame(
            columns=["PhotoName", "PhotoWidth", "PhotoHeight",
                "FocusPosition2", "Temperature", "Datetimesec",
                "Exposure", "FocusLength35mm"
            ]
        )
        self.meta_filename = "PHOTO_METAINFO.csv"
 
    def read_jpg_exif(self):
        imageset_name = os.path.basename(self.root_dir)
        jpg_paths = sorted(glob.glob(
            os.path.join(glob.escape(self.root_dir), "*.JPG")))
        # treat speical characters in paths by glob.escape
        
        print("Reading Photo Exif")
        for jpg in tqdm(jpg_paths):
            with pyexiv2.Image(jpg) as photo:
                row= {}
                metadata = photo.read_exif()
                row["PhotoName"] =  os.path.basename(jpg)
                row["PhotoWidth"]= int(metadata["Exif.Photo.PixelXDimension"])
                row["PhotoHeight"]= int(metadata["Exif.Photo.PixelYDimension"])
                row["FocusPosition2"] = int(metadata["Exif.Sony2Fp.FocusPosition2"])
                row["Temperature"] = int(metadata["Exif.Sony2Fp.AmbientTemperature"])
                datetime = metadata["Exif.Photo.DateTimeOriginal"]
                subsectime = metadata["Exif.Photo.SubSecTimeOriginal"]
                row["Datetimesec"] = f"{datetime}.{subsectime}"
                row["Exposure"] = metadata["Exif.Photo.ExposureTime"]
                row["Focuslength35mm"] = int(metadata["Exif.Photo.FocalLengthIn35mmFilm"])
            self.photo_df.loc[len(self.photo_df)] = row

    def group_shot_by_timestamp(self, gap_threshold_s=0.15):
        """
        Sony A1 continuous drive max speed
        AUTO/Electronic Shutter: Continuous shooting: Hi+: 30fps, Hi: 20fps, Mid: 15fps, 
        Mechanical Shutter: Continuous shooting: Hi+: 10fps, Hi: 8fps, Mid: 6fps
        """
        df = self.photo_df
        # Combine datetime and subsectime into a Unix timestamp in seconds
        ts = pd.to_datetime(df["Datetimesec"].str.strip(), 
                    format="%Y:%m:%d %H:%M:%S.%f").astype("int64") / 1e6

        # Assign ShotIDs based on timestamp gaps larger than threshold
        df["ShotID"] = (ts.sort_values().diff() > gap_threshold_s).cumsum()
        df["ShotID"] = df["ShotID"].fillna(0).astype(int)
        shot_count = len(df["ShotID"].unique())
        print(f"Found {shot_count} shots in {len(df)} photos.")

    def save_to_csv(self, name):
        meta_dir = os.path.join(self.root_dir, "METAINFO")
        os.makedirs(meta_dir, exist_ok=True)
        path = os.path.join(meta_dir, name)
        self.photo_df.to_csv(path, index=False)

    def scan(self, root_dir):
        self.root_dir = root_dir
        self.read_jpg_exif()
        self.group_shot_by_timestamp()
        self.save_to_csv(self.meta_filename)

if __name__== "__main__":
    import sys
    root_dir = sys.argv[1]
    scanner = ExifScanner()
    scanner.scan(root_dir)
