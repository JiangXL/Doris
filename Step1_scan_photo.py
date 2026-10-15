from Step1a_scan_photo_exif import ExifScanner
from Step1b_analyse_photo_jitter import BurstJitterAnalyzer

if __name__== "__main__":
    import sys
    root_dir = sys.argv[1]
    exif_scanner = ExifScanner()
    exif_scanner.scan(root_dir)

    analyzer = BurstJitterAnalyzer()
    analyzer.run(root_dir)

