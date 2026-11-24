"""Download the Ausgrid Solar Home Electricity Data (archive copy) and extract the half-hourly CSVs.

Ausgrid removed the dataset from its own website in 2026. The only complete copy we found is the
archive published by P. Haessig (see README, "Data"). Skip this step if you use the processed file
data/processed/hourly.csv that is committed to the repository.

Usage:  python scripts/download_data.py
"""
import urllib.request
import zipfile
from pathlib import Path

URL = "https://pierreh.eu/downloads/Ausgrid_solar_home_data.zip"
EXPECTED_BYTES = 57_038_633
DEST = Path("data/raw/ausgrid")
WANT = ["README.md", "Solar home 2010-2011.csv", "Solar home 2011-2012.csv", "Solar home 2012-2013.csv",
        "Ausgrid solar home electricity data notes (Aug 2014).pdf"]


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    zip_path = DEST / "Ausgrid_solar_home_data.zip"
    if not zip_path.exists():
        print("downloading", URL, "(~57 MB)")
        urllib.request.urlretrieve(URL, zip_path)
    if zip_path.stat().st_size != EXPECTED_BYTES:
        raise SystemExit(f"unexpected file size {zip_path.stat().st_size}, expected {EXPECTED_BYTES}; delete and retry")
    with zipfile.ZipFile(zip_path) as z:
        if z.testzip() is not None:
            raise SystemExit("corrupt zip")
        for name in WANT:  # extract only known file names, never arbitrary paths from the archive
            z.extract(name, DEST / "extracted")
    print("done ->", DEST / "extracted")


if __name__ == "__main__":
    main()
