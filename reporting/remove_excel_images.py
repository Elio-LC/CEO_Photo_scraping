import openpyxl
from pathlib import Path
import argparse
import sys

def remove_images_from_excel(file_path: Path, output_path: Path = None):
    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        # Try looking in current directory if not found
        if file_path.parent.name == 'scrapephoto':
             alt_path = Path(file_path.name)
             if alt_path.exists():
                 print(f"Found file in current directory: {alt_path}")
                 file_path = alt_path
             else:
                 return
        else:
            return

    print(f"Loading workbook: {file_path}")
    try:
        wb = openpyxl.load_workbook(file_path)
    except Exception as e:
        print(f"Error loading workbook: {e}")
        return

    total_images_removed = 0

    for ws in wb.worksheets:
        # openpyxl stores images in the _images list attribute of the worksheet
        # This is an internal attribute but commonly used for this purpose
        if hasattr(ws, '_images'):
            count = len(ws._images)
            if count > 0:
                print(f"Sheet '{ws.title}': Removing {count} images...")
                ws._images = [] # Clear the list of images
                total_images_removed += count
            else:
                print(f"Sheet '{ws.title}': No images found.")
        else:
             print(f"Sheet '{ws.title}': Does not support image removal (no _images attribute).")

    if output_path is None:
        output_path = file_path

    print(f"Saving to: {output_path}")
    try:
        wb.save(output_path)
        print(f"Done. Removed {total_images_removed} images.")
    except PermissionError:
        print(f"Error: Permission denied when saving to {output_path}. Please close the file if it is open in Excel.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove all images from an Excel file.")
    parser.add_argument("file_path", type=Path, nargs='?', default=Path("scrapephoto/ceo_photo_check2.xlsx"), help="Path to the Excel file")
    parser.add_argument("--output", type=Path, help="Path to save the modified file (defaults to overwriting input)")
    
    args = parser.parse_args()
    
    remove_images_from_excel(args.file_path, args.output)
