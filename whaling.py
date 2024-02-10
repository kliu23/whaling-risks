import dropbox
from PIL import Image, ImageEnhance
import os
import cv2 as cv
import numpy as np
from image_to_scan import ManualScanner


def get_local_path():
    local_path = os.path.dirname(os.path.abspath(__file__))
    return local_path


def download_and_convert_images(folder_path, local_path):
    dbx = dropbox.Dropbox(os.environ["DROPBOX_ACCESS_TOKEN"])  # Replace with your Dropbox access token

    num = 0
    for entry in dbx.files_list_folder("/NBWM").entries:
        LOCAL_DOWNLOAD_PATH = os.path.join(local_path, entry.name)

        with open(LOCAL_DOWNLOAD_PATH, "wb") as f:
            _, res = dbx.files_download(path=entry.path_lower)
            f.write(res.content)

        img = Image.open(LOCAL_DOWNLOAD_PATH)
        img = img.convert('L')

        scanner = ManualScanner(LOCAL_DOWNLOAD_PATH)
        processed_image = scanner.run()
        
        print(f'{local_path}processed_{entry.name}')
        cv.imwrite(f'{local_path}processed_{entry.name}', processed_image)

        num += 1
        
        if num == 5:
            break
        


def list_files_folders(path):
    dbx = dropbox.Dropbox(os.environ["DROPBOX_ACCESS_TOKEN"])  # Replace with your Dropbox access token
    try:
        result = dbx.files_list_folder(path, recursive=True)
        for entry in result.entries:
            print(entry.path_display)
    except dropbox.exceptions.ApiError as err:
        print(f"API Error: {err}")


if __name__ == "__main__":
    dropbox_folder_path = ""
    local_save_path = get_local_path() + "/processed_images/"

    # list_files_folders("/NBWM/")

    download_and_convert_images(dropbox_folder_path, local_save_path)





