import dropbox
from PIL import Image, ImageEnhance
import os
import cv2 as cv
import numpy as np


def get_local_path():
    local_path = os.path.dirname(os.path.abspath(__file__))
    return local_path


def image_normalizer_cropper(image_path):
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    edged = cv.Canny(gray, 30, 200)

    contours, _ = cv.findContours(
        edged.copy(), 
        cv.RETR_LIST, 
        cv.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv.contourArea, reverse=True)[:5]

    approx = image.copy()
    for c in contours:
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02 * peri, True)

        print(len(approx))
        if len(approx) >= 4:
            print("YUH")
            break

    pts = approx.reshape(len(approx) // 2, 4)
    rect = np.zeros((len(approx) // 2, 4), dtype="float32")

    s = pts.sum(axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    max_width = max(int(widthA), int(widthB))
    max_height = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], 
        dtype="float32"
    )

    M = cv.getPerspectiveTransform(rect, dst)
    warp = cv.warpPerspective(image, M, (max_width, max_height))

 
    return warp



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

        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)
        img.save(LOCAL_DOWNLOAD_PATH)

        processed_image = image_normalizer_cropper(LOCAL_DOWNLOAD_PATH)
        cv.imwrite('processed_document.jpg', processed_image)

        num += 1
        
        if num == 3:
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
    local_save_path = get_local_path() + "/images/"

    # list_files_folders("/NBWM/")

    download_and_convert_images(dropbox_folder_path, local_save_path)





