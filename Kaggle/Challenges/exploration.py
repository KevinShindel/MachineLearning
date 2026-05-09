import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def callback(input):
    pass

def main():
    img_path = 'cassava_data/cassava_id_8_888_500_orig.jpeg'
    root = os.getcwd()
    path = os.path.join(root, img_path)

    img = cv2.imread(path)

    height, width, _ = img.shape

    if width > height:
        # wide image → crop left and right
        coef = (width - height) // 2
        img = img[:, coef:coef + height]

    elif height > width:
        # tall image → crop top and bottom
        coef = (height - width) // 2
        img = img[coef:coef + width, :]

    img = cv2.resize(img, (256, 256))

    unsharp = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=5)

    win_name = 'cassava_sharped'
    cv2.namedWindow(win_name)

    cv2.createTrackbar("usm_factor", win_name, 0, 10, callback)


    def usm(orig_img: np.ndarray, unsharp_img: np.ndarray, amount: float = 0.9) -> np.ndarray:
        orig_f = orig_img.astype(np.float64)
        unsharp_f = unsharp_img.astype(np.float64)
        sharpened = orig_f + (orig_f - unsharp_f) * amount
        return np.clip(sharpened, 0, 255).astype(np.uint8)


    while True:
        if cv2.waitKey(1) == ord('q'):
            break

        usm_factor = cv2.getTrackbarPos("usm_factor", win_name)
        usm_factor *= 0.2

        sharped_img = usm(img, unsharp, usm_factor)

        win_name = 'cassava'
        cv2.imshow(win_name, img)

        win_name = 'cassava_sharped'
        cv2.imshow(win_name, sharped_img)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
