import cv2


def inflateThePicture(img):
    # im_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    im_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    initial_saturation = im_hsv[:,:,1]

    gained_saturation = initial_saturation.copy()
    height, width = initial_saturation.shape

    for i in range(height):
        for j in range(width):
            new_val = initial_saturation[i, j] * 2.0
            if new_val > 1.0:
                new_val = 1.0
            gained_saturation[i, j] = new_val

    im_hsv_gained = im_hsv.copy()
    im_hsv_gained[:,:,1] = gained_saturation[:]

    # im_bgr_gained = cv2.cvtColor(im_hsv_gained, cv2.COLOR_HSV2BGR)

    return  cv2.cvtColor(im_hsv_gained, cv2.COLOR_HSV2RGB)