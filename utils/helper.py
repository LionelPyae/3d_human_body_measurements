import numpy as np
import cv2
import os
import pathlib
from torch import Tensor


def add_letter_box_text(img, text_pts, color=(0, 255, 0), box_text='', thickness=1, scale=0.8, lower_pos=False):
    x1, y1 = text_pts
    box_text = box_text.capitalize()

    def get_label_position(label, lower_pos=False):
        w, h = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
        offset_w, offset_h = w + 3, h + 5
        xmax = x1 + offset_w
        is_upper_pos = True
        if (y1 - offset_h) < 0 or lower_pos:
            ymax = y1 + offset_h
            y_text = ymax - 2
        else:
            ymax = y1 - offset_h
            y_text = y1 - 2
            is_upper_pos = False
        return xmax, ymax, y_text, is_upper_pos

    if box_text != '':
        *track_loc, is_upper_pos = get_label_position(box_text, lower_pos)
        cv2.rectangle(img, (x1, y1), (track_loc[0], track_loc[1]), color, -1)
        cv2.putText(img, box_text, (x1 + 1, track_loc[2]), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness)
    

def draw_text(image, text, pos=None, scale=0.7, size=1, color=(0, 255, 0), bg_color=(10, 10, 10)):
    if pos is None:
        pos = []
    if len(pos) == 0:
        h, w = image.shape[:2]
        pos = ((w // 2) - 80, 30)
    text_size, _ = cv2.getTextSize(text, 0, scale, size)
    cv2.line(image, pos, (pos[0] + text_size[0], pos[1]), (0, 0, 0), text_size[1] + size * 14)
    cv2.line(image, pos, (pos[0] + text_size[0], pos[1]), bg_color, text_size[1] + size * 10)
    cv2.putText(image, str(text), (pos[0], pos[1] + size * 4), 0, scale, color, size)


def show(imgs, wait=0, window='show', text=None, text_pos=None):
    """
    Display one or more images in a window.

    Args:
        imgs (Union[str, pathlib.PosixPath, List, Tuple, Tensor]): A single image, a list or tuple of images, or a tensor of images.
        wait (int): The number of milliseconds to wait for a key event before closing the window. A value of 0 means the window will not close automatically.
        window (str): The name of the window to display the images in.
        text (str): Optional text to display on the image.
        text_pos (Tuple[int, int]): Optional position to display the text on the image.

    Raises:
        KeyboardInterrupt: If the user presses the ESC key or the 'q' key.
    """
    if text_pos is None:
        text_pos = []

    def make_3channels(img):
        if isinstance(img, (str, pathlib.PosixPath)):
            img = cv2.imread(str(img))
        return np.stack([img] * 3, axis=-1) if img.ndim == 2 else img

    def onMouse(event, x, y, flags, param):
        nonlocal cnt
        if event == cv2.EVENT_LBUTTONDOWN:
            cnt += 1
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            add_letter_box_text(image, (x, y), box_text=f'{cnt}: {x},{y}')
            add_letter_box_text(image, ((image.shape[1] // 2) - 20, 10), box_text=f'{x},{y}')

    if isinstance(imgs, Tensor):
        imgs = imgs.cpu().detach().squeeze().permute(1, 2, 0).numpy()
    elif isinstance(imgs, (str, pathlib.PosixPath)):  # is path
        assert os.path.exists(imgs), f'File not exists {imgs}'
        imgs = cv2.imread(str(imgs))
    elif isinstance(imgs, (list, tuple)):
        imgs = [make_3channels(img) for img in imgs]
        imgs = np.concatenate([*imgs], axis=1)  # horizonal concat
    elif imgs.ndim == 2:
        imgs = np.stack([imgs] * 3, axis=-1)

    image = imgs.copy().astype('uint8')
    cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    if text:
        draw_text(image, str(text), pos=text_pos)

    cv2.imshow(window, image)
    key = cv2.waitKey(wait)
    if key == 27 or key == ord('q') or key == ord(' '):
        raise KeyboardInterrupt
    elif key == ord('c'):
        cv2.setMouseCallback(window, onMouse)
        cnt = 0
        while True:
            cv2.imshow(window, image)
            k = cv2.waitKey(1)
            if k == 27 or k == ord('q') or k == ord(' '):
                cv2.destroyWindow(window)
                break
    elif not wait:
        cv2.destroyWindow(window)
