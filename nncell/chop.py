import numpy as np
from skimage import feature

"""
chop parent image into separate images for each nuclei
"""


def is_outside_img(x, y, img_size, size):
    """
    determine if bounding box around co-ordinates is outside the image limits

    Parameters:
    ------------
    x : number
        x co-ordinate
    y : number
        y co-ordinate
    img_size : list [num, num]
        image shape
    size : number
        width and height of the bounding box (in pixels)

    Returns:
    ---------
    Boolean:
        True if box exceeds image boundary
        False if box lies within image boundary
    """
    dist = size / 2.0
    if (x + dist > img_size[0]) or (x - dist < 0):
        return True
    if (y + dist > img_size[1]) or (y - dist < 0):
        return True
    return False


def nudge_coords(x, y, img_size, size):
    """
    move co-ordinates to fit box within image

    Parameters:
    ------------
    x : number
        x co-ordinate
    y : number
        y co-ordinate
    img_size : [num (x), num (y)]
        image shape
    size : number
        width and height of the bounding box (in pixels)
    """
    _check_size(size)
    dist = size / 2
    # for x co-ordinate
    if (x + dist) > img_size[0]:
        # past x max limit
        diff = abs((x + dist) - img_size[0])
        x -= diff
    if (x - dist) < 0:
        # past x min limit
        diff = abs(x - dist)
        x += diff
    # for y co-ordinate
    if (y + dist) > img_size[1]:
        # past y max limit
        diff = abs((y + dist) - img_size[1])
        y -= diff
    if (y - dist) < 0:
        # past y min limit
        diff = abs(y - dist)
        y += diff
    return [x, y]



def crop_to_box(x, y, img, size, edge="keep"):
    """
    create bounding box around co-ordinates
    Parameters:
    ------------
    x : number
        x co-ordinate
    y : number
        y co-ordinate
    img : np.array
        image
    size : number
        width and height of bounding box (in pixels)
    edge : string
        options : ("keep", "remove")
            what to do for a box which would go beyond the edge of the image.
        "keep" : will keep the box within the image boundaries at the specified
            size, though the cells may not be centered within the box.
        "remove" : do not use points which will have boxes beyond the image
            boundary.
    """
    _check_size(size)
    _check_edge_args(edge)
    for dim in img.shape:
        if dim < size:
            raise ValueError("image is too small for specified box size")
    dist = size / 2
    # determine if the box will be within the image
    if is_outside_img(x, y, img.shape, size):
        if edge == "keep":
            # adjust x, y co-ordinates
            x, y = nudge_coords(x, y, img.shape, size)
        if edge == "remove":
            # don't use this x,y co-ordinate
            return None
    return img[x - dist: x + dist, y - dist: y + dist]


def chop_nuclei(img, size=100, edge="keep", threshold=0.1, **kwargs):
    """
    Chop an image into separate images for each nuclei. Each image will be the
    same dimensions of `size`*`size` pixels. Nuclei on the edge of the image
    can either be kept or removed, though if kept they may not be centered
    within the individual image.

    Parameters:
    ------------
    img : numpy.array
        image
    size : integer
        size of image, must be a positive even integer (in pixels)
    edge : string
        what to do with nuclei on the ed`ge of the parent image
        options:
            keep   : nuclei will be kept though they may not be centered within
                     the individual image
            remove : nuclei near the edge of the image will be ignored
    threshold : number (default = 0.1)
        threshold argument to skimage.feature.blob_dog
    **kwargs : additional arguments to skimage.feature.blob_dog to detect the
        nuclei.
    """
    _check_edge_args(edge)
    # find nuclei positions within the image
    nuclei = feature.blob_dog(img, threshold=threshold, **kwargs)
    # loop through x-y co-ordinates for each nucleus
    # create list of sub-arrays
    cropped_imgs = []
    for x, y, _ in nuclei:
        cropped = crop_to_box(x, y, img, size, edge)
        cropped_imgs.append(cropped)
    # remove any empty arrays
    cropped_remove_na = [i for i in cropped_imgs if i is not None]
    return np.stack(cropped_remove_na)


def _check_size(size):
    """checks size is an even positive integer"""
    if size % 2 != 0:
        raise ValueError("size must be an even number")
    if size < 0:
        raise ValueError("size must be positive")
    if not isinstance(size, int):
        raise ValueError("size must be an integer")


def _check_edge_args(edge):
    edge_args = ["keep", "remove"]
    if edge not in edge_args:
        raise ValueError("unknown edge argument. options: {}".format(edge_args))
