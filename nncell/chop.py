from skimage import feature

def find_nuclei(img, threshold=0.1, **kwargs):
    """
    find nuclei, return array
    Returns:
    +---+---+-------+
    | x | y | sigma |
    +---+---+-------+
    """
    return feature.blob_dog(img, threshold, **kwargs)


def is_outside_img(x, y, img_size, size):
    """
    determine if bounding box around co-ordinates is
    outside the image limits

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
    if (x + dist < img_size[0]) or (x - dist > 0):
        return False
    if (y + dist < img_size[1]) or (y - dist > 0):
        return False
    return True


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
    dist = size / 2.0
    # find delta x
    if x + dist > img_size[0]:
        d_x = (x + dist) - img_size[0]
    elif x - dist < 0:
        d_x = abs(x - dist)
        assert d_x < 0
    # find delta y
    if y + dist > img_size[1]:
        d_y = (y + dist) - img_size[0]
    elif y - dist < 0:
        d_y = abs(y - dist)
        assert d_y < 0
    # new x, y co_ordinates
    new_x = x - d_x
    new_y = y - d_y
    return [new_x, new_y]


def crop_to_box(x, y, img, size=100, edge="keep"):
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
    edge : string (default "keep)
        options = ("keep", "remove") what to do for box which would go beyond
        the edge of the image.
        "keep" : will keep the box within the image boundaries at the specified size,
        though the cells may not be centered within the box.
        "remove" : do not use points which will have boxes beyond the image boundary.
    """
    dist = size / 2.0
    # determine if the box will be within the image
    if is_outside_img(x, y, img.shape, size):
        if edge == "keep":
            # adjust x, y co-ordinates
            x, y = nudge_coords(x, y, img.size, size)
        elif edge == "remove":
            # don't use this x,y co-ordinate
            return None
        else:
            raise ValueError("unknown edge argument, options: keep, remove")
    else:
        # not near the edge, just crop and return image
        return img[x - dist: x + dist, y - dist: y + dist]
