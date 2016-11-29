import os

def get_filename(file_path):
    """return the final image URLS from a file path"""
    filename = str(file_path.split(os.sep)[-1])
    return filename.strip()


def get_well(img_url, char="_"):
    """return well from image URL"""
    return str(img_url.split(char)[1])


def get_site(img_url, char="_"):
    """return site from image URL"""
    site_str = img_url.split(char)[2]
    return int("".join(x for x in site_str if x.isdigit()))


def get_channel(img_url, char="_"):
    """return channel from image URL"""
    return int(img_url.split(char)[3][1])
