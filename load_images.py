import numpy as np
import pandas as pd
import parse_paths
from skimage import io
from skimage import img_as_float


class ImageLoader(object):
    """
    Load ImageXpress images in a way that is suitable to keras and
    processing with Theano's 2D convolutional layers.

    The image dimension ordering follows Theano's convention:
        (n_samples, n_channels, x_dim, y_dim)

    This store images in memory as numpy arrays.
    Not sure if there's an efficient way to load images when needed without
    really slowing down the training.
    """

    def __init__(self, url_list, **kwargs):
        self.url_list = _prune(url_list, **kwargs)
        self.img_list = None
        self.grouped_channels = None
        self.img_array = None


    def group_channels(self):
        """
        given a list of img urls, this will group them into the same well and
        site
        # TODO make sure the channels are in the right order
        """
        grouped_list = []
        urls = [parse_paths.get_filename(i) for i in self.url_list]
        tmp_df = pd.DataFrame(self.url_list, columns=["img_url"])
        tmp_df["well"] = [parse_paths.get_well(i) for i in urls]
        tmp_df["site"] = [parse_paths.get_site(i) for i in urls]
        grouped_df = tmp_df.groupby(["well", "site"])
        for _, group in grouped_df:
            grouped_list.append(list(group["img_url"]))
        self.grouped_channels = grouped_list


    def exclude_channels(self, channels_to_exclude, from_url_list=False):
        """
        Given a list of integers, these channels will be removed from
        self.grouped_channels.

        If from_url_list is True, then these channels will be removed from
        self.url_list as well and will be lost.
        """
        # TODO
        pass


    @staticmethod
    def _load_image(img):
        """load single image into numpy array"""
        return img_as_float(io.imread(img))


    def add_metadata(self, metadata):
        """
        attach labels to dataset
        Can match this by plate & well
        """
        if isinstance(metadata, pd.DataFrame):
            # do dataframe stuff
            pass
        if isinstance(metadata, str):
            # load to dataframe, then do stuff
            pass
        if isinstance(metadata, dict):
            # no idea -- figure something out
            pass
        # TODO


    def prep_images(self):
        """load images into np.arrays and store in img_list"""
        images = []
        for group in self.grouped_channels:
            images.append(list([self._load_image(i) for i in group]))
        self.img_array = np.asarray(images, dtype=float)


def _prune(url_list, ext=".tif"):
    """
    remove weird images from url_list such as thumbnails
    Outside the class as can't call self.function within __init__
    """
    good_images = []
    for url in url_list:
        if not "thumb" in url and url.endswith(ext):
            good_images.append(url)
    return good_images

