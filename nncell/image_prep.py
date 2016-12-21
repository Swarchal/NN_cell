"""
Prep Images into RGB .png images for keras ImageDataGenerator.
Need to split these by directory into the different classes.
Also need to split into a test and a training directory
"""

import os
import random
import numpy as np
import pandas as pd
from parserix import parse
import skimage
from skimage import io

class ImagePrep(object):
    """
    Prepare images for Keras.preprocessing.image.ImageDataGenerator.
    Creating directories split into test, training, and within those
    images are in directories named after their class.
    Images stored as three channel (RGB) .png files

    Input:
    -------
    input dictionary from ImageDict:

        train:
            class1 : [image_list]
            class2 : [image_list]
        test:
            class1 : [image_list]
            class2 : [image_list]
    """

    def __init__(self, img_dict):
        if isinstance(dict, img_dict):
            self.img_dict = img_dict
        else:
            raise ValueError("input needs to be a dictionary")


    @staticmethod
    def convert_to_rgb(img_channels):
        """read in three channels and merge to an 8-bit RGB array"""
        image_collection = io.imread_collection(img_channels)
        img_ubyte = skimage.img_as_ubyte(image_collection)
        return np.dstack(img_ubyte)


    @staticmethod
    def write_img_to_disk(img, name, path, extension=".png"):
        """write image to disk"""
        full_path = os.path.join(path, name + extension)
        io.imsave(fname=full_path, arr=img)


    def _check_dict(self):
        """check validity of input dict"""
        # make sure it has train and test sub-dictionaries
        if len(self.img_dict.keys()) != 2:
            raise ValueError("input dict has too few keys")
        # make sure it has train and test sub-dictionaries
        if ("train" or "test") not in self.img_dict.keys():
            raise ValueError("missing train or test keys in input dictionary")
        # TODO more checks, check we have lists of strings in sub-directories
        # check we have the same classes in train and test directories
        train_classes = self.img_dict["train"].keys()
        test_classes = self.img_dict["test"].keys()
        if sorted(train_classes) != sorted(test_classes):
            raise ValueError("train and test sets contain differing classes")


    @staticmethod
    def _make_dir(directory):
        """sensible to way to make directory if it doesn't exist"""
        try:
            os.makedirs(directory)
        except OSError:
            if os.path.isdir(directory):
                pass
            else:
                err_msg = "failed to create directory {}".format(directory)
                raise RuntimeError(err_msg)


    def create_directories(self, base_dir):
        """
        create directory structure for prepared images

        Arguments:
        -----------
        base_dir : string
            Path to directory in which to hold training and test datasets
            A directory will be created if it does not already exist
        """
        self._make_dir(base_dir)
        # create training and test directories
        for group in self.img_dict.keys():
            for key, img_list in self.img_dict[group].items():
                # create directory item/key from key
                new_dir_path = os.path.join(os.path.abspath(base_dir), group, key)
                self._make_dir(new_dir_path)
                # create and save images in new_dir_path
                for i, img in enumerate(img_list, 1):
                    self.write_img_to_disk(img=img, name="img_{}".format(i),
                                           path=new_dir_path)


class ImageDict(object):
    """
    Class to make an image dictionary for ImagePrep()

    No idea on how to handle this yet
        - Iteractively add image lists that are appended to a dictionary?
        - Construct a directory system and give the directory path to ImageDict?
        - Give plate, wells etc for various classes?
    """

    def __init__(self, train_test_sets=False):
        self.train_test_sets = train_test_sets
        self.grouped = False
        self.parent_dict = dict()
        self.train_test_dict = dict()


    @staticmethod
    def _group_channels(url_list, order):
        """
        given a list of img urls, this will group them into the same well and
        site, per plate
        order : sorts channel numbers into numerical order
        """
        grouped_list = []
        urls = [parse.img_filename(i) for i in url_list]
        tmp_df = pd.DataFrame(url_list, columns=["img_url"])
        tmp_df["plate_name"] = [parse.plate_name(i) for i in url_list]
        tmp_df["plate_num"] = [parse.plate_num(i) for i in url_list]
        # get_well and get_site use the image URL rather than the full path
        tmp_df["well"] = [parse.img_well(i) for i in urls]
        tmp_df["site"] = [parse.img_site(i) for i in urls]
        grouped_df = tmp_df.groupby(["plate_name", "plate_num", "well", "site"])
        if order is True:
            # order by channel
            for _, group in grouped_df:
                grouped = list(group["img_url"])
                channel_nums = [parse.img_channel(i) for i in grouped]
                # create tuple(path, channel_number) and sort by channel_number
                sort_im = sorted(zip(grouped, channel_nums), key=lambda x: x[1])
                # return only the file-paths back from the list of tuples
                grouped_list.append([i[0] for i in sort_im])
        elif order is False:
            for _, group in grouped_df:
                grouped_list.append(list(group["img_url"]))
        else:
            raise ValueError("order needs to be a boolean")
        return grouped_list


    @staticmethod
    def _split_train_test(list_like, test_size):
        """randomly split list object into training and test set"""
        test_n = round(test_size * len(list_like))
        train_n = len(list_like) - test_n
        random.shuffle(list_like)
        training = list_like[:train_n]
        test = list_like[-test_n:]
        assert len(list_like) == len(training) + len(test)
        return [training, test]


    @staticmethod
    def get_wells(img_list, well):
        """
        given a list of image paths, this will return the images matching
        the well or wells given in well
        """
        # parse wells from metadata
        wells = [parse.img_well(path) for path in img_list]
        combined = zip(img_list, wells)
        if isinstance(list, well):
            wanted_images = []
            for i in well:
                for path, parsed_well in combined:
                    if i == parsed_well:
                        wanted_images.append(path)
        elif isinstance(str, well):
            wanted_images = []
            for path, parsed_well in combined:
                if well == parsed_well:
                    wanted_images.append(path)
        return wanted_images


    def group_image_channels(self, order=True):
        """group each image list into RGB channels"""
        if self.grouped is True:
            raise Warning("images already grouped")
        if self.train_test_sets is True:
            raise AttributeError("already formed training and test sets")
        for key, img_list in self.parent_dict.items():
            self.parent_dict[key] = self._group_channels(img_list, order)
        self.grouped = True


    def train_test_split(self, test_size=0.3):
        """
        split into train and test sets
        these are stored in separate dictionary keys
        """
        if self.grouped is False:
            raise AttributeError("image channels not grouped")
        # create train and test sub-dictionaries
        self.train_test_dict["test"] = dict()
        self.train_test_dict["train"] = dict()
        # loop through class lists
        # split into training and test, place in approp dicts under the same key
        for key, img_list in self.parent_dict.items():
            train, test = self._split_train_test(img_list, test_size)
            self.train_test_dict["train"][key] = train
            self.train_test_dict["test"][key] = test
        # once finished, indicate we have created training and test sets
        self.train_test_sets = True


    def add_class(self, class_name, url_list):
        """add a new class of images to the url_list"""
        if self.grouped is True:
            raise Warning("channels already grouped, this will need to be " +
                          "called again to group the new class")
        if self.train_test_sets is True:
            raise AttributeError("cannot add new class once training and test" +
                                 " sets have been sampled")
        self.parent_dict[class_name] = url_list


    def make_dict(self):
        """return image dictionary"""
        return self.train_test_dict
