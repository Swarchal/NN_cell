"""
tests for nncell.image_prep
"""
import os
from nncell import image_prep
from parserix import parse
from parserix import clean
import pytest

# sort out data for tests
TEST_DIR = os.path.abspath("tests")
PATH_TO_IMG_URLS = os.path.join(TEST_DIR, "test_images/images.txt")
IMG_URLS = clean.clean([i.strip() for i in open(PATH_TO_IMG_URLS).readlines()])

####################################
# ImagePrep tests
####################################

def test_ImagePrep_convert_to_rgb():
    pass


def test_ImagePrep_write_img_to_disk():
    pass


def test_ImageDict_sort_channels():
    pass


def test_ImagePrep_create_directories():
    pass


def test_ImagePrep_prepare_images():
    pass


#####################################
# ImageDict tests
#####################################

def test_ImageDict_remove_channels():
    channels_to_remove = [4, 5]
    ImgDict = image_prep.ImageDict()
    ans = ImgDict.remove_channels(IMG_URLS, channels_to_remove)
    # parse channel numbers out of ans
    img_names = [parse.img_filename(f) for f in ans]
    img_channels = [parse.img_channel(name) for name in img_names]
    for channel in img_channels:
        assert channel not in channels_to_remove


def test_ImageDict_keep_channels():
    channels_to_keep = [1, 2, 3]
    ImgDict = image_prep.ImageDict()
    ans = ImgDict.keep_channels(IMG_URLS, channels_to_keep)
    # parse channel numbers out of ans
    img_names = [parse.img_filename(f) for f in ans]
    img_channels = [parse.img_channel(name) for name in img_names]
    for channel in img_channels:
        assert channel in channels_to_keep


def test_ImageDict_add_class():
    ImgDict = image_prep.ImageDict()
    ImgDict.add_class("test", IMG_URLS)
    out = ImgDict.parent_dict
    assert isinstance(out, dict)
    assert out.keys() == ["test"]


def test_ImageDict_group_channels():
    ImgDict = image_prep.ImageDict()
    ImgDict.add_class("test", IMG_URLS)
    ImgDict.group_image_channels()
    out = ImgDict.parent_dict
    assert isinstance(out, dict)


def test_ImageDict_make_dict():
    ImgDict = image_prep.ImageDict()
    ImgDict.add_class("test", IMG_URLS)
    ImgDict.group_image_channels()
    ImgDict.train_test_split()
    out = ImgDict.make_dict()
    assert isinstance(out, dict)


def test_ImageDict_train_test_split():
    ImgDict = image_prep.ImageDict()
    ImgDict.add_class("test", IMG_URLS)
    ImgDict.group_image_channels()
    ImgDict.train_test_split()
    out = ImgDict.make_dict()
    assert isinstance(out, dict)
    assert set(out.keys()) == set(["train", "test"])
    # train test split doesn't lose any images
    n_train = len(out["train"]["test"])
    n_test = len(out["test"]["test"])
    n_images = len(ImgDict._group_channels(IMG_URLS, order=False))
    assert n_train + n_test == n_images
