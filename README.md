# Helper functions for using keras with microscopy data


**N.B**: using the Theano dimension order `(n_samples, n_channels, x_dim, y_dim)`.


## `ImageLoader`

### Example

With a file-list of images to use:

e.g `images.txt`
```
/data/HCC15691/2015-07-31/4014/val screen_B02_s1_w1A52B1177-9DC7-4534-9623-9DCA396EFA00.tif
/data/HCC15691/2015-07-31/4014/val screen_B02_s1_w1_thumb62D4A363-7C7E-40D0-8A9E-55EC6681574D.tif
/data/HCC15691/2015-07-31/4014/val screen_B02_s1_w2D1E11FCC-8DCD-42F5-B2BD-B270F2E9223D.tif
/data/HCC15691/2015-07-31/4014/val screen_B02_s1_w2_thumb16F0EB1B-44C9-4313-9E6A-A7A1F5236885.tif
/data/HCC15691/2015-07-31/4014/val screen_B02_s1_w3E75611A2-A874-4065-BDAC-EE2467105EEB.tif
...
```

Note that we can include thumbnails and other rubbish, as `ImageLoader` will exlude them automatically.

```python
from load_images import ImageLoader

urls = [i.strip() for i in open("images.txt").readlines()]

img_store = ImageLoader(urls)
img_store.group_channels()
img_store.prep_images()
```

This gives us an array of `(n_samples, n_channels, x_dim, y_dim)` with the images converted into numpy arrays of floats. These can be accessed through `img_store.img_array`.

```
array([[[ 0.00834668,  0.00886549,  0.00856031, ...,  0.01026932,
          0.00895705,  0.00831617],
        [ 0.00849928,  0.00842298,  0.00958267, ...,  0.01022354,
          0.00947585,  0.00988785],
        [ 0.00888075,  0.0083772 ,  0.00895705, ...,  0.00933852,
          0.00985733,  0.00972   ],
        ...,
        [ 0.00990311,  0.01055924,  0.00917067, ...,  0.00959792,
          0.01039139,  0.0106508 ],
        [ 0.01133745,  0.01040665,  0.00944533, ...,  0.01075761,
          0.01036088,  0.00961318],
        [ 0.00984207,  0.00967422,  0.00999466, ...,  0.00947585,
          0.0106508 ,  0.00939956]],

       [[ 0.01829557,  0.01651026,  0.01620508, ...,  0.01901274,
          0.01884489,  0.01765469],
        [ 0.01895171,  0.01831083,  0.01788357, ...,  0.01997406,
          0.0199588 ,  0.02067597],
        [ 0.01878386,  0.01838712,  0.01898222, ...,  0.02017243,
          0.01979095,  0.02172885],
        ...
```

These arrays can then be modified by `keras.preprocessing` or `numpy` functions.
