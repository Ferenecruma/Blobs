from random import randint
from PIL import Image
import numpy as np
from scipy.signal import correlate2d


def correlate_2d(image: np.array, ffilter: np.array) -> np.array:
    """ 
    Cross-correlate two 2-dimensional arrays (grayscale images).

    Correlation formula for 2-d arrays is:
        F * I(x, y) = Sum(j from -N to N)(Sum i from -N to N) of F(i, j) * I(x + i, y + j).
    So basically we just need to take into consideration corner cases, when x + i > N or x + 1 < 0.
    It can be achived by padding array with zeros.
    """

    # for convenience let's assume our filter has odd number of rows and columns
    # it's easy to make such a filter from original one, just pad it with zeros
    if len(ffilter) % 2 == 0:
        ffilter = np.pad(ffilter, [(1, 0), (0, 0)], 'constant')
    if len(ffilter[0]) % 2 == 0:
        ffilter = np.pad(ffilter, [(0, 0), (1, 0)], 'constant')
        
    h, w = len(ffilter), len(ffilter[0])
    # add padding for computation convenience.
    padded_image = np.pad(image, [(h // 2, h // 2), (w // 2, w // 2)], 'constant')
    # output array will have same shape as image 
    res_image = np.zeros(image.shape)

    for i in range(len(image)):
        for j in range(len(image[0])):
            res_image[i][j] = np.sum(padded_image[i:i+h, j:j+w] * ffilter)
    return res_image


def correlate_images(image1: Image, image2: Image) -> Image:
    """Produce a correlation image given 2 grayscale images."""
    par1 = np.asarray(image1.convert('L'))
    par2 = np.asarray(image2.convert('L'))
    corr_array = correlate_2d(par1, par2)
    return Image.fromarray(np.uint8(corr_array))


if __name__ == "__main__":
    # Function testing
    while test_cases_passed != 1000:
        a, b = randint(1, 100), randint(1, 100)
        test_image = np.zeros((a, b))
        h, w = test_image.shape
        test_image[h // 2][w // 2] = 1

        a, b = randint(1, 100), randint(1, 100)
    
        shape = (a, b)
        test_filter = np.arange(1, shape[0] * shape[1] + 1, dtype=float).reshape(shape)

        res = correlate_2d(test_image, test_filter)
        res1 = correlate2d(test_image, test_filter, mode="same")

        assert np.array_equal(res, res1)





