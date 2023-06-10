import math
import numpy as np
import multiprocessing
from numpy.random import randint
from numpy.linalg import norm, eigh
from numpy.fft import fft, ifft

# function for z-normalization
'''
Parameters:
a: the data used to be z-normalized
axis: axis along which the computation is performed
ddof: ddof of numpy.std. Means Delta Degrees of Freedom.

Return: z-normalized result
'''
def zscore(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    mns = a.mean(axis=axis)
    sstd = a.std(axis=axis, ddof=ddof)
    # If axis != 0 and dimension doesn't match
    if axis and mns.ndim < a.ndim:
        res = ((a - np.expand_dims(mns, axis=axis)) /
               np.expand_dims(sstd, axis=axis))
    else:
        res = (a - mns) / sstd

    return np.nan_to_num(res)

# shifting performed along given axis
'''
Parameters:
a: the data to be shifted
shift: shift amount
axis: shift along which axis

Return: shifted result
'''
def roll_zeropad(a, shift, axis=None):
    a = np.asanyarray(a)

    if shift == 0:
        return a
    # If no axis is provided, flat the array and shift
    if axis is None:
        n = a.size
        reshape = True
    else:
        n = a.shape[axis]
        reshape = False

    if np.abs(shift) > n:
        res = np.zeros_like(a)
    elif shift < 0:
        shift += n
        zeros = np.zeros_like(a.take(np.arange(n-shift), axis))
        res = np.concatenate((a.take(np.arange(n-shift, n), axis), zeros), axis)
    else:
        zeros = np.zeros_like(a.take(np.arange(n-shift, n), axis))
        res = np.concatenate((zeros, a.take(np.arange(n-shift), axis)), axis)

    if reshape:
        return res.reshape(a.shape)
    else:
        return res

# Normalized Cross Correlation c
'''
Parameter:
data: data used to compute NCCc

Return: Computed NCCc
'''
def _ncc_c_3dim(data):
    x, y = data[0], data[1]
    den = norm(x, axis=(0, 1)) * norm(y, axis=(0, 1))
    # Why do we need this??
    if den < 1e-9:
        den = np.inf
    # I believe that if dimension is 2 for x and y, we should use fft2 and ifft2.
    # also, I think univariate should be separated from multivariate instead of adding extra axis.

    # I believe that padding along axis = 0 is incorrect. Because, in multivariate case, axis = 0 represents how many time series and we shouldn't add new time series. shape[1]
    # ??????? Need to be discussed
    x_len = x.shape[0]
    fft_size = 1 << (2*x_len-1).bit_length()

    cc = ifft(fft(x, fft_size, axis=0) * np.conj(fft(y, fft_size, axis=0)), axis=0)
    cc = np.concatenate((cc[-(x_len-1):], cc[:x_len]), axis=0)
    # Why we only keep the real part instead of all parts?????
    return np.real(cc).sum(axis=-1) / den

# Perform SBD computation
'''
Parameters:
x: reference data
y: data that will be shifted

Return: shifted y
'''
def _sbd(x, y):
    '''
    Based on discussion with John, SBD is supposed to have two situations.
    The first situation is line by line alignment (local-wise alignment)
    The second situation is 2D aligmment (global-wise alignment)

    Also, we should ignore the up-and-down shifting but only focus on left-and-right shifting.
    '''
    ncc = _ncc_c_3dim([x, y])
    idx = np.argmax(ncc)
    # I believe that len(x) is equal to len(y) because they are supposed to have the same dimension.
    yshift = roll_zeropad(y, (idx + 1) - max(len(x), len(y)))

    return yshift

# Collective shifting
'''
Parameter:
data: data that contains time-series and centroid

Return: "aligned" result
'''
def collect_shift(data):
    x, cur_center = data[0], data[1]
    if np.all(cur_center==0):
        return x
    else:
        return _sbd(cur_center, x)

# ShapeExtraction in the paper
'''
Parameters:
idx: vector that contains the assignment of n time series to k clusters
x: time series data
j: centroid index
cur_center: current centroid

Return: new centroid
'''
def _extract_shape(idx, x, j, cur_center):
    pool = multiprocessing.Pool()
    args = []
    # Find time series in current cluster
    for i in range(len(idx)):
        if idx[i] == j:
            args.append([x[i], cur_center])
    _a = pool.map(collect_shift, args)
    pool.close()

    a = np.array(_a)
    if len(a) == 0:
        indices = np.random.choice(x.shape[0], 1)
        return np.squeeze(x[indices].copy())
        #return np.zeros((x.shape[1]))

    columns = a.shape[1]
    y = zscore(a, axis=1, ddof=1)

    # Following John's advise, we compute centroid for each line pair separately then concatenate them to generate a new one.

    # why y[:,:, 0]?? I believe that we can compute directly for univariate.
    # For multivariate, John and I have hard time to prove these computations are correct or not. Thus, John proposed his advise.
    s = np.dot(y[:, :, 0].transpose(), y[:, :, 0])
    p = np.empty((columns, columns))
    p.fill(1.0/columns)
    p = np.eye(columns) - p
    m = np.dot(np.dot(p, s), p)

    _, vec = eigh(m)
    centroid = vec[:, -1]
    # Why these steps are needed????
    finddistance1 = np.sum(np.linalg.norm(a - centroid.reshape((x.shape[1], 1)), axis=(1, 2)))
    finddistance2 = np.sum(np.linalg.norm(a + centroid.reshape((x.shape[1], 1)), axis=(1, 2)))

    if finddistance1 >= finddistance2:
        centroid *= -1

    return zscore(centroid, ddof=1)

# KShape helper function
''''
Parameters:
x: time series data
k: k clusters
centroid_init: method to initialize centroids
max_iter: maximum iterations

Returns:
idx: the assignments of n time series to k clusters
centroids: k centroids
'''
def _kshape(x, k, centroid_init='zero', max_iter=100):
    m = x.shape[0]
    idx = randint(0, k, size=m)
    if centroid_init == 'zero':
        centroids = np.zeros((k, x.shape[1], x.shape[2]))
    elif centroid_init == 'random':
        indices = np.random.choice(x.shape[0], k)
        centroids = x[indices].copy()
    distances = np.empty((m, k))
    
    for it in range(max_iter):
        old_idx = idx

        for j in range(k):
            for d in range(x.shape[2]):
                # I think here is equal to John's advice about centroid.....
                centroids[j, :, d] = _extract_shape(idx, np.expand_dims(x[:, :, d], axis=2), j, np.expand_dims(centroids[j, :, d], axis=1))
                #centroids[j] = np.expand_dims(_extract_shape(idx, x, j, centroids[j]), axis=1)

        pool = multiprocessing.Pool()
        args = []
        for p in range(m):
            for q in range(k):
                args.append([x[p, :], centroids[q, :]])
        result = pool.map(_ncc_c_3dim, args)
        pool.close()
        r = 0
        for p in range(m):
            for q in range(k):
                distances[p, q] = 1 - result[r].max()
                r = r + 1

        idx = distances.argmin(1)
        if np.array_equal(old_idx, idx):
            break

    return idx, centroids

# KShape function
'''
Parameters:
x: time series data
k: k clusters
centroid_init: method to initialize centroid
max_iter: maximum iterations

Return: clusters
'''
def kshape(x, k, centroid_init='zero', max_iter=100):
    idx, centroids = _kshape(np.array(x), k, centroid_init=centroid_init, max_iter=max_iter)
    clusters = []
    for i, centroid in enumerate(centroids):
        series = []
        for j, val in enumerate(idx):
            if i == val:
                series.append(j)
        clusters.append((centroid, series))

    return clusters


if __name__ == "__main__":
    import sys
    import doctest
    sys.exit(doctest.testmod()[0])
