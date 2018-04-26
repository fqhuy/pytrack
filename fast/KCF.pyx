cimport numpy as np
import numpy as np
cimport KCF

cdef class KCFTracker(object):
    cdef KCF.Ptr[KCF.TrackerKCFX] self_
    def __init__(self, double sigma=2.0, bool resize=False, bool compress_feature=True, int compressed_size=4,
                 double detect_thresh=0.2, double max_patch_size=100*100):
        self.self_ = KCF.make_tracker(sigma, resize, compress_feature, compressed_size, detect_thresh, max_patch_size) # double sigma, bool resize, unsigned compressed_size, double detect_thresh

    cpdef init(self, np.ndarray[np.uint8_t, ndim=3, mode="c"] image, list X, list Y, double w, double h):
        cdef int width = image.shape[1]
        cdef int height = image.shape[0]
        cdef int dtype = KCF.CV_8UC3

        cdef KCF.Mat cv_image = KCF.Mat(height, width, dtype, &image[0,0,0])
        cdef KCF.MultiRect rois = KCF.MultiRect(X, Y, w, h)
        # cdef KCF.Rect2d roi = KCF.Rect2d(x, y, w, h)

        KCF.init(self.self_, cv_image, rois)

    cpdef update(self, np.ndarray[np.uint8_t, ndim=3, mode="c"] image, list X, list Y, double w, double h):
        cdef int width = image.shape[1]
        cdef int height = image.shape[0]
        cdef int dtype = KCF.CV_8UC3
        cdef KCF.Mat cv_image = KCF.Mat(height, width, dtype, &image[0,0,0])

        # cdef KCF.Rect2d roi = KCF.Rect2d(0, 0, 0, 0)
        cdef KCF.MultiRect rois = KCF.MultiRect(X, Y, w, h)
        KCF.update(self.self_, cv_image, rois)
        return rois.x, rois.y, rois.width, rois.height


    def __del__(self):
        pass
        # self.self_.release()