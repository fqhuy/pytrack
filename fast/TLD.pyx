cimport numpy as np
import numpy as np
cimport TLD

cdef class TLDTracker(object):
    cdef TLD.MultiTrackerTLD self_
    def __init__(self):
        self.self_ = TLD.MultiTrackerTLD()

    cpdef init(self, np.ndarray[np.uint8_t, ndim=3, mode="c"] image, list X, list Y, double w, double h):
        cdef int width = image.shape[1]
        cdef int height = image.shape[0]
        cdef int dtype = TLD.CV_8UC3

        cdef TLD.Mat cv_image = TLD.Mat(height, width, dtype, &image[0,0,0])
        # cdef TLD.Rect2d roi
        cdef TLD.Ptr[TLD.Tracker] tracker
        # self.self_ = TLD.MultiTrackerTLD()

        for i in range(len(X)):
            # cdef TLD.Ptr[TLD.Tracker] tracker= TLD.MedianFlowTracker.create()
            tracker = TLD.TrackerMedianFlow.create()
            self.self_.addTarget(cv_image, TLD.Rect2d(X[i], Y[i], int(w), int(h)), tracker)

    cpdef update(self, np.ndarray[np.uint8_t, ndim=3, mode="c"] image):
        cdef int width = image.shape[1]
        cdef int height = image.shape[0]
        cdef int dtype = TLD.CV_8UC3
        cdef TLD.Mat cv_image = TLD.Mat(height, width, dtype, &image[0,0,0])

        cdef bool ok = self.self_.update(cv_image)
        cdef list roi_list = []
        if ok:
            for i in range(self.self_.targetNum):
                roi_list.append((self.self_.boundingBoxes[i].x, self.self_.boundingBoxes[i].y,
                                 self.self_.boundingBoxes[i].width, self.self_.boundingBoxes[i].height))
        return ok, roi_list

    def __del__(self):
        pass
        # self.self_.release()