from libcpp cimport bool
from libcpp.vector cimport vector

cdef extern from "<opencv2/core.hpp>":
    int CV_8UC1
    int CV_8UC3

cdef extern from "<opencv2/core.hpp>" namespace "cv":
    cdef cppclass _InputArray
    cdef cppclass Rect_[T]
    cdef cppclass Ptr[T]:
        Ptr()
        pass

    cdef cppclass Mat :
        Mat() except +
        Mat( int height, int width, int type, void* data  ) except+
        Mat( int height, int width, int type ) except+

    ctypedef _InputArray InputArray
    cdef cppclass Rect2d:
        double x, y, width, height
        Rect2d()
        Rect2d(int x, int y, int w, int h)

cdef extern from "<opencv2/tracking.hpp>" namespace "cv":
    cdef cppclass Tracker:
        bool init(Mat image, Rect2d boundingBox)
        bool update(Mat image, Rect2d boundingBox);

    cdef cppclass TrackerMedianFlow:
        @staticmethod
        Ptr[Tracker] create()

    cdef cppclass MultiTrackerTLD:
        MultiTrackerTLD()
        bool addTarget(Mat image, Rect2d boundingBox, Ptr[Tracker] trackingAlgorithm)
        bool update(Mat image)
        vector[Rect2d] boundingBoxes
        int targetNum