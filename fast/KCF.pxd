from libcpp cimport bool
from libcpp.vector cimport vector

cdef extern from "<opencv2/core.hpp>":
    int CV_8UC1
    int CV_8UC3

cdef extern from "<opencv2/core.hpp>" namespace "cv":

    cdef cppclass _InputArray
    cdef cppclass Rect_[T]
    cdef cppclass Ptr[T]:
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
    cdef cppclass TrackerKCF:
        bool init(Mat image, Rect2d boundingBox)
        bool update(Mat image, Rect2d boundingBox );

cdef extern from "TrackerCustomKCF.h" namespace "cv":
    cdef cppclass MultiRect:
        double x, y, width, height
        MultiRect()
        MultiRect(vector[double] _X, vector[double] _Y,  double w, double h)

    cdef cppclass TrackerKCFX:
        bool init(Mat image, Rect2d boundingBox)
        bool update(Mat image, Rect2d boundingBox)

cdef extern from "KCFUtils.cpp":
    Ptr[TrackerKCFX] make_tracker(double sigma, bool resize, bool compress_feature, unsigned compressed_size, double detect_thresh, double max_patch_size)
    bool update(Ptr[TrackerKCFX] tracker, Mat im, MultiRect roi)
    bool init(Ptr[TrackerKCFX] tracker, Mat im, MultiRect roi)