#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include "TrackerCustomKCF.h"

using namespace std;
using namespace cv;

// prototype of the functino for feature extractor
void sobelExtractor(const Mat img, const Rect2d& roi, Mat& feat);

Ptr<TrackerKCFX> make_tracker(double sigma, bool resize, bool compress_feature, unsigned compressed_size, double detect_thresh, double max_patch_size) {
    TrackerKCFX::Params param;
    if(compress_feature){
        param.desc_pca = TrackerKCFX::GRAY; // | TrackerKCFX::CN;
        param.desc_npca = 0;
    } else {
        param.desc_npca = TrackerKCFX::GRAY | TrackerKCFX::CN;
    }

    param.compress_feature = compress_feature;
    param.compressed_size = compressed_size;
    param.resize = resize;//false;
    param.max_patch_size = max_patch_size; // 100*100
    param.sigma = sigma; // 2.2f; // TODO: very important, kernel variance
    param.detect_thresh = detect_thresh; //0.2;

    Ptr<TrackerKCFX> tracker = TrackerKCFX::create(param);
//    tracker->setFeatureExtractor(sobelExtractor, compress_feature);
    return tracker;
}

bool init(const Ptr<TrackerKCFX> &tracker, const Mat &im, const MultiRect& roi){
    return tracker->init(im, roi);
}

bool update(const Ptr<TrackerKCFX> &tracker, const Mat &im, MultiRect& roi) {
    return tracker->update(im, roi);
}

void sobelExtractor(const Mat img, const Rect2d& roi, Mat& feat){
    MultiRect mroi = static_cast<const MultiRect&>(roi);
    size_t n_rois = mroi.X.size();
    Mat sobel[2 * n_rois];

    for (int i=0; i < mroi.X.size(); ++i) {
        Rect roi = Rect((int) mroi.X[i],(int)  mroi.Y[i], (int) mroi.W[i], (int) mroi.H[i]);
        Rect region = roi;
        Mat patch;
        //! [insideimage]
        // extract patch inside the image
        if (roi.x < 0) {
            region.x = 0;
            region.width += roi.x;
        }
        if (roi.y < 0) {
            region.y = 0;
            region.height += roi.y;
        }
        if (roi.x + roi.width > img.cols)region.width = img.cols - roi.x;
        if (roi.y + roi.height > img.rows)region.height = img.rows - roi.y;
        if (region.width > img.cols)region.width = img.cols;
        if (region.height > img.rows)region.height = img.rows;
        //! [insideimage]

        patch = img(region).clone();
        cvtColor(patch, patch, CV_BGR2GRAY);

        //! [padding]
        // add some padding to compensate when the patch is outside image border
        int addTop, addBottom, addLeft, addRight;
        addTop = region.y - roi.y;
        addBottom = (roi.height + roi.y > img.rows ? roi.height + roi.y - img.rows : 0);
        addLeft = region.x - roi.x;
        addRight = (roi.width + roi.x > img.cols ? roi.width + roi.x - img.cols : 0);

        copyMakeBorder(patch, patch, addTop, addBottom, addLeft, addRight, BORDER_REPLICATE);
        //! [padding]

        //! [sobel]
        Sobel(patch, sobel[i * 2 + 0], CV_32F, 1, 0, 1);
        Sobel(patch, sobel[i * 2 + 1], CV_32F, 0, 1, 1);

        //! [sobel]
    }
    //! [postprocess]
    merge(sobel, n_rois * 2, feat);
    feat.convertTo(feat,CV_32F);
    feat=feat / 255.0-0.5; // normalize to range -0.5 .. 0.5
    //! [postprocess]
}
