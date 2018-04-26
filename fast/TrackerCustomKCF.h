//
// Created by Phan Quoc Huy on 10/27/17.
//

#ifndef OPENCV_TRACKING_TEST_MY_TRACKER_KCF_H
#define OPENCV_TRACKING_TEST_MY_TRACKER_KCF_H

#include <opencv2/tracking.hpp>
#include <vector>
#include <numeric>

namespace cv {
    using namespace std;

    class MultiRect: public Rect2d {
    public:
        typedef double value_type;
        MultiRect() {

        }
        MultiRect(value_type x, value_type y, value_type w, value_type h): Rect2d(x, y, w, h) {
            X = vector<value_type>({x});
            Y = vector<value_type>({y});
            W = vector<value_type>({w});
            H = vector<value_type>({h});
        }

        MultiRect(const vector<value_type> _X, const vector<value_type> _Y,  value_type w, value_type h) :
                X(_X), Y(_Y)
        {
            for (int i = 0; i < _X.size(); ++i) {
                W.push_back(w);
                H.push_back(h);
            }

            size_t N = X.size();
            x = accumulate(_X.begin(), _X.end(), 0.0) / N;
            y = accumulate(_Y.begin(), _Y.end(), 0.0) / N;
            width = w; height = h;
//            width = (*max_element(_X.begin(), _X.end()) + w) - (*min_element(_X.begin(), _X.end()));
//            height = (*max_element(_Y.begin(), _Y.end()) + h) - (*min_element(_Y.begin(), _Y.end()));
        }

        MultiRect(const vector<value_type> _X, const vector<value_type> _Y, const vector<value_type> _W, const vector<value_type> _H):
                X(_X), Y(_Y), W(_W), H(_H)
        {
            size_t N = _X.size();
            width = W[0]; height = H[0];
            x = accumulate(_X.begin(), _X.end(), 0.0) / N;
            y = accumulate(_Y.begin(), _Y.end(), 0.0) / N;

        }

        inline void moveBy(value_type dx, value_type dy) {
            x += dx;
            y += dy;
            for(int i=0; i < X.size(); ++i) {
                X[i] += dx;
                Y[i] += dy;
            }
        }

        inline void scaleBy(value_type sx, value_type sy, value_type sw, value_type sh) {
            width *= sw; height *= sh;
            x *= sx; y *= sy;
            for(int i=0; i < W.size(); ++i) {
                W[i] *= sw;
                H[i] *= sh;
                X[i] *= sx;
                Y[i] *= sy;
            }
        }

        inline void setXY(value_type x, value_type y) {
            for(int i=0; i < X.size(); ++i) {
                X[i] = x;
                Y[i] = y;
            }
        }

        inline void setWH(value_type w, value_type h) {
            width = w; height = h;
            for(int i=0; i < W.size(); ++i) {
                W[i] = w;
                H[i] = h;
            }
        }

        vector<value_type> X, Y, W, H;
    };

    class CV_EXPORTS_W TrackerKCFX : public Tracker {
    public:
        /**
        * \brief Feature type to be used in the tracking grayscale, colornames, compressed color-names
        * The modes available now:
        -   "GRAY" -- Use grayscale values as the feature
        -   "CN" -- Color-names feature
        */
        enum MODE {
            GRAY = (1 << 0),
            CN = (1 << 1),
            CUSTOM = (1 << 2)
        };

        struct CV_EXPORTS Params {
            /**
            * \brief Constructor
            */
            Params();

            /**
            * \brief Read parameters from file, currently unused
            */
            void read(const FileNode & /*fn*/);

            /**
            * \brief Read parameters from file, currently unused
            */
            void write(FileStorage & /*fs*/) const;

            float detect_thresh;         //!<  detection confidence threshold
            float sigma;                 //!<  gaussian kernel bandwidth
            float lambda;                //!<  regularization
            float interp_factor;         //!<  linear interpolation factor for adaptation
            float output_sigma_factor;   //!<  spatial bandwidth (proportional to target)
            float pca_learning_rate;     //!<  compression learning rate
            bool resize;                  //!<  activate the resize feature to improve the processing speed
            bool split_coeff;             //!<  split the training coefficients into two matrices
            bool wrap_kernel;             //!<  wrap around the kernel values
            bool compress_feature;        //!<  activate the pca method to compress the features
            int max_patch_size;           //!<  threshold for the ROI size
            int compressed_size;          //!<  feature size after compression
            int desc_pca;        //!<  compressed descriptors of TrackerKCF::MODE
            int desc_npca;       //!<  non-compressed descriptors of TrackerKCF::MODE
        };

        virtual void setFeatureExtractor(void(*)(const Mat, const Rect2d&, Mat &), bool pca_func = false) = 0;

        /** @brief Constructor
        @param parameters KCF parameters TrackerKCF::Params
        */
        static Ptr<TrackerKCFX> create(const TrackerKCFX::Params &parameters);

        CV_WRAP static Ptr<TrackerKCFX> create();

        virtual ~TrackerKCFX() {}
    };
}
#endif //OPENCV_TRACKING_TEST_MY_TRACKER_KCF_H
