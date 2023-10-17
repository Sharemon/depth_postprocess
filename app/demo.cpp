
#include <opencv2/opencv.hpp>
#include <iostream>

#define FRAME_WIDTH         (1280)
#define FRAME_HEIGHT        (720)
#define MAX_DISPARITY       (160)
#define MATCH_KERNEL_SIZE   (5)


void init_rectify_remap(const std::string yaml_path, cv::Mat* remap_left, cv::Mat* rempa_right)
{
    cv::Size img_size(FRAME_WIDTH, FRAME_HEIGHT);
    cv::FileStorage param_file(yaml_path, cv::FileStorage::READ);

    cv::Mat Kl, Dl, Kr, Dr, R, T;
    param_file["Kl"] >> Kl;
    param_file["Dl"] >> Dl;
    param_file["Kr"] >> Kr;
    param_file["Dr"] >> Dr;
    param_file["R"] >> R;
    param_file["T"] >> T;

    std::cout << "============== read parameters ==================" << std::endl;
    std::cout << "Kl:\n"
        << Kl << std::endl;
    std::cout << "Dl:\n"
        << Dl << std::endl;
    std::cout << "Kr:\n"
        << Kr << std::endl;
    std::cout << "Dr:\n"
        << Dr << std::endl;
    std::cout << "R: \n"
        << R << std::endl;
    std::cout << "T: \n"
        << T << std::endl;

    // 4. stereo rectify
    cv::Mat R1, R2, P1, P2, Q;
    cv::stereoRectify(Kl, Dl, Kr, Dr, img_size, R, T, R1, R2, P1, P2, Q);
    std::cout << "============== rectify parameters ==================" << std::endl;
    std::cout << "P1:\n"
        << P1 << std::endl;
    std::cout << "P2:\n"
        << P2 << std::endl;
    std::cout << "Q: \n"
        << Q << std::endl;

    cv::Mat left_mapx, left_mapy, right_mapx, right_mapy;
    cv::initUndistortRectifyMap(Kl, Dl, R1, P1, img_size, CV_32FC1, remap_left[0], remap_left[1]);
    cv::initUndistortRectifyMap(Kr, Dr, R2, P2, img_size, CV_32FC1, rempa_right[0], rempa_right[1]);
}


int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cout << "Usage: ./depth_postprocess [stereo_paramter_yaml_path]" << std::endl;
        return -1;
    }

    // initialize camera
    cv::VideoCapture cap(0);

    // set camera properties
    cap.set(cv::CAP_PROP_FRAME_WIDTH, FRAME_WIDTH * 2);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);
    cap.set(cv::CAP_PROP_EXPOSURE, -11);

    // read and initialize stereo rectify map
    cv::Mat remap_left[2], remap_right[2];
    init_rectify_remap(std::string(argv[1]), remap_left, remap_right);

    // capture
    cv::Mat frame;
    cv::Mat left_with_speckle, right_with_speckle;
    cv::Mat left_with_texture, right_with_texture;

    while (cap.isOpened())
    {
        // projector on


        // capture
        cap.read(frame);
        left_with_speckle  = frame.colRange(0, FRAME_WIDTH).clone();
        right_with_speckle = frame.colRange(FRAME_WIDTH, FRAME_WIDTH * 2).clone();

        // projector off


        // capture
        cap.read(frame);
        left_with_texture  = frame.colRange(0, FRAME_WIDTH).clone();
        right_with_texture = frame.colRange(FRAME_WIDTH, FRAME_WIDTH * 2).clone();

        // process
        // stereo rectify
        cv::remap(left_with_speckle, left_with_speckle, remap_left[0], remap_left[1], cv::INTER_LINEAR);
        cv::remap(right_with_speckle, right_with_speckle, remap_right[0], remap_right[1], cv::INTER_LINEAR);
        cv::remap(left_with_texture, left_with_texture, remap_left[0], remap_left[1], cv::INTER_LINEAR);
        //cv::remap(right_with_texture, right_with_texture, remap_right[0], remap_right[1], cv::INTER_LINEAR);

        // stereo match
        cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
            0, MAX_DISPARITY, MATCH_KERNEL_SIZE, 
            8 * MATCH_KERNEL_SIZE * MATCH_KERNEL_SIZE, 32 * MATCH_KERNEL_SIZE * MATCH_KERNEL_SIZE, 
            1, 63, 10, 100, 1, cv::StereoSGBM::MODE_HH);

        cv::Mat disp_s16;
        sgbm->compute(left_with_speckle, right_with_speckle, disp_s16);


        // ---------- postprocess -------------
        
        // ---------- postprocess -------------

        // show result
        cv::Mat disp_u8;
        disp_s16.convertTo(disp_u8, CV_8UC1, 1.0 / 16);

        cv::Mat disp_colormap;
        cv::applyColorMap(disp_u8, disp_colormap, cv::COLORMAP_JET);

        cv::Mat merge = disp_colormap * 0.5 + left_with_texture * 0.5;

        cv::imshow("merge", merge);
        cv::waitKey(1);
    }

    cap.release();

    return 0;
}