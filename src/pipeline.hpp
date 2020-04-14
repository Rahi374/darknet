#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <queue>
#include <fstream>
#include <thread>
#include <future>
#include <atomic>
#include <mutex>
#include <cmath>
#include <csignal>
#include <condition_variable>
#include <fstream>

#include "shared_queue.hpp"
#include "yolo_v2_class.hpp"

struct detection_data_t {
	detection_data_t()
		: new_detection(false) {}

	cv::Mat cap_frame;
	std::shared_ptr<image_t> det_image;
	std::vector<bbox_t> result_vec;
	cv::Mat draw_frame;
	bool new_detection;
	uint64_t frame_id;
	std::chrono::steady_clock::time_point time_captured;
};

class Pipeline {

public:
	Pipeline(std::string cfg_file, std::string weights_file, std::string names_file, std::string video_filename, bool show_stream, bool save_output_videofile, float thresh);
	~Pipeline();

	void run();

#ifdef OPENCV
	std::vector<bbox_t> get_3d_coordinates(std::vector<bbox_t> bbox_vect, cv::Mat xyzrgba);
	void draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec, std::vector<std::string> obj_names,
			int current_det_fps = -1, int current_cap_fps = -1);
#endif

	void show_console_result(std::vector<bbox_t> const result_vec, std::vector<std::string> const obj_names, int frame_id = -1);
	std::vector<std::string> objects_names_from_file(std::string const filename);

private:
	bool stop_loop;
	bool display_done;
	std::thread t_cap, t_prepare, t_detect, t_draw, t_write, t_monitor, t_display;
	bool const use_kalman_filter;
	SharedQueue<detection_data_t> q_prepare, q_detect, q_draw, q_write, q_show;
	std::atomic<int> fps_cap_counter, fps_det_counter;
	std::atomic<int> current_fps_cap, current_fps_det;

	Detector detector;

	cv::Size frame_size;
	float thresh;
	track_kalman_t track_kalman;

	cv::VideoCapture cap;
	cv::VideoWriter output_video;

	std::chrono::steady_clock::time_point steady_start, steady_end;
	uint64_t final_frame_id;

	bool show_stream;
	std::vector<std::string> obj_names;

	void capture_thread(cv::VideoCapture &cap);
	void prepare_thread(Detector &detector);
	void detect_thread(Detector &detector);
	void draw_and_track_thread(Detector &detector, std::vector<std::string> &obj_names);
	void write_frame_thread();
	void display_thread();
	void monitoring_thread();
};
