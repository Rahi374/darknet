#ifndef PIPELINE_HPP
#define PIPELINE_HPP

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

#include "dark_cuda.h"
#include "shared_queue.hpp"
#include "yolo_v2_class.hpp"

class Pipeline {

public:
	Pipeline(int thread_id, std::string cfg_file, std::string weights_file, std::string names_file, std::string video_filename, bool show_stream, bool save_output_videofile, float thresh, bool output_to_console);
	~Pipeline();

	void run();

#ifdef OPENCV
	std::vector<bbox_t> get_3d_coordinates(std::vector<bbox_t> bbox_vect, cv::Mat xyzrgba);
	void draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec, std::vector<std::string> obj_names,
			int current_det_fps, int current_cap_fps, uint64_t frame_id);
#endif

	void show_console_result(std::vector<bbox_t> const result_vec, std::vector<std::string> const obj_names, int frame_id = -1);
	std::vector<std::string> objects_names_from_file(std::string const filename);

private:
	int thread_id;

	bool is_running;
	bool stop_loop;
	bool display_done;
	std::thread t_cap, t_prepare, t_detect, t_track, t_draw, t_write, t_monitor, t_display;
	SharedQueue<detection_data_t> q_prepare, q_detect, q_track, q_draw, q_write, q_show;
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
	bool show_console;
	std::vector<std::string> obj_names;

	void capture_thread(cv::VideoCapture &cap);
	void prepare_thread(Detector &detector);
	void detect_thread(Detector &detector);
	void track_thread(Detector &detector);
	void draw_thread(std::vector<std::string> &obj_names);
	void write_frame_thread();
	void display_thread();
	void monitoring_thread();
};

#endif // PIPELINE_HPP
