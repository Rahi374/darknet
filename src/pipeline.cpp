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

#include "pipeline.hpp"

void Pipeline::capture_thread(cv::VideoCapture &cap)
{
	std::cout << " t_cap start\n";

	uint64_t frame_id = 0;
	do {
		detection_data_t detection_data = detection_data_t();
		cap >> detection_data.cap_frame;
		detection_data.time_captured = std::chrono::steady_clock::now();
		fps_cap_counter++;
		detection_data.frame_id = frame_id++;

		if (detection_data.cap_frame.empty() || stop_loop) {
			std::cout << " exit_flag: detection_data.cap_frame.size = "
				  << detection_data.cap_frame.size()
				  << ", frame_id = " << frame_id << std::endl;
			final_frame_id = frame_id-1;
			stop_loop = true;
			detection_data.cap_frame = cv::Mat(frame_size, CV_8UC3);
		}

		q_prepare.push_back(detection_data);

	} while (!stop_loop);

	std::cout << " t_cap exit\n";
}

void Pipeline::prepare_thread(Detector &detector)
{
	std::cout << " t_prepare start\n";
	detection_data_t detection_data;
	std::shared_ptr<image_t> det_image;
	do {
		detection_data = q_prepare.front();
		q_prepare.pop_front();

		detector.mat_to_image_resize(detection_data);

		q_detect.push_back(detection_data);

	} while (!stop_loop || detection_data.frame_id < final_frame_id);
	std::cout << " t_prepare exit\n";
}

void Pipeline::detect_thread(Detector &detector)
{
	std::cout << " t_detect start\n";
	detection_data_t detection_data;
	std::shared_ptr<image_t> det_image;
	do {
		detection_data = q_detect.front();
		q_detect.pop_front();

		detector.detect_resized(detection_data, thresh);
		fps_det_counter++;

		q_track.push_back(detection_data);

	} while (!stop_loop || detection_data.frame_id < final_frame_id);
	std::cout << " t_detect exit\n";
}

void Pipeline::track_thread(Detector &detector)
{
	std::cout << " t_track start\n";
	detection_data_t detection_data;
	do {
		detection_data = q_track.front();

		q_track.pop_front();
		std::vector<bbox_t> result_vec = detection_data.result_vec;

		detection_data.draw_frame = detection_data.cap_frame.clone();

		result_vec = detector.tracking_id(result_vec, true, std::max(5, current_fps_cap.load()), 40);

		detection_data.result_vec = result_vec;
		q_draw.push_back(detection_data);
	} while (!stop_loop || detection_data.frame_id < final_frame_id);
	std::cout << " t_track exit\n";
}

void Pipeline::draw_thread(std::vector<std::string> &obj_names)
{
	std::cout << " t_draw start\n";
	detection_data_t detection_data;
	do {
		detection_data = q_draw.front();
		q_draw.pop_front();
		std::vector<bbox_t> result_vec = detection_data.result_vec;

		cv::Mat draw_frame = detection_data.draw_frame.clone();

		draw_boxes(draw_frame, detection_data.result_vec, obj_names, current_fps_det, current_fps_cap, detection_data.frame_id);
		if (show_console)
			show_console_result(detection_data.result_vec, obj_names, detection_data.frame_id);

		detection_data.draw_frame = draw_frame;
		detection_data.result_vec = result_vec;
		q_show.push_back(detection_data);

		if (output_video.isOpened())
			q_write.push_back(detection_data);
	} while (!stop_loop || detection_data.frame_id < final_frame_id);
	std::cout << " t_draw exit\n";
}

void Pipeline::write_frame_thread()
{
	std::cout << " t_write start\n";
	detection_data_t detection_data;
	if (output_video.isOpened()) {
		cv::Mat output_frame;
		do {
			detection_data = q_write.front();
			q_write.pop_front();
			if (detection_data.draw_frame.channels() == 4)
				cv::cvtColor(detection_data.draw_frame, output_frame, CV_RGBA2RGB);
			else
				output_frame = detection_data.draw_frame;
			output_video << output_frame;
		} while (!stop_loop || detection_data.frame_id < final_frame_id);
		output_video.release();
	}
	std::cout << " t_write exit\n";
}

void Pipeline::display_thread()
{
	std::cout << " t_display start\n";
	detection_data_t detection_data;
	std::chrono::steady_clock::time_point last_frame_dequeued = std::chrono::steady_clock::now();
	do {
		steady_end = std::chrono::steady_clock::now();
		float time_sec = std::chrono::duration<double>(steady_end - steady_start).count();
		if (time_sec >= 1) {
			current_fps_det = fps_det_counter.load() / time_sec;
			current_fps_cap = fps_cap_counter.load() / time_sec;
			steady_start = steady_end;
			fps_det_counter = 0;
			fps_cap_counter = 0;
		}

		detection_data = q_show.front();
		q_show.pop_front();
		std::chrono::steady_clock::time_point frame_dequeued = std::chrono::steady_clock::now();

		// calculate time taken to get through pipeline
		auto time_frame_in_pipeline = frame_dequeued - detection_data.time_captured;
		auto i_millis = std::chrono::duration_cast<std::chrono::milliseconds>(time_frame_in_pipeline);
		float time_frame_in_pipeline_ms = i_millis.count();

		// calculate fps per-frame
		auto frame_interval = frame_dequeued - last_frame_dequeued;
		auto f_millis = std::chrono::duration_cast<std::chrono::milliseconds>(frame_interval);
		float fps = f_millis.count() ? (1000/f_millis.count()) : FLT_MAX;

		if (show_stream) {
			cv::Mat draw_frame = detection_data.draw_frame;
			cv::imshow("window name", draw_frame);
			cv::waitKey(1);
		}

		std::cout << " thread_id " << thread_id << " frame " << detection_data.frame_id
			  << " latency (ms) = " << time_frame_in_pipeline_ms
			  << " current_fps_det = " << current_fps_det
			  << " current_fps_cap = " << current_fps_cap << std::endl;
		last_frame_dequeued = frame_dequeued;
	} while (!stop_loop || detection_data.frame_id < final_frame_id);
	is_running = false;
	display_done = true;
	std::cout << " t_display exit\n";
}

void Pipeline::monitoring_thread()
{
	std::cout << " monitor start\n";
	std::ofstream outfile;
	outfile.open("queue-hist.log");

	do {
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
		outfile << "cap-prep: " << q_prepare.counted_size()
			<< " prep-detect: " << q_detect.counted_size()
			<< " detect-track: " << q_track.counted_size()
			<< " track-draw: " << q_draw.counted_size()
			//<< " draw-write: " << q_write.counted_size()
			<< " draw-show: " << q_show.counted_size()
			<< std::endl;
	} while (!stop_loop || !display_done);
	std::cout << " monitor exit\n";
}

Pipeline::Pipeline(int thread_id, std::string cfg_file, std::string weights_file, std::string names_file, std::string video_filename, bool show_stream, bool save_output_videofile, float thresh, bool output_to_console)
	: thread_id(thread_id), stop_loop(false), display_done(false), final_frame_id(UINT_MAX), fps_cap_counter(0), fps_det_counter(0), current_fps_cap(0), current_fps_det(0), detector(cfg_file, weights_file), show_stream(show_stream), thresh(thresh), is_running(true), show_console(output_to_console)
{
	std::cout << "CONSTRUCTING PIPELINE" << std::endl;
	obj_names = objects_names_from_file(names_file);
	std::string out_videofile = "result_" + (video_filename.find("/") == std::string::npos ? video_filename : video_filename.substr(video_filename.rfind("/") + 1)) + ".avi";
	std::cout << "OUTPUT VIDEO FILE IS " << out_videofile << std::endl;

	// init for the detection loop
	cv::Mat cur_frame;
	cap.open(video_filename);
	cap >> cur_frame;

	int video_fps = 25;

	video_fps = cap.get(cv::CAP_PROP_FPS);
	frame_size = cur_frame.size();
	//cv::Size const frame_size(cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT));
	std::cout << "\n Video size: " << frame_size << std::endl;

	if (save_output_videofile) {
		std::cout << "WE GONNA SAVE SOME VIDEO" << std::endl;
		output_video.open(out_videofile, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), std::max(35, video_fps), frame_size, true);
	}
	std::cout << "DONE CONSTRUCTING PIPELINE" << std::endl;
}

Pipeline::~Pipeline()
{
	// wait for all threads
	if (t_cap.joinable())
		t_cap.join();
	if (t_prepare.joinable())
		t_prepare.join();
	if (t_detect.joinable())
		t_detect.join();
	if (t_track.joinable())
		t_track.join();
	if (t_draw.joinable())
		t_draw.join();
	if (t_write.joinable())
		t_write.join();
	if (t_display.joinable())
		t_display.join();
	if (t_monitor.joinable())
		t_monitor.join();
	if (show_stream)
		cv::destroyWindow("window name");
}

void Pipeline::run()
{
	std::cout << "WE BE RUNNING!" << std::endl;
	t_monitor = std::thread([=] {monitoring_thread();});

	// capture new video-frame
	t_cap = std::thread([=] {capture_thread(cap);});

	// pre-processing video frame (resize, convertion)
	t_prepare = std::thread([=] {prepare_thread(detector);});

	// detection by Yolo
	t_detect = std::thread([=] {detect_thread(detector);});

	// track objects
	t_track = std::thread([=] {track_thread(detector);});

	// draw rectangles
	t_draw = std::thread([=] {draw_thread(obj_names);});

	// write frame to videofile
	t_write = std::thread([=] {write_frame_thread();});

	// show detection
	t_display = std::thread([=] {display_thread();});
}

#ifdef OPENCV
std::vector<bbox_t> Pipeline::get_3d_coordinates(std::vector<bbox_t> bbox_vect, cv::Mat xyzrgba)
{
	return bbox_vect;
}

#include <opencv2/opencv.hpp> // C++
#include <opencv2/core/version.hpp>
#ifndef CV_VERSION_EPOCH // OpenCV 3.x and 4.x
#include <opencv2/videoio/videoio.hpp>
#define OPENCV_VERSION CVAUX_STR(CV_VERSION_MAJOR) \
"" CVAUX_STR(CV_VERSION_MINOR) "" CVAUX_STR(CV_VERSION_REVISION)
#ifndef USE_CMAKE_LIBS
#pragma comment(lib, "opencv_world" OPENCV_VERSION ".lib")
#endif // USE_CMAKE_LIBS
#else // OpenCV 2.x
#define OPENCV_VERSION CVAUX_STR(CV_VERSION_EPOCH) \
"" CVAUX_STR(CV_VERSION_MAJOR) "" CVAUX_STR(CV_VERSION_MINOR)
#ifndef USE_CMAKE_LIBS
#pragma comment(lib, "opencv_core" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_imgproc" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_highgui" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_video" OPENCV_VERSION ".lib")
#endif // USE_CMAKE_LIBS
#endif // CV_VERSION_EPOCH

void Pipeline::draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec, std::vector<std::string> obj_names,
		int current_det_fps, int current_cap_fps, uint64_t frame_id)
{
	// int const colors[6][3] = { { 1, 0, 1 }, { 0, 0, 1 }, { 0, 1, 1 }, { 0, 1, 0 }, { 1, 1, 0 }, { 1, 0, 0 } };

	for (auto &i : result_vec) {
		cv::Scalar color = obj_id_to_color(i.obj_id);
		cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 2);
		if (obj_names.size() > i.obj_id) {
			std::string obj_name = obj_names[i.obj_id];
			if (i.track_id > 0)
				obj_name += " - " + std::to_string(i.track_id);
			cv::Size const text_size = getTextSize(obj_name, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2, 0);
			int max_width = ((unsigned int)text_size.width > i.w + 2) ? text_size.width : (i.w + 2);
			max_width = std::max(max_width, (int)i.w + 2);
			//max_width = std::max(max_width, 283);
			std::string coords_3d;
			if (!std::isnan(i.z_3d)) {
				std::stringstream ss;
				ss << std::fixed << std::setprecision(2) << "x:" << i.x << " y:" << i.y << " p:" << i.prob;
				coords_3d = ss.str();
				cv::Size const text_size_3d = getTextSize(ss.str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, 1, 0);
				int const max_width_3d = ((unsigned int)text_size_3d.width > i.w + 2) ? text_size_3d.width : (i.w + 2);
				if (max_width_3d > max_width)
					max_width = max_width_3d;
			}

			cv::rectangle(mat_img, cv::Point2f(std::max((int)i.x - 1, 0), std::max((int)i.y - 35, 0)),
				      cv::Point2f(std::min((int)i.x + max_width, mat_img.cols - 1), std::min((int)i.y, mat_img.rows - 1)),
				      color, CV_FILLED, 8, 0);
			putText(mat_img, obj_name, cv::Point2f(i.x, i.y - 16), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(0, 0, 0), 2);
			if (!coords_3d.empty())
				putText(mat_img, coords_3d, cv::Point2f(i.x, i.y - 1), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0, 0, 0), 1);
		}
	}

	if (current_det_fps >= 0 && current_cap_fps >= 0) {
		std::string fps_str = "FPS detection: " + std::to_string(current_det_fps) + "   FPS capture: " + std::to_string(current_cap_fps);
		putText(mat_img, fps_str, cv::Point2f(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(50, 255, 0), 2);
	}

	putText(mat_img, "frame " + std::to_string(frame_id), cv::Point2f(10, 50), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(50, 255, 0), 2);
}
#endif // OPENCV

void Pipeline::show_console_result(std::vector<bbox_t> const result_vec, std::vector<std::string> const obj_names, int frame_id)
{
	for (auto &i : result_vec) {
		std::cout << "Entry f " << frame_id << " ";

		if (obj_names.size() > i.obj_id)
			std::cout << obj_names[i.obj_id] << " - ";
		else
			std::cout << "unknown" << " - ";

		std::cout << "\tobj_id = " << i.obj_id
			  << "\ttrack_id = " << std::setw(3) << std::setfill('0') << i.track_id
			  << "\tx = " << i.x << "   \ty = " << i.y
			  << "  \tw = " << i.w << "  \th = " << i.h
			  << std::setprecision(3) << "  \tprob = " << i.prob << std::endl;
	}
}

std::vector<std::string> Pipeline::objects_names_from_file(std::string const filename)
{
	std::ifstream file(filename);
	std::vector<std::string> file_lines;
	if (!file.is_open())
		return file_lines;
	for (std::string line; getline(file, line);)
		file_lines.push_back(line);
	std::cout << "object names loaded \n";
	return file_lines;
}
