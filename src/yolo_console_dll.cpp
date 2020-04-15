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

#include "argagg.hpp"
#include "pipeline.hpp"
#include "shared_queue.hpp"
#include "yolo_v2_class.hpp"

int main(int argc, const char *argv[])
{
	argagg::parser argparser {{
		{ "names_file", {"-n", "--names-file"},
			"path to file with class labels (default 'data/coco.names')", 1},
		{ "cfg_file", {"-c", "--cfg-file"},
			"path to darknet config file (default 'cfg/yolov3.cfg')", 1},
		{ "weights_file", {"-w", "--weights-file"},
			"path to weights file (default 'yolov3.weights')", 1},
		{ "thresh", {"-t", "--thresh"},
			"threshold (default 0.2)", 1},
		{ "show_stream", {"-s", "--show-stream"},
			"show stream", 0},
		{ "write_to_video", {"-v", "--write-to-video"},
			"output video", 0},
		{ "show_console", {"-o", "--show-console"},
			"output detections and object ids to console", 0},
		{ "help", {"-h", "--help"},
			"shows this help message", 0},
	}};

	argagg::parser_results args;
	try {
		args = argparser.parse(argc, argv);
	} catch (const std::exception &e) {
		std::cerr << e.what() << std::endl;
		exit(0);
	}

	if (args["help"]) {
		std::cerr << "Usage: " << std::string(argv[0]) << " [options] video ..." << std::endl
			<< argparser;
		exit(0);
	}

	std::string names_file = args["names_file"].as<std::string>("data/coco.names");
	std::string cfg_file = args["cfg_file"].as<std::string>("cfg/yolov3.cfg");
	std::string weights_file = args["weights_file"].as<std::string>("yolov3.weights");
	bool show_stream = args["show_stream"] ? true : false;
	bool write_to_video = args["write_to_video"] ? true : false;
	bool output_to_console = args["show_console"] ? true : false;
	float thresh = args["thresh"].as<float>(0.2);

	if (args.pos.size() == 0) {
		std::cerr << "must provide at least one video file" << std::endl;
		exit(0);
	}

	std::vector<std::thread> pipeline_threads;
	int i = 0;
	for (const char *video_name : args.pos) {
		pipeline_threads.push_back(std::thread([=] {
				Pipeline pipeline(i, cfg_file, weights_file, names_file, std::string(video_name), show_stream, write_to_video, thresh, output_to_console);
				pipeline.run();
				}));
		i++;
	}

	for (std::thread &thread : pipeline_threads)
		if (thread.joinable())
			thread.join();

	return 0;
}
