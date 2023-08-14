
#include"main.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <float.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <chrono>
#include <ncnn/layer.h>
#include <fstream>
#include "dxgi-capture.h"

/*



*/
#define INPUT_SIZE 640		// 模型的输入大小
#define CONF 0.35			// 置信度
#define IOU 0.15f			// 置信度
#define MAX_STRIDE 32		// or 64


//当前全局变量
static std::string feature_layer_1 = "output";
static std::string feature_layer_2 = "336";
static std::string feature_layer_3 = "340";
static int temp_str = 0;
int img_h, img_w, w_pad, h_pad;
float scale = 1.f;

struct DXCaptrue_info global_data;

struct Object{
	cv::Rect_<float> rect;
	int label;
	float prob;
};

static inline float intersection_area(const Object& a, const Object& b) {
	cv::Rect_<float> inter = a.rect & b.rect;
	return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right) {
	int i = left;
	int j = right;
	float p = faceobjects[(static_cast<std::vector<Object, std::allocator<Object>>::size_type>(left) + right) / 2].prob;

	while (i <= j) {
		while (faceobjects[i].prob > p)
			i++;

		while (faceobjects[j].prob < p)
			j--;
		if (i <= j) {
			//交换排序
			std::swap(faceobjects[i], faceobjects[j]);
			i++;
			j--;
		}
	}

#pragma omp parallel sections	// 并行计算
	{
#pragma omp section
		{
			if (left < j) qsort_descent_inplace(faceobjects, left, j);
		}
#pragma omp section
		{
			if (i < right) qsort_descent_inplace(faceobjects, i, right);
		}
	}
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects) {
	if (faceobjects.empty())
		return;
	qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold, bool agnostic = true) {
	picked.clear();

	const int n = faceobjects.size();

	std::vector<float> areas(n);
	for (int i = 0; i < n; i++) {
		areas[i] = faceobjects[i].rect.area();
	}

	for (int i = 0; i < n; i++) {
		const Object& a = faceobjects[i];

		int keep = 1;
		for (int j = 0; j < (int)picked.size(); j++) {
			const Object& b = faceobjects[picked[j]];

			if (!agnostic && a.label != b.label)
				continue;

			// iou 排序
			float inter_area = intersection_area(a, b);
			float union_area = areas[i] + areas[picked[j]] - inter_area;
			// float IoU = inter_area / union_area
			if (inter_area / union_area > nms_threshold)
				keep = 0;
		}

		if (keep)
			picked.push_back(i);
	}
}

static inline float sigmoid(float x) {
	return static_cast<float>(1.f / (1.f + exp(-x)));
}

static void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects) {
	const int num_grid = feat_blob.h;

	int num_grid_x;
	int num_grid_y;
	if (in_pad.w > in_pad.h)
	{
		num_grid_x = in_pad.w / stride;
		num_grid_y = num_grid / num_grid_x;
	}
	else
	{
		num_grid_y = in_pad.h / stride;
		num_grid_x = num_grid / num_grid_y;
	}

	const int num_class = feat_blob.w - 5;

	const int num_anchors = anchors.w / 2;

	for (int q = 0; q < num_anchors; q++)
	{
		const float anchor_w = anchors[static_cast<size_t>(q) * 2];
		const float anchor_h = anchors[static_cast<size_t>(q) * 2 + 1];

		const ncnn::Mat feat = feat_blob.channel(q);

		for (int i = 0; i < num_grid_y; i++)
		{
			for (int j = 0; j < num_grid_x; j++)
			{
				const float* featptr = feat.row(i * num_grid_x + j);
				float box_confidence = sigmoid(featptr[4]);
				if (box_confidence >= prob_threshold)
				{
					// find class index with max class score
					int class_index = 0;
					float class_score = -FLT_MAX;
					for (int k = 0; k < num_class; k++)
					{
						float score = featptr[5 + k];
						if (score > class_score)
						{
							class_index = k;
							class_score = score;
						}
					}
					float confidence = box_confidence * sigmoid(class_score);
					if (confidence >= prob_threshold)
					{
						// yolov5/models/yolo.py Detect forward
						// y = x[i].sigmoid()
						// y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
						// y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

						float dx = sigmoid(featptr[0]);
						float dy = sigmoid(featptr[1]);
						float dw = sigmoid(featptr[2]);
						float dh = sigmoid(featptr[3]);

						float pb_cx = (dx * 2.f - 0.5f + j) * stride;
						float pb_cy = (dy * 2.f - 0.5f + i) * stride;

						float pb_w = pow(dw * 2.f, 2) * anchor_w;
						float pb_h = pow(dh * 2.f, 2) * anchor_h;

						float x0 = pb_cx - pb_w * 0.5f;
						float y0 = pb_cy - pb_h * 0.5f;
						float x1 = pb_cx + pb_w * 0.5f;
						float y1 = pb_cy + pb_h * 0.5f;

						Object obj;
						obj.rect.x = x0;
						obj.rect.y = y0;
						obj.rect.width = x1 - x0;
						obj.rect.height = y1 - y0;
						obj.label = class_index;
						obj.prob = confidence;

						objects.push_back(obj);
					}
				}
			}
		}
	}
}

static void S_Draw(const cv::Mat& image, const std::vector<Object>& objects) {
	for (size_t i = 0; i < objects.size(); i++) {
		cv::rectangle(image, objects[i].rect, cv::Scalar(0, 255, 0), 2);
	}
	cv::imshow("image", image);
	cv::waitKey(1);
}

static void S_Get_Output_Layer_Name(std::string temp) {
	int idx = 0;
	idx = temp.find("0=1");
	if (idx != std::string::npos) {				//找到0=1
		temp = temp.substr(0, idx);
		idx = temp.find_last_of(" ");
		if (idx != std::string::npos) {			//找到空格
			temp = temp.substr(0, idx);
			idx = temp.find_last_of(" ");
			if (idx != std::string::npos) {			//找到空格
				temp = temp.substr(idx + strlen(" "));
				if (temp == "output") {
					feature_layer_1 = "output";
				}
				else {
					int temp_int = std::stoi(temp);
					if ((temp_int / 100 % 10) == 3) {
						if (temp_str > 300) {

							if (temp_int > temp_str)
								feature_layer_2 = std::to_string(temp_str);
							feature_layer_3 = std::to_string(temp_int);
						}
						else temp_str = temp_int;
					}
				}
			}
		}
	}
}

static bool S_Find_Output_Layer(const char* pathname) {
	FILE* fileline = fopen(pathname, "r+");	// 打开文件
	if (fileline == NULL) {
		std::cout << "打开param模型错误" << std::endl;
		return FALSE;
	}

	struct stat filestate;		// 文件状态结构体
	stat(pathname, &filestate);	// 获取文件状态
	char* filebuffer = new char[sizeof(char) * (static_cast<unsigned long long>(filestate.st_size) + 1)];	// 创建内存大小
	for (int i = 0, ch = 0; ch != EOF; i++) {		// EOF表示文件末尾 -1
		ch = fgetc(fileline);			// 读取一个字符串，后移1位 字符串的行索引
		*(filebuffer + i) = ch;		// 将字符串写入流
	}
	std::string test = filebuffer;
	int idx = 0;
	while (idx != std::string::npos) {
		idx = test.find("Permute");
		if (idx != std::string::npos) {
			test = test.substr(idx + strlen("Permute"));
			int n_idx = test.find("\n");
			if (n_idx != std::string::npos)
				S_Get_Output_Layer_Name(test.substr(0, n_idx));
		}
	}
	return TRUE;
}

static bool Subtit(const char* s1, const char* s2, const char* pathname) {
	int len_s1 = strlen(s1);		// s1的长度
	int len_s2 = strlen(s2);		// s2的长度
	FILE* fileline = fopen(pathname, "r+");	// 打开文件
	if (fileline == NULL) {
		std::cout << "打开param模型错误" << std::endl;
		return FALSE;
	}
	struct stat filestate;		// 文件状态结构体
	stat(pathname, &filestate);	// 获取文件状态
	char* filebuffer = new char[sizeof(char) * (static_cast<unsigned long long>(filestate.st_size) + 1)];	// 创建内存大小

	for (int i = 0, ch = 0; ch != EOF; i++) {		// EOF表示文件末尾 -1
		ch = fgetc(fileline);			// 读取一个字符串，后移1位 字符串的行索引
		//filebuffer[i] = ch;				
		*(filebuffer + i) = ch;		// 将字符串写入流
	}

	fseek(fileline, 0, SEEK_SET);		// 移动到文件开头 0
	char* index_1 = filebuffer, * index_2 = filebuffer;
	while (true) {
		// 查找Permute
		index_2 = strstr(index_1, s1);	// 按s1位匹配字符串，返回位置
		if (index_2) {
			for (int i = 0; i < index_2 - index_1; i++)
				fputc(index_1[i], fileline);
			for (int i = 0; i < len_s2; i++)
				fputc(s2[i], fileline);
			index_1 = index_2 + len_s1;
		}
		else {
			// 写入流
			while (*index_1 != EOF)
				fputc(*index_1++, fileline);
			break;
		}
	}
	// free
	delete[]filebuffer;
	std::fclose(fileline);
	return TRUE;
}

static bool S_Modify_Model_Reshape(const char* model_path) {
	bool hr = FALSE;
	hr = Subtit("0=6400", "0=-1", model_path);
	if (!hr)
		return FALSE;
	hr = Subtit("0=1600", "0=-1", model_path);
	if (!hr)
		return FALSE;
	hr = Subtit("0=400", "0=-1", model_path);
	if (!hr)
		return FALSE;
	return TRUE;
}

static bool S_Load_Ncnn_Model(ncnn::Net& yolov5, const char* param_path, const char* bin_path) {
	// 设置yolov5对象属性
	yolov5.opt.use_vulkan_compute = true;		// 使用vulkan推理
	yolov5.opt.use_fp16_arithmetic = true;		//fp16运算加速


	// 加载模型
	if (yolov5.load_param(param_path)) {
		std::cout << "load_param Error!" << std::endl;
		return FALSE;
	}

	if (yolov5.load_model(bin_path)) {
		std::cout << "load_bin Error!" << std::endl;
		return FALSE;
	}
	return TRUE;
}

static bool S_Init_Param(const char* param_path, const char* bin_path) {
	if (!S_Modify_Model_Reshape(param_path))
		return FALSE;
	if (!S_Find_Output_Layer(param_path))
		return FALSE;
	return TRUE;
}

static void S_Mat_Preproces(cv::Mat& img, ncnn::Mat& input) {
	img_w = img.cols;
	img_h = img.rows;

	int w = img_w;
	int h = img_h;
	if (w > h)
	{
		scale = (float)INPUT_SIZE / w;
		w = INPUT_SIZE;
		h = h * scale;
	}
	else {
		scale = (float)INPUT_SIZE / h;
		h = INPUT_SIZE;
		w = w * scale;
	}

	ncnn::Mat in = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

	w_pad = (w + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
	h_pad = (h + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;

	ncnn::copy_make_border(in, input, h_pad / 2, h_pad - h_pad / 2, w_pad / 2, w_pad - w_pad / 2, ncnn::BORDER_CONSTANT, 114.f);

	const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
	input.substract_mean_normalize(0, norm_vals);
}

static void S_Detect(ncnn::Net& yolov5, ncnn::Mat& input, std::vector<Object>& proposals) {
	ncnn::Extractor ex = yolov5.create_extractor();
	// 输入
	ex.input("images", input);

	ncnn::Mat anchors8(6);
	anchors8[0] = 10.f;
	anchors8[1] = 13.f;
	anchors8[2] = 16.f;
	anchors8[3] = 30.f;
	anchors8[4] = 33.f;
	anchors8[5] = 23.f;

	ncnn::Mat anchors16(6);
	anchors16[0] = 30.f;
	anchors16[1] = 61.f;
	anchors16[2] = 62.f;
	anchors16[3] = 45.f;
	anchors16[4] = 59.f;
	anchors16[5] = 119.f;

	ncnn::Mat anchors32(6);
	anchors32[0] = 116.f;
	anchors32[1] = 90.f;
	anchors32[2] = 156.f;
	anchors32[3] = 198.f;
	anchors32[4] = 373.f;
	anchors32[5] = 326.f;


	// stride 8
	ncnn::Mat out8;
	std::vector<Object> objects8;
	ex.extract(feature_layer_1.c_str(), out8);	// 第一个有效特征层，通过层名称获取
	generate_proposals(anchors8, 8, input, out8, CONF, objects8);
	proposals.insert(proposals.end(), objects8.begin(), objects8.end());

	// stride 16
	ncnn::Mat out16;
	std::vector<Object> objects16;
	ex.extract(feature_layer_2.c_str(), out16);
	generate_proposals(anchors16, 16, input, out16, CONF, objects16);
	proposals.insert(proposals.end(), objects16.begin(), objects16.end());

	// stride 32
	ncnn::Mat out32;
	std::vector<Object> objects32;
	ex.extract(feature_layer_3.c_str(), out32);
	generate_proposals(anchors32, 32, input, out32, CONF, objects32);
	proposals.insert(proposals.end(), objects32.begin(), objects32.end());
}

static void S_Post_Proces(std::vector<Object>& proposals, std::vector<Object>& objects) {

	// 按置信度从高到低对所有先验框进行排序
	qsort_descent_inplace(proposals);

	// nms
	std::vector<int> picked;
	nms_sorted_bboxes(proposals, picked, IOU);

	int count = picked.size();

	objects.resize(count);
	for (int i = 0; i < count; i++) {
		objects[i] = proposals[picked[i]];

		// 调整原始未填充的偏移量
		float x0 = (objects[i].rect.x - (w_pad * 0.5)) / scale;
		float y0 = (objects[i].rect.y - (h_pad * 0.5)) / scale;
		float x1 = (objects[i].rect.x + objects[i].rect.width - (w_pad * 0.5)) / scale;
		float y1 = (objects[i].rect.y + objects[i].rect.height - (h_pad * 0.5)) / scale;

		// clip
		x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
		y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
		x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
		y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

		objects[i].rect.x = x0;
		objects[i].rect.y = y0;
		objects[i].rect.width = x1 - x0;
		objects[i].rect.height = y1 - y0;
	}
}

int main(int argc, char** argv) {

	// 初始变量
	const char* param_path = "ncnn.param";
	const char* bin_path = "ncnn.bin";
	const char* image_path = "C:\\Users\\Zzzz\\Desktop\\000013.jpg";
	ncnn::Net yolov5;		// ncnn对象

	// 初始化
	if (!Init_dxgi())
		return FALSE;

	//初始化模型
	if (!S_Init_Param(param_path, bin_path))
		return FALSE;

	//加载模型
	if (!S_Load_Ncnn_Model(yolov5, param_path, bin_path))
		return FALSE;

	std::vector<Object> proposals;
	std::vector<Object> tager_array;

	while (true) {
		std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
		DXG_Capture_Map();

		// 图片预处理
		ncnn::Mat input;
		S_Mat_Preproces(global_data.img, input);

		// 运行推理
		proposals.clear();
		S_Detect(yolov5, input, proposals);

		// 后处理
		tager_array.clear();
		S_Post_Proces(proposals, tager_array);

		std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
		std::cout << "while: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms\t" << std::endl;

		// 画框
		S_Draw(global_data.img, tager_array);
	}

	return 0;
}