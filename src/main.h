
#include <ncnn/net.h>
#include<opencv2/opencv.hpp>

class YoloV5Focus : public ncnn::Layer {
public:
	YoloV5Focus()
	{
		one_blob_only = true;
	}

	virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
	{
		int w = bottom_blob.w;
		int h = bottom_blob.h;
		int channels = bottom_blob.c;

		int outw = w / 2;
		int outh = h / 2;
		int outc = channels * 4;

		top_blob.create(outw, outh, outc, 4u, 1, opt.blob_allocator);
		if (top_blob.empty())
			return -100;

#pragma omp parallel for num_threads(opt.num_threads)
		for (int p = 0; p < outc; p++)
		{
			const float* ptr = bottom_blob.channel(p % channels).row((p / channels) % 2) + ((p / channels) / 2);
			float* outptr = top_blob.channel(p);

			for (int i = 0; i < outh; i++)
			{
				for (int j = 0; j < outw; j++)
				{
					*outptr = *ptr;

					outptr += 1;
					ptr += 2;
				}

				ptr += w;
			}
		}

		return 0;
	}
};


//DEFINE_LAYER_CREATOR(YoloV5Focus)	// ע����


struct DXCaptrue_info {

	cv::Mat img;
	int origin_x, origin_y, cx, cy;       //ԭ�㣬���ĵ�
	int window_width, window_height;    //���ڿ��
	bool (*capture_map)();              //capture������ַ
	bool (*capture_free)();             //capture�ͷŵ�ַ
	int capture_width=640, capture_height=640;	//
};

extern struct DXCaptrue_info global_data;      //ȫ�ֽṹ��