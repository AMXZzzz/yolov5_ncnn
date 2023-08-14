#pragma once
#include<opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <d3d11.h>
#include <dxgi1_2.h>
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")


bool Init_dxgi();
bool Free_dxgi();
bool DXG_Capture_Map();

struct dxgi_info{
	cv::Rect rect;
	UINT output = 0;
	ID3D11Device* device = nullptr;	
	ID3D11DeviceContext* Context = nullptr;	
	IDXGIDevice2* DxgiDevice2 = nullptr;
	IDXGIAdapter* DxgiAdapter = nullptr;
	IDXGIOutput* DxgiOutput = nullptr;
	IDXGIOutput1* DxgiOutput1 = nullptr;


	IDXGIOutputDuplication* desktopDupl = nullptr;
	DXGI_OUTDUPL_FRAME_INFO frameInfo;	
	IDXGIResource* desktopResource = nullptr;	
	ID3D11Texture2D* acquiredDesktopImage = nullptr;	
	D3D11_TEXTURE2D_DESC dataDesc = {NULL};
	ID3D11Texture2D* copyDesktop = nullptr;	
	D3D11_MAPPED_SUBRESOURCE MapResource;	
};