#include "main.h"
#include"dxgi-capture.h"

struct dxgi_info dxgi_data;
static bool Desc_Should_Init = false;
static inline bool Init_Device() {
	D3D_FEATURE_LEVEL FeatureLevel;
	HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE,
		nullptr, 0, NULL, NULL, D3D11_SDK_VERSION,
		&dxgi_data.device, &FeatureLevel, &dxgi_data.Context);
	if (FAILED(hr))
		return false;
	return true;
}
static inline bool GetDevice2() {
	HRESULT hr = dxgi_data.device->QueryInterface(__uuidof(IDXGIDevice2), (void**)&dxgi_data.DxgiDevice2);
	if (FAILED(hr))
		return false;
	dxgi_data.DxgiDevice2->SetMaximumFrameLatency(10);
	return true;
}
static inline bool Get_DXGI_Adapter() {
	HRESULT hr = dxgi_data.DxgiDevice2->GetParent(__uuidof(IDXGIAdapter), reinterpret_cast<void**>(&dxgi_data.DxgiAdapter));
	if (FAILED(hr))
		return false;
	return true;
}
static inline bool Get_DxgiOutput() {
	HRESULT hr = dxgi_data.DxgiAdapter->EnumOutputs(dxgi_data.output, &dxgi_data.DxgiOutput);
	if (FAILED(hr))
		return false;
	return true;
}
static inline bool Get_Output1_Interface() {
	HRESULT hr = dxgi_data.DxgiOutput->QueryInterface(__uuidof(dxgi_data.DxgiOutput1), reinterpret_cast<void**>(&dxgi_data.DxgiOutput1));	//查询接口
	if (FAILED(hr))
		return false;
	return true;
}

static inline bool Get_desk_Duplicate() {
	HRESULT hr = dxgi_data.DxgiOutput1->DuplicateOutput(dxgi_data.device, &dxgi_data.desktopDupl);
	if (FAILED(hr))
		return false;
	return true;
}

static inline bool set_global_data() {
	global_data.window_width = int(dxgi_data.dataDesc.Width);
	global_data.window_height = int(dxgi_data.dataDesc.Height);
	global_data.cx = int(dxgi_data.dataDesc.Width * 0.5f);
	global_data.cy = int(dxgi_data.dataDesc.Height * 0.5f);
	global_data.origin_x = global_data.cx - int(global_data.capture_width * 0.5f);
	global_data.origin_y = global_data.cy - int(global_data.capture_height * 0.5f);
	dxgi_data.rect = cv::Rect(global_data.origin_x, global_data.origin_y, global_data.capture_width, global_data.capture_height);
	return true;
}

static inline bool Get_Frame_prt() {
	HRESULT hr = dxgi_data.desktopResource->QueryInterface(__uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&dxgi_data.acquiredDesktopImage));
	if (FAILED(hr))
		return false;
	return true;
}

static bool Init_Desc() {
	dxgi_data.acquiredDesktopImage->GetDesc(&dxgi_data.dataDesc);
	dxgi_data.dataDesc.Usage = D3D11_USAGE_STAGING;
	dxgi_data.dataDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
	dxgi_data.dataDesc.BindFlags = 0;
	dxgi_data.dataDesc.MiscFlags = 0;
	dxgi_data.dataDesc.MipLevels = 1;
	dxgi_data.dataDesc.SampleDesc.Count = 1;
	return true;
}

static inline bool Create_Tex() {
	dxgi_data.device->CreateTexture2D(&dxgi_data.dataDesc, NULL, &dxgi_data.copyDesktop);
	if (dxgi_data.copyDesktop == NULL)
		return false;
	return true;
}

bool DXG_Capture_Map() {

	HRESULT hr = dxgi_data.desktopDupl->AcquireNextFrame(15, &dxgi_data.frameInfo, &dxgi_data.desktopResource);
	if (!Get_Frame_prt())
		return false;
	if (Desc_Should_Init) {
		if (Init_Desc())
			Desc_Should_Init = false;
	}
	if (!Create_Tex())
		return false;
	dxgi_data.Context->CopyResource(dxgi_data.copyDesktop, dxgi_data.acquiredDesktopImage);
	dxgi_data.Context->Map(dxgi_data.copyDesktop, 0, D3D11_MAP_READ, 0, &dxgi_data.MapResource);
	global_data.img = cv::Mat(int(dxgi_data.dataDesc.Height), int(dxgi_data.dataDesc.Width), CV_8UC4, dxgi_data.MapResource.pData)(dxgi_data.rect);
	cv::cvtColor(global_data.img, global_data.img, cv::COLOR_BGRA2BGR);
	dxgi_data.desktopDupl->ReleaseFrame();
	return true;
}

static inline bool DXGI_Screenshot_size() {
	HRESULT hr = dxgi_data.desktopDupl->AcquireNextFrame(0, &dxgi_data.frameInfo, &dxgi_data.desktopResource);
	if (Get_Frame_prt())
		dxgi_data.acquiredDesktopImage->Release();
	if (!Init_Desc())
		return true;
	set_global_data();
	return true;
}

bool Free_dxgi() {
	if (dxgi_data.device) {
		dxgi_data.device->Release();	//释放
		dxgi_data.device = nullptr;	//防野
	}
	if (dxgi_data.Context) {
		dxgi_data.Context->Release();
		dxgi_data.Context = nullptr;
	}

	if (dxgi_data.DxgiDevice2) {
		dxgi_data.DxgiDevice2->Release();	//释放
		dxgi_data.DxgiDevice2 = nullptr;	//防野
	}
	if (dxgi_data.DxgiDevice2) {
		dxgi_data.DxgiDevice2->Release();	//释放
		dxgi_data.DxgiDevice2 = nullptr;	//防野
	}
	if (dxgi_data.DxgiAdapter) {
		dxgi_data.DxgiAdapter->Release();
		dxgi_data.DxgiAdapter = nullptr;
	}
	if (dxgi_data.DxgiOutput) {
		dxgi_data.DxgiOutput->Release();
		dxgi_data.DxgiOutput = nullptr;
	}
	if (dxgi_data.DxgiOutput1) {
		dxgi_data.DxgiOutput1->Release();
		dxgi_data.DxgiOutput1 = nullptr;
	}
	if (dxgi_data.desktopDupl) {
		dxgi_data.desktopDupl->Release();
		dxgi_data.desktopDupl = nullptr;
	}
	if (dxgi_data.desktopResource) {
		dxgi_data.desktopResource->Release();
		dxgi_data.desktopResource = nullptr;
	}

	if (dxgi_data.acquiredDesktopImage) {
		dxgi_data.acquiredDesktopImage->Release();
		dxgi_data.acquiredDesktopImage = nullptr;
	}

	if (dxgi_data.acquiredDesktopImage) {
		dxgi_data.acquiredDesktopImage->Release();
		dxgi_data.acquiredDesktopImage = nullptr;
	}
	return true;
}

bool Init_dxgi() {
	//创建d3d11设备
	if (!Init_Device()) {
		return false;
	}
	if (!GetDevice2()) {
		return false;
	}
	if (!Get_DXGI_Adapter()) {
		return false;
	}
	if (!Get_DxgiOutput()) {
		return false;
	}
	if (!Get_Output1_Interface()) {
		return false;
	}
	if (!Get_desk_Duplicate()) {
		return false;
	}
	if (!DXGI_Screenshot_size()) {
		return false;
	}
	Desc_Should_Init = true;
	return true;
}

