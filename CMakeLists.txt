cmake_minimum_required(VERSION 3.18)			# cmake 最低版本，需要3.18版本以上的新特性

project(SF_NCNN_Project) 				# 项目名称

# -- 改成自己的路径
set(OpenCV_DIR "C:\\Program Files\\opencv\\build")   	# 你自己的opecv根路径

# -- 设置标准
add_definitions(-DAPI_EXPORTS)			# 这行解决警告
set(CMAKE_CXX_STANDARD 14)			# 设置cpp版本
set(CMAKE_BUILD_TYPE  Release)			# 设置debug or Release

# -- 项目设置
include_directories(${PROJECT_SOURCE_DIR}/include)	# 指定项目的include目录
LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/lib)		# 指定项目的lib路径

# -- 设置 opencv
message(STATUS "OpenCV_DIR = ${OpenCV_DIR}")
find_package(OpenCV REQUIRED)				# 引入opecv库
include_directories(${OpenCV_INCLUDE_DIRS})		# opencv  的 include目录

# 设置项目头文件
file(GLOB_RECURSE CURRENT_HEADERS ${PROJECT_SOURCE_DIR}/src/*.h)
source_group("Include" FILES ${CURRENT_HEADERS}) 

# 设置cpp源文件文件
file(GLOB_RECURSE CPP_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp)

# -- 指定生成的文件 
add_executable(SF_NCNN ${CPP_FILES} ${CURRENT_HEADERS})

# -- 链接依赖
target_link_libraries(SF_NCNN "ncnn.lib" "libprotobuf.lib" )  	# 链接lib
target_link_libraries(SF_NCNN "vulkan-1.lib" "GenericCodeGen.lib" "SPIRV.lib" "OSDependent.lib" "OGLCompiler.lib" "MachineIndependent.lib" "glslang.lib")  	# 链接lib
target_link_libraries(SF_NCNN ${OpenCV_LIBS})     		# 链接opencv库

