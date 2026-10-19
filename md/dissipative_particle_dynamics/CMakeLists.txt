# DPD算子项目 - 顶层CMakeLists.txt
# 遵循CANN算子开发规范，适配Ascend 910/310P

cmake_minimum_required(VERSION 3.14)
project(dpdkernel VERSION 1.0.0 LANGUAGES C CXX)

# 设置CANN路径
set(CANN_PATH $ENV{ASCEND_TOOLKIT_HOME})
if(NOT CANN_PATH)
    message(FATAL_ERROR "请设置ASCEND_TOOLKIT_HOME环境变量")
endif()

# CANN 头文件与库目录（与 velocity_verlet/PME 保持一致：位于 <CANN>/aarch64-linux 下）
set(CANN_INCLUDE_DIR ${CANN_PATH}/aarch64-linux/include)
set(CANN_LIB_DIR ${CANN_PATH}/aarch64-linux/lib64)

# 设置编译选项
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 调试/发布配置
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

# 编译器选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 -DNDEBUG")

# 单元测试默认关闭：产品编译不依赖 GTest，UT 需显式 -DBUILD_TESTS=ON 开启
option(BUILD_TESTS "构建单元测试" OFF)

# 添加子目录
# add_subdirectory(op_kernel)  # 暂时跳过内核编译
add_subdirectory(op_host)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
add_subdirectory(examples)

# 安装目标
# install(TARGETS dpd_kernel DESTINATION lib)  # 暂时跳过内核库安装
install(TARGETS dpd_host DESTINATION lib)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/include)
    install(DIRECTORY include/ DESTINATION include)
endif()

message(STATUS "DPD算子项目配置完成")
message(STATUS "CANN路径: ${CANN_PATH}")
message(STATUS "构建类型: ${CMAKE_BUILD_TYPE}")
