# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.4.1)

#↓↓↓ 追加 ↓↓↓
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/jniLibs/${ANDROID_ABI})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 ")
#↑↑↑ 追加 ↑↑↑

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

add_library( # Sets the name of the library.
             native-lib

             # Sets the library as a shared library.
             SHARED

             # Provides a relative path to your source file(s).
             native-lib.cpp )

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
              log-lib

              # Specifies the name of the NDK library that
              # you want CMake to locate.
              log )

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
                       native-lib

                       # Links the target library to the log library
                       # included in the NDK.
                       ${log-lib} )

# ↓↓↓ 追加 ↓↓↓
### For OpenCV
#set(OpenCV_DIR "D:/devel/opencv-4.3.0-android-sdk/OpenCV-android-sdk/sdk/native/jni")
set(OpenCV_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../sdk/native/jni")
find_package(OpenCV REQUIRED)
target_include_directories(native-lib PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(native-lib ${OpenCV_LIBS})

### For my module
set(INFERENCE_HELPER_ENABLE_OPENCV OFF CACHE BOOL "OPENCV" FORCE)
set(INFERENCE_HELPER_ENABLE_MNN ON CACHE BOOL "MNN")

set(ImageProcessor_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../../pj_mnn_posenet_mobilenet_v1/image_processor")
message(${ImageProcessor_DIR})
add_subdirectory(${ImageProcessor_DIR} ImageProcessor)
target_include_directories(native-lib PUBLIC ${ImageProcessor_DIR})
target_link_libraries(native-lib ImageProcessor)
# ↑↑↑ 追加 ↑↑↑
