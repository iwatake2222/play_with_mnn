# Play with MNN
![00_doc/demo_multi.gif](00_doc/demo_multi.gif)

Sample projects to use MNN (https://github.com/alibaba/MNN )

## Target environment
- Platform
    - Linux (x64)
        - Tested in Xubuntu 18 in VirtualBox in Windows 10
    - Linux (armv7)
        - Tested in Raspberry Pi4 (Raspbian 32-bit)
    - Linux (aarch64)
        - Tested in Jetson Nano (JetPack 4.3) and Jetson NX (JetPack 4.4)
    - Android (aarch64)
        - Tested in Pixel 4a
    - Windows (x64). Visual Studio 2017, 2019
        - Tested in Windows10 64-bit

## How to build application code
### Preparation
- Get source code
    ```sh
    git clone https://github.com/iwatake2222/play_with_mnn.git
    cd play_with_mnn
    git submodule update --init
    ```

- Download prebuilt libraries
    - Download prebuilt libraries (ThirdParty.zip) from https://github.com/iwatake2222/InferenceHelper/releases/  (<- Not in this repository)
    - Extract it to `InferenceHelper/ThirdParty/`
- Download models
    - Download models (resource.zip) from https://github.com/iwatake2222/play_with_mnn/releases
    - Extract it to `resource/`


### Linux
```
cd pj_mnn_cls_mobilenet_v2
mkdir build && cd build
cmake ..
make
./main
```

### Option (Camera input)
```sh
cmake .. -DSPEED_TEST_ONLY=off
```

### Windows (Visual Studio)
- Configure and Generate a new project using cmake-gui for Visual Studio 2017 64-bit
    - `Where is the source code` : path-to-play_with_tflite/pj_tflite_cls_mobilenet_v2	(for example)
    - `Where to build the binaries` : path-to-build	(any)
- Open `main.sln`
- Set `main` project as a startup project, then build and run!

### Android project
If you want to run Android project, please select `ViewAndroid` directory in Android Studio.

You will need the following settings at first.

- Configure NDK
    - File -> Project Structure -> SDK Location -> Android NDK location
        - C:\Users\abc\AppData\Local\Android\Sdk\ndk\21.3.6528147
- Import OpenCV
    - Download and extract OpenCV android-sdk (https://github.com/opencv/opencv/releases )
    - File -> New -> Import Module
        - path-to-opencv\opencv-4.3.0-android-sdk\OpenCV-android-sdk\sdk
    - FIle -> Project Structure -> Dependencies -> app -> Declared Dependencies -> + -> Module Dependencies
        - select `sdk`
    - In case you cannot import OpenCV module, remove `sdk` module and dependency of `app` to `sdk` in Project Structure

## How to create pre-built MNN library
pre-built MNN library is stored in InferenceHelper/ThirdParty/MNN_prebuilt . Please use the following commands if you want to build them by yourself.


### Linux
Follow the instruction (https://www.yuque.com/mnn/en/build_android ).

- Native build
```
cd /path/to/MNN
./schema/generate.sh
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=./ .. && make -j4 && make install
```

- Android
```
export ANDROID_NDK=/path/to/android-ndk
cd /path/to/MNN
./schema/generate.sh
cd project/android
mkdir build_32 && cd build_32 && ../build_32.sh
mkdir build_64 && cd build_64 && ../build_64.sh
```

### Windows (Visual Studio)
- Build `MNN\3rd_party\flatbuffers` in Visual Studio (use cmake-gui)
- Copy flatc.exe to any place (Prebuilt executable file is stored in `play_with_mnn\third_party\MNN_prebuilt\tools\x64_windows\flatc.exe`
- Modify `MNN\schema\generate.ps1` not to build flatc.exe and use the created flatc.exe
    ```
    +++ b/schema/generate.ps1
    @@ -9,7 +9,7 @@ if (($args[0] -eq "-lazy") -and ( Test-Path "current" -PathType Container )) {
    }

    # check is flatbuffer installed or not
    -Set-Variable -Name "FLATC" -Value "..\3rd_party\flatbuffers\tmp\flatc.exe"
    +Set-Variable -Name "FLATC" -Value "..\..\MNN_prebuilt\tools\x64_windows\flatc.exe"
    if (-Not (Test-Path $FLATC -PathType Leaf)) {
    echo "*** building flatc ***"

    @@ -21,8 +21,8 @@ if (-Not (Test-Path $FLATC -PathType Leaf)) {
    (cd tmp) -and (rm -r -force *)

    # build
    -  cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release ..
    -  cmake --build . --target flatc
    +  # cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release ..
    +  # cmake --build . --target flatc

    # dir recover
    popd
    ```
- Run the modified `generate.ps` in PowerShell
    - You may need `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
- Create a Visual Studio project using cmake-gui
- Build the project in Visual Studio


## How to create a model converter tool
### Linux
- Install protobuf
    ```
    cd ~/
    git clone https://github.com/protocolbuffers/protobuf.git
    cd protobuf
    git checkout v3.14.0
    ./autogen.sh
    ./configure CXXFLAGS="-fPIC" --prefix=/usr/local --disable-shared 2>&1 > /dev/null
    make -j2
    sudo make install
    sudo ldconfig
    ```
- Build MNN with the following option
    - `cmake .. -DMNN_BUILD_QUANTOOLS=on -DMNN_BUILD_CONVERTER=on -DMNN_BUILD_SHARED_LIBS=off`

### Windows(Visual Studio)
- Install protobuf
    ```
    cd ~/
    git clone https://github.com/protocolbuffers/protobuf.git
    cd protobuf
    git checkout v3.14.0
    ```
    - Create a Visual Studio project for protobuf usin cmake-gui
    - Build protobuf and install
        - Set Runtime library in Code Generation in property as Multithread DLL (/MD) (use the same setting as MNN)
        - Or, change the setting in MNN to use the same setting as protobuf (probably, using /MT is better)
- Create a Visual Studio project for MNN using cmake-gui
    - MNN_BUILD_SHARED_LIBS=off
    - MNN_BUILD_QUANTOOLS=on
    - MNN_BUILD_CONVERTER=on
    - Fill out protobuflib settings
    
    ![protobuflib](00_doc/windows_tool.png) 
- Build the project in Visual Studio

# License
- Copyright 2020 iwatake2222
- Licensed under the Apache License, Version 2.0
    - [LICENSE](LICENSE)

# Acknowledgements
- This project utilizes OSS (Open Source Software)
    - [NOTICE.md](NOTICE.md)
