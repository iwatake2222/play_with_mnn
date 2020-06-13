# Play with MNN
Sample projects to use MNN (https://github.com/alibaba/MNN )

## Target environment
- Windows(MSVC) (x64)
- Linux (x64)
- Linux (armv7) e.g. Raspberry Pi 3,4
- Linux (aarch64) e.g. Jetson Nano
- *Native build only (Cross build is not supported)


## How to build application code
```
cd play_with_mnn
git submodule init
git submodule update

cd pj_mnn_cls_mobilenet_v2
mkdir build && cd build
cmake ..
make

./main
```

If you use Visual Studio, please use cmake-gui to generate project files.

## How to create pre-built MNN library
pre-built MNN library is stored in third_party/MNN_prebuilt . Please use the following commands if you want to build them by yourself.


### Linux
```
cd third_party/MNN/schema
./generate.sh
cd ../../../

cd pj_mnn_cls_mobilenet_v2
mkdir build && cd build
cmake .. -DUSE_PREBUILT_MNN=off
make

./main
```

### Windows (Visual Studio)
- Build `third_party\MNN\3rd_party\flatbuffers` in Visual Studio (use cmake-gui)
- Copy flatc.exe to any place (Prebuilt executable file is stored in `play_with_mnn\third_party\MNN_prebuilt\tools\x64_windows\flatc.exe`
- Modify `play_with_mnn\third_party\MNN\schema\generate.ps1` not to build flatc.exe and use the created flatc.exe
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
	git checkout  v3.7.1
	./autogen.sh
	./configure CXXFLAGS="-fPIC" --prefix=/usr/local --disable-shared 2>&1 > /dev/null
	make -j2
	sudo make install
	sudo ldconfig
	```
- Build MNN with the following option
	- `cmake .. -DUSE_PREBUILT_MNN=off -DMNN_BUILD_QUANTOOLS=on -DMNN_BUILD_CONVERTER=on `
- (in my environment, the converter with `-DMNN_BUILD_SHARED_LIBS=off` causes segmentation fault...)

### Windows(Visual Studio)
- Install protobuf
	```
	cd ~/
	git clone https://github.com/protocolbuffers/protobuf.git
	cd protobuf
	git checkout  v3.7.1
	```
	- Create a Visual Studio project for protobuf usin cmake-gui
	- Build protobuf and install
		- Set Runtime library in Code Generation in property as Multithread (/MT)
- Create a Visual Studio project for MNN using cmake-gui
	- MNN_BUILD_SHARED_LIBS=off
	- MNN_BUILD_QUANTOOLS=on
	- MNN_BUILD_CONVERTER=on
	- Fill out protobuflib settings
		
		![protobuflib](00_doc/windows_tool.png) 
- Build the project in Visual Studio

# Acknowledgements
- This project includes the work that is distributed in the Apache License 2.0.
- This project includes MNN (https://github.com/alibaba/MNN ) as a submodule
- The models are retrieved from the followings:
	- mobilenet_v2_1.0_224.tflite
		- https://www.tensorflow.org/lite/guide/hosted_models
		- https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz
		- `MNNConvert.exe -f TFLITE --modelFile mobilenet_v2_1.0_224.tflite --MNNModel mobilenet_v2_1.0_224.mnn --bizCode biz`
	- posenet-mobilenet_v1_075.pb
		- https://github.com/czy2014hust/posenet-python
		- https://github.com/czy2014hust/posenet-python/blob/master/models/model-mobilenet_v1_075.pb?raw=true
		- `MNNConvert.exe -f TF --modelFile posenet-mobilenet_v1_075.pb --MNNModel posenet-mobilenet_v1_075.mnn --bizCode biz`
	- deeplabv3_257_mv_gpu.tflite
		- https://www.tensorflow.org/lite/models/segmentation/overview
		- https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/deeplabv3_257_mv_gpu.tflite
		- `MNNConvert.exe -f TFLITE --modelFile deeplabv3_257_mv_gpu.tflite --MNNModel deeplabv3_257_mv_gpu.mnn --bizCode biz`
