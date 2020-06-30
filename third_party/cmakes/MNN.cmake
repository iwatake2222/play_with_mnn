set(MNN_DIR ${CMAKE_CURRENT_LIST_DIR}/../MNN)
set(MNN_INC ${MNN_DIR}/include)


set(USE_PREBUILT_MNN on CACHE BOOL "Use Prebuilt MNN? [on/off]")
if(USE_PREBUILT_MNN)
	if(DEFINED  ANDROID_ABI)
		# set(MNN_LIB ${CMAKE_CURRENT_LIST_DIR}/../MNN_prebuilt/android/${ANDROID_ABI}/libMNN.so)
		add_library( MNN SHARED IMPORTED GLOBAL )
		set_target_properties(
			MNN
			PROPERTIES IMPORTED_LOCATION
			${CMAKE_CURRENT_LIST_DIR}/../MNN_prebuilt/android/${ANDROID_ABI}/libMNN.so
			)
			set(MNN_LIB MNN)
	elseif(MSVC_VERSION)
		file(COPY ${CMAKE_CURRENT_LIST_DIR}/../MNN_prebuilt/x64_windows/VS2017/Debug/MNN.dll DESTINATION ${CMAKE_BINARY_DIR}/Debug)
		file(COPY ${CMAKE_CURRENT_LIST_DIR}/../MNN_prebuilt/x64_windows/VS2017/RelWithDebInfo/MNN.dll DESTINATION ${CMAKE_BINARY_DIR}/RelWithDebInfo)
		file(COPY ${CMAKE_CURRENT_LIST_DIR}/../MNN_prebuilt/x64_windows/VS2017/Release/MNN.dll DESTINATION ${CMAKE_BINARY_DIR}/Release)
		file(COPY ${CMAKE_CURRENT_LIST_DIR}/../MNN_prebuilt/x64_windows/VS2017/MinSizeRel/MNN.dll DESTINATION ${CMAKE_BINARY_DIR}/MinSizeRel)
		# file(COPY ${CMAKE_CURRENT_LIST_DIR}/../MNN_prebuilt/x64_windows/MNNd.dll DESTINATION ${CMAKE_BINARY_DIR})
		if((MSVC_VERSION GREATER_EQUAL 1910) AND (MSVC_VERSION LESS 1920))
			set(MNN_LIB
				$<$<CONFIG:Debug>:${CMAKE_CURRENT_LIST_DIR}/../MNN_prebuilt/x64_windows/VS2017/Debug/MNN.lib>
				$<$<CONFIG:RelWithDebInfo>:${CMAKE_CURRENT_LIST_DIR}/../MNN_prebuilt/x64_windows/VS2017/RelWithDebInfo/MNN.lib>
				$<$<CONFIG:Release>:${CMAKE_CURRENT_LIST_DIR}/../MNN_prebuilt/x64_windows/VS2017/Release/MNN.lib>
				$<$<CONFIG:MinSizeRel>:${CMAKE_CURRENT_LIST_DIR}/../MNN_prebuilt/x64_windows/VS2017/MinSizeRel/MNN.lib>
			)
		else()
			message(FATAL_ERROR "[MNN] unsupported MSVC version")
		endif()
	else()
		set(MNN_LIB
			# $<$<STREQUAL:${BUILD_SYSTEM},x64_windows>:${CMAKE_CURRENT_LIST_DIR}/../MNN_prebuilt/x64_windows/lib/ncnn.lib>
			$<$<STREQUAL:${BUILD_SYSTEM},x64_linux>:${CMAKE_CURRENT_LIST_DIR}/../MNN_prebuilt/x64_linux/libMNN.so>
			$<$<STREQUAL:${BUILD_SYSTEM},armv7>:${CMAKE_CURRENT_LIST_DIR}/../MNN_prebuilt/armv7/libMNN.so>
			$<$<STREQUAL:${BUILD_SYSTEM},aarch64>:${CMAKE_CURRENT_LIST_DIR}/../MNN_prebuilt/aarch64/libMNN.so>
		)
	endif()
else()
	add_subdirectory(${MNN_DIR} MNN)
	target_link_libraries(${PROJECT_NAME} MNN)
endif()

