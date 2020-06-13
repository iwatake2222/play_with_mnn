set(MNN_DIR ${CMAKE_SOURCE_DIR}/../third_party/MNN)
target_include_directories(${PROJECT_NAME} PUBLIC ${MNN_DIR})
target_include_directories(${PROJECT_NAME} PUBLIC ${MNN_DIR}/include)
target_include_directories(${PROJECT_NAME} PUBLIC ${MNN_DIR}/3rd_party/imageHelper)

set(USE_PREBUILT_MNN on CACHE BOOL "Use Prebuilt MNN? [on/off]")
if(USE_PREBUILT_MNN)
	if(MSVC_VERSION)
		file(COPY ${CMAKE_CURRENT_LIST_DIR}/../MNN_prebuilt/x64_windows/MNN.dll DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
		# file(COPY ${CMAKE_CURRENT_LIST_DIR}/../MNN_prebuilt/x64_windows/MNNd.dll DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
		if((MSVC_VERSION GREATER_EQUAL 1910) AND (MSVC_VERSION LESS 1920))
			target_link_libraries(${PROJECT_NAME}
				$<$<CONFIG:Debug>:${CMAKE_CURRENT_LIST_DIR}/../MNN_prebuilt/x64_windows/MNN.lib>
				$<$<CONFIG:RelWithDebInfo>:${CMAKE_CURRENT_LIST_DIR}/../MNN_prebuilt/x64_windows/MNN.lib>
				$<$<CONFIG:Release>:${CMAKE_CURRENT_LIST_DIR}/../MNN_prebuilt/x64_windows/MNN.lib>
				$<$<CONFIG:MinSizeRel>:${CMAKE_CURRENT_LIST_DIR}/../MNN_prebuilt/x64_windows/MNN.lib>
			)
		else()
			message(FATAL_ERROR "[MNN] unsupported MSVC version")
		endif()
	else()
		target_link_libraries(${ProjectName}
			# $<$<STREQUAL:${BUILD_SYSTEM},x64_windows>:${CMAKE_SOURCE_DIR}/../third_party/ncnn_prebuilt/x64_windows/lib/ncnn.lib>
			$<$<STREQUAL:${BUILD_SYSTEM},x64_linux>:${CMAKE_SOURCE_DIR}/../third_party/ncnn_prebuilt/x64_linux/lib/libncnn.a>
			$<$<STREQUAL:${BUILD_SYSTEM},armv7>:${CMAKE_SOURCE_DIR}/../third_party/ncnn_prebuilt/armv7/lib/libncnn.a>
			$<$<STREQUAL:${BUILD_SYSTEM},aarch64>:${CMAKE_SOURCE_DIR}/../third_party/ncnn_prebuilt/aarch64/lib/libncnn.a>
		)
	endif()
else()
	add_subdirectory(${MNN_DIR} MNN)
	target_link_libraries(${PROJECT_NAME} MNN)
endif()

