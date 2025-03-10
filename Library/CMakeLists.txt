set(NAPI_JAVASCRIPT_ENGINE "Chakra" CACHE STRING "JavaScript engine for N-API.")
add_subdirectory(Dependencies/napi EXCLUDE_FROM_ALL)
set_property(TARGET napi PROPERTY FOLDER Dependencies/napi)
set(BABYLON_NATIVE_DEPENDENCIES ${NAPI_DEPENDENCIES} PARENT_SCOPE)

add_subdirectory(Dependencies/glslang EXCLUDE_FROM_ALL)
set_property(TARGET glslang PROPERTY FOLDER Dependencies/glslang)
set_property(TARGET OGLCompiler PROPERTY FOLDER Dependencies/glslang)
set_property(TARGET OSDependent PROPERTY FOLDER Dependencies/glslang)
set_property(TARGET SPIRV PROPERTY FOLDER Dependencies/glslang)
set_property(TARGET SPVRemapper PROPERTY FOLDER Dependencies/glslang)
set_property(TARGET HLSL PROPERTY FOLDER Dependencies/glslang)

add_subdirectory(Dependencies/SPIRV-Cross EXCLUDE_FROM_ALL)
set_property(TARGET spirv-cross-core PROPERTY FOLDER Dependencies/SPIRV-Cross)
set_property(TARGET spirv-cross-glsl PROPERTY FOLDER Dependencies/SPIRV-Cross)
set_property(TARGET spirv-cross-hlsl PROPERTY FOLDER Dependencies/SPIRV-Cross)

add_compile_definitions(BGFX_CONFIG_DEBUG_UNIFORM=0)
add_compile_definitions(BGFX_CONFIG_MULTITHREADED=0)
add_compile_definitions(BGFX_CONFIG_MAX_VERTEX_STREAMS=32)
add_compile_definitions(BGFX_CONFIG_MAX_COMMAND_BUFFER_SIZE=12582912)
set(BGFX_BUILD_EXAMPLES OFF CACHE BOOL "Build the BGFX examples.")
set(BGFX_BUILD_TOOLS OFF CACHE BOOL "Build the BGFX tools.")
add_subdirectory(Dependencies/bgfx.cmake EXCLUDE_FROM_ALL)
set_property(TARGET astc PROPERTY FOLDER Dependencies/bgfx/3rdparty)
set_property(TARGET astc-codec PROPERTY FOLDER Dependencies/bgfx/3rdparty)
set_property(TARGET edtaa3 PROPERTY FOLDER Dependencies/bgfx/3rdparty)
set_property(TARGET etc1 PROPERTY FOLDER Dependencies/bgfx/3rdparty)
set_property(TARGET etc2 PROPERTY FOLDER Dependencies/bgfx/3rdparty)
set_property(TARGET iqa PROPERTY FOLDER Dependencies/bgfx/3rdparty)
set_property(TARGET nvtt PROPERTY FOLDER Dependencies/bgfx/3rdparty)
set_property(TARGET pvrtc PROPERTY FOLDER Dependencies/bgfx/3rdparty)
set_property(TARGET squish PROPERTY FOLDER Dependencies/bgfx/3rdparty)
set_property(TARGET bgfx PROPERTY FOLDER Dependencies/bgfx)
set_property(TARGET bimg PROPERTY FOLDER Dependencies/bgfx)
set_property(TARGET bx PROPERTY FOLDER Dependencies/bgfx)

set(CMAKE_USE_WINSSL ON CACHE BOOL "Set cURL to use WinSSL by default.")
add_subdirectory(Dependencies/curl EXCLUDE_FROM_ALL)
set_property(TARGET libcurl PROPERTY FOLDER Dependencies/curl)

set(SOURCES
    "Include/Babylon/Runtime.h"
    "Include/Babylon/Runtime${BABYLON_NATIVE_PLATFORM}.h"
    "Source/Common.h"
    "Source/Common${BABYLON_NATIVE_PLATFORM}.cpp"
    "Source/Console.cpp"
    "Source/Console.h"
    "Source/NapiBridge.h"
    "Source/NativeEngine.cpp"
    "Source/NativeEngine.h"
    "Source/ResourceLimits.cpp"
    "Source/ResourceLimits.h"
    "Source/Runtime.cpp"
    "Source/RuntimeImpl.cpp"
    "Source/RuntimeImpl.h"
    "Source/Runtime${BABYLON_NATIVE_PLATFORM}.cpp"
    "Source/ShaderCompiler.h"
    "Source/Window.cpp"
    "Source/Window.h"
    "Source/XMLHttpRequest.cpp"
    "Source/XMLHttpRequest.h"
    "Source/XMLHttpRequest${BABYLON_NATIVE_PLATFORM}.cpp")

if(BABYLON_NATIVE_PLATFORM STREQUAL "Win32" OR BABYLON_NATIVE_PLATFORM STREQUAL "UWP")
    set(SOURCES ${SOURCES}
        "Source/ShaderCompilerD3D.cpp")
endif()

add_library(Library ${SOURCES})

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${SOURCES})

target_compile_definitions(Library PRIVATE NOMINMAX)
target_compile_definitions(Library PRIVATE _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING)

target_include_directories(Library PRIVATE "Dependencies/arcana.cpp/Source/Shared")
target_include_directories(Library PRIVATE "Dependencies/arcana.cpp/Source/Submodules/GSL/include")
target_include_directories(Library PRIVATE "Dependencies/base-n/include")
target_include_directories(Library PRIVATE "Dependencies/bgfx.cmake/bgfx/include")
target_include_directories(Library PRIVATE "Dependencies/bgfx.cmake/bimg/include")
target_include_directories(Library PRIVATE "Dependencies/bgfx.cmake/bx/include")
target_include_directories(Library PRIVATE "Dependencies/bgfx.cmake/bx/include/compat/msvc")
target_include_directories(Library PRIVATE "Dependencies/curl/include")
target_include_directories(Library PRIVATE "Dependencies/glslang")
target_include_directories(Library PRIVATE "Dependencies/SPIRV-Cross")

target_link_libraries(Library
    PRIVATE bx
    PRIVATE bimg
    PRIVATE bgfx
    PRIVATE libcurl
    PRIVATE glslang
    PRIVATE SPIRV
    PRIVATE spirv-cross-hlsl)

target_link_libraries(Library
    PRIVATE "crypt32.lib"
    PRIVATE "d2d1.lib"
    PRIVATE "d3d11.lib"
    PRIVATE "d3d9.lib"
    PRIVATE "d3dcompiler.lib"
    PRIVATE "dxgi.lib"
    PRIVATE "dxguid.lib"
    PRIVATE "normaliz.lib"
    PRIVATE "psapi.lib"
    PRIVATE "shlwapi.lib"
    PRIVATE "ws2_32.lib"
    PRIVATE "wldap32.lib")

target_include_directories(Library PUBLIC "Include")
target_link_libraries(Library PUBLIC napi)