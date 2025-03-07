// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#include "targetver.h"
#define _NO_SCRIPT_GUIDS 0
//#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
// Windows Header Files
#include <windows.h>
#include <d3d11.h>

// C RunTime Header Files
#include <stdlib.h>
#include <malloc.h>
#include <memory.h>
#include <tchar.h>
#include <guiddef.h>


// Spectre
#include <NativeRenderer/Engine.h>
#include <NativeRenderer/Resources/IndexBuffer.h>
#include <NativeRenderer/Resources/ShaderProgram.h>
#include <NativeRenderer/Resources/VertexBuffer.h>
#include <NativeRendererD3D11/RendererD3D11.h>
#include <NativeRendererD3D11/RenderOutputD3D11.h>
#include <Framework/PerformanceTraceLogging.h>
//
// angle
#include <common/platform.h>
#include <angle_gl.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <libANGLE/Context.h>
#include <libANGLE/renderer/d3d/d3d11/Context11.h>
//
// Chakra
#include <jsrt.h>
//
//// C++/WinRT
//#include <winrt/Windows.ApplicationModel.h>
//#include <winrt/Windows.Graphics.Display.h>
//#include <winrt/Windows.UI.Core.h>

// Arcana
#include <arcana/macros.h>
#include <arcana/string.h>
#include <arcana/experimental/array.h>
#include <arcana/threading/cancellation.h>
#include <arcana/threading/coroutine.h>
#include <arcana/threading/dispatcher.h>
#include <arcana/threading/task.h>
#include <arcana/threading/task_conversions.h>

//// GSL
//#include <gsl/gsl>
//
// C++ standard library
#include <algorithm>
#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>
#include <variant>

// C standard library
#include <assert.h>

// N-API
#define NODE_ADDON_API_DISABLE_DEPRECATED
#include <napi.h>
