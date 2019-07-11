#include "NativeEngine.h"

#include "RuntimeImpl.h"
#include "NapiBridge.h"
#include "ShaderCompiler.h"

#include <bgfx/bgfx.h>
#include <bgfx/platform.h>

// TODO: this needs to be fixed in bgfx
namespace bgfx
{
    uint16_t attribToId(Attrib::Enum _attr);
}

#define BGFX_UNIFORM_FRAGMENTBIT UINT8_C(0x10) // Copy-pasta from bgfx_p.h
#define BGFX_UNIFORM_SAMPLERBIT  UINT8_C(0x20) // Copy-pasta from bgfx_p.h

#include <bimg/bimg.h>
#include <bimg/decode.h>
#include <bimg/encode.h>

#include <bx/math.h>
#include <bx/readerwriter.h>

#include <regex>
#include <sstream>

#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>
#include <array>
#include <map>
#include <algorithm>
#include <assert.h>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma comment(lib, "kernel32.lib")
#pragma comment(lib, "advapi32.lib")

#include <d3d11.h>
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "windowscodecs.lib")


#define XR_USE_PLATFORM_WIN32
#define XR_USE_GRAPHICS_API_D3D11
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#include <XrError.h>
#include "xr_linear.h"

namespace babylon
{
    namespace
    {
        struct UniformInfo final
        {
            uint8_t Stage{};
            bgfx::UniformHandle Handle{};
        };

        template<typename AppendageT>
        inline void AppendBytes(std::vector<uint8_t>& bytes, const AppendageT appendage)
        {
            auto ptr = reinterpret_cast<const uint8_t*>(&appendage);
            auto stride = static_cast<std::ptrdiff_t>(sizeof(AppendageT));
            bytes.insert(bytes.end(), ptr, ptr + stride);
        }

        template<typename AppendageT = std::string&>
        inline void AppendBytes(std::vector<uint8_t>& bytes, const std::string& string)
        {
            auto ptr = reinterpret_cast<const uint8_t*>(string.data());
            auto stride = static_cast<std::ptrdiff_t>(string.length());
            bytes.insert(bytes.end(), ptr, ptr + stride);
        }

        template<typename ElementT>
        inline void AppendBytes(std::vector<uint8_t>& bytes, const gsl::span<ElementT>& data)
        {
            auto ptr = reinterpret_cast<const uint8_t*>(data.data());
            auto stride = static_cast<std::ptrdiff_t>(data.size() * sizeof(ElementT));
            bytes.insert(bytes.end(), ptr, ptr + stride);
        }

        void FlipYInImageBytes(gsl::span<uint8_t> bytes, size_t rowCount, size_t rowPitch)
        {
            std::vector<uint8_t> buffer{};
            buffer.reserve(rowPitch);

            for (size_t row = 0; row < rowCount / 2; row++)
            {
                auto frontPtr = bytes.data() + (row * rowPitch);
                auto backPtr = bytes.data() + ((rowCount - row - 1) * rowPitch);

                std::memcpy(buffer.data(), frontPtr, rowPitch);
                std::memcpy(frontPtr, backPtr, rowPitch);
                std::memcpy(backPtr, buffer.data(), rowPitch);
            }
        }

        void AppendUniformBuffer(std::vector<uint8_t>& bytes, const spirv_cross::Compiler& compiler, const spirv_cross::Resource& uniformBuffer, bool isFragment)
        {
            const uint8_t fragmentBit = (isFragment ? BGFX_UNIFORM_FRAGMENTBIT : 0);

            const spirv_cross::SPIRType& type = compiler.get_type(uniformBuffer.base_type_id);
            for (uint32_t index = 0; index < type.member_types.size(); ++index)
            {
                auto name = compiler.get_member_name(uniformBuffer.base_type_id, index);
                auto offset = compiler.get_member_decoration(uniformBuffer.base_type_id, index, spv::DecorationOffset);
                auto memberType = compiler.get_type(type.member_types[index]);

                bgfx::UniformType::Enum bgfxType;
                uint16_t regCount;

                if (memberType.basetype != spirv_cross::SPIRType::Float)
                {
                    throw std::exception("Not supported");
                }

                if (memberType.columns == 1 && 1 <= memberType.vecsize && memberType.vecsize <= 4)
                {
                    bgfxType = bgfx::UniformType::Vec4;
                    regCount = 1;
                }
                else if (memberType.columns == 4 && memberType.vecsize == 4)
                {
                    bgfxType = bgfx::UniformType::Mat4;
                    regCount = 4;
                }
                else
                {
                    throw std::exception("Not supported");
                }

                for (const auto size : memberType.array)
                {
                    regCount *= size;
                }

                AppendBytes(bytes, static_cast<uint8_t>(name.size()));
                AppendBytes(bytes, name);
                AppendBytes(bytes, static_cast<uint8_t>(bgfxType | fragmentBit));
                AppendBytes(bytes, static_cast<uint8_t>(0)); // Value "num" not used by D3D11 pipeline.
                AppendBytes(bytes, static_cast<uint16_t>(offset));
                AppendBytes(bytes, static_cast<uint16_t>(regCount));
            }
        }

        void AppendSamplers(std::vector<uint8_t>& bytes, const spirv_cross::Compiler& compiler, const spirv_cross::SmallVector<spirv_cross::Resource>& samplers, bool isFragment, std::unordered_map<std::string, UniformInfo>& cache)
        {
            const uint8_t fragmentBit = (isFragment ? BGFX_UNIFORM_FRAGMENTBIT : 0);

            for (const spirv_cross::Resource& sampler : samplers)
            {
                AppendBytes(bytes, static_cast<uint8_t>(sampler.name.size()));
                AppendBytes(bytes, sampler.name);
                AppendBytes(bytes, static_cast<uint8_t>(bgfx::UniformType::Sampler | BGFX_UNIFORM_SAMPLERBIT));

                // These values (num, regIndex, regCount) are not used by D3D11 pipeline.
                AppendBytes(bytes, static_cast<uint8_t>(0));
                AppendBytes(bytes, static_cast<uint16_t>(0));
                AppendBytes(bytes, static_cast<uint16_t>(0));

                cache[sampler.name].Stage = compiler.get_decoration(sampler.id, spv::DecorationBinding);
            }
        }

        void CacheUniformHandles(bgfx::ShaderHandle shader, std::unordered_map<std::string, UniformInfo>& cache)
        {
            const auto MAX_UNIFORMS = 256;
            bgfx::UniformHandle uniforms[MAX_UNIFORMS];
            auto numUniforms = bgfx::getShaderUniforms(shader, uniforms, MAX_UNIFORMS);

            bgfx::UniformInfo info{};
            for (uint8_t idx = 0; idx < numUniforms; idx++)
            {
                bgfx::getUniformInfo(uniforms[idx], info);
                cache[info.name].Handle = uniforms[idx];
            }
        }

        enum class WebGLAttribType
        {
            BYTE = 5120,
            UNSIGNED_BYTE = 5121,
            SHORT = 5122,
            UNSIGNED_SHORT = 5123,
            INT = 5124,
            UNSIGNED_INT = 5125,
            FLOAT = 5126
        };

        bgfx::AttribType::Enum ConvertAttribType(WebGLAttribType type)
        {
            switch (type)
            {
            case WebGLAttribType::UNSIGNED_BYTE:    return bgfx::AttribType::Uint8;
            case WebGLAttribType::SHORT:            return bgfx::AttribType::Int16;
            case WebGLAttribType::FLOAT:            return bgfx::AttribType::Float;
            }

            throw std::exception("Unsupported attribute type");
        }

        // Must match constants.ts in Babylon.js.
        constexpr std::array<uint64_t, 11> ALPHA_MODE
        {
            // ALPHA_DISABLE
            0x0,

            // ALPHA_ADD: SRC ALPHA * SRC + DEST
            BGFX_STATE_BLEND_FUNC_SEPARATE(BGFX_STATE_BLEND_SRC_ALPHA, BGFX_STATE_BLEND_ONE, BGFX_STATE_BLEND_ZERO, BGFX_STATE_BLEND_ONE),

            // ALPHA_COMBINE: SRC ALPHA * SRC + (1 - SRC ALPHA) * DEST
            BGFX_STATE_BLEND_FUNC_SEPARATE(BGFX_STATE_BLEND_SRC_ALPHA, BGFX_STATE_BLEND_INV_SRC_ALPHA, BGFX_STATE_BLEND_ONE, BGFX_STATE_BLEND_ONE),

            // ALPHA_SUBTRACT: DEST - SRC * DEST
            BGFX_STATE_BLEND_FUNC_SEPARATE(BGFX_STATE_BLEND_ZERO, BGFX_STATE_BLEND_INV_SRC_COLOR, BGFX_STATE_BLEND_ONE, BGFX_STATE_BLEND_ONE),

            // ALPHA_MULTIPLY: SRC * DEST
            BGFX_STATE_BLEND_FUNC_SEPARATE(BGFX_STATE_BLEND_DST_COLOR, BGFX_STATE_BLEND_ZERO, BGFX_STATE_BLEND_ONE, BGFX_STATE_BLEND_ONE),

            // ALPHA_MAXIMIZED: SRC ALPHA * SRC + (1 - SRC) * DEST
            BGFX_STATE_BLEND_FUNC_SEPARATE(BGFX_STATE_BLEND_SRC_ALPHA, BGFX_STATE_BLEND_INV_SRC_COLOR, BGFX_STATE_BLEND_ONE, BGFX_STATE_BLEND_ONE),

            // ALPHA_ONEONE: SRC + DEST
            BGFX_STATE_BLEND_FUNC_SEPARATE(BGFX_STATE_BLEND_ONE, BGFX_STATE_BLEND_ONE, BGFX_STATE_BLEND_ZERO, BGFX_STATE_BLEND_ONE),

            // ALPHA_PREMULTIPLIED: SRC + (1 - SRC ALPHA) * DEST
            BGFX_STATE_BLEND_FUNC_SEPARATE(BGFX_STATE_BLEND_ONE, BGFX_STATE_BLEND_INV_SRC_ALPHA, BGFX_STATE_BLEND_ONE, BGFX_STATE_BLEND_ONE),

            // ALPHA_PREMULTIPLIED_PORTERDUFF: SRC + (1 - SRC ALPHA) * DEST, (1 - SRC ALPHA) * DEST ALPHA
            BGFX_STATE_BLEND_FUNC_SEPARATE(BGFX_STATE_BLEND_ONE, BGFX_STATE_BLEND_INV_SRC_ALPHA, BGFX_STATE_BLEND_ONE, BGFX_STATE_BLEND_INV_SRC_ALPHA),

            // ALPHA_INTERPOLATE: CST * SRC + (1 - CST) * DEST
            BGFX_STATE_BLEND_FUNC_SEPARATE(BGFX_STATE_BLEND_FACTOR, BGFX_STATE_BLEND_INV_FACTOR, BGFX_STATE_BLEND_FACTOR, BGFX_STATE_BLEND_INV_FACTOR),

            // ALPHA_SCREENMODE: SRC + (1 - SRC) * DEST, SRC ALPHA + (1 - SRC ALPHA) * DEST ALPHA
            BGFX_STATE_BLEND_FUNC_SEPARATE(BGFX_STATE_BLEND_ONE, BGFX_STATE_BLEND_INV_SRC_COLOR, BGFX_STATE_BLEND_ONE, BGFX_STATE_BLEND_INV_SRC_ALPHA),
        };
    }

	inline std::string Fmt(const char* fmt, ...) {
		va_list vl;
		va_start(vl, fmt);
		int size = std::vsnprintf(nullptr, 0, fmt, vl);
		va_end(vl);

		if (size != -1) {
			std::unique_ptr<char[]> buffer(new char[size + 1]);

			va_start(vl, fmt);
			size = std::vsnprintf(buffer.get(), size + 1, fmt, vl);
			va_end(vl);
			if (size != -1) {
				return std::string(buffer.get(), size);
			}
		}

		throw std::runtime_error("Unexpected vsnprintf failure");
	}

	namespace Log {
		enum class Level { Verbose, Info, Warning, Error };
		void SetLevel(Level minSeverity) { }

		void Write(Level severity, const std::string& msg) {
		}
	}

	class BGFXApp {
	public:
		void initGraphicsDevice(XrInstance xrInstance, XrSystemId systemId) {
			// Create the D3D11 device for the adapter associated with the system.
			XrGraphicsRequirementsD3D11KHR graphicsRequirements{ XR_TYPE_GRAPHICS_REQUIREMENTS_D3D11_KHR };
			CHECK_XRCMD(xrGetD3D11GraphicsRequirementsKHR(xrInstance, systemId, &graphicsRequirements));

			// Init bgfx
			// bgfx::renderFrame(); only needed with multithreaded
			bgfx::Init bgfxInit;
			bgfxInit.type = bgfx::RendererType::Direct3D11;
			bgfxInit.vendorId = BGFX_PCI_ID_NVIDIA;
			bgfxInit.deviceId = (uint16_t)graphicsRequirements.adapterLuid.LowPart;
			bgfxInit.resolution.width = 1000;
			bgfxInit.resolution.height = 1000;
			bgfxInit.resolution.reset = BGFX_RESET_VSYNC;
			bgfx::init(bgfxInit);

			// Print device id
			auto caps = bgfx::getCaps();
			//std::cout << "BGFX initialized with DeviceID: " << caps->deviceId << "\n";

			// Enable debug text.
			// bgfx::setDebug(true);

			m_graphicsBinding.device = getDeviceReference();
		}

		ID3D11Device* getDeviceReference() {
			auto intern = bgfx::getInternalData();
			return (ID3D11Device*)intern->context;
		}

		int64_t selectColorSwapchainFormat(const std::vector<int64_t>& runtimeFormats) {
			// List of supported color swapchain formats, in priority order.
			constexpr DXGI_FORMAT SupportedColorSwapchainFormats[] = {
				DXGI_FORMAT_R8G8B8A8_UNORM,
				DXGI_FORMAT_B8G8R8A8_UNORM,
				DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
				DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
			};

			auto swapchainFormatIt =
				std::find_first_of(std::begin(SupportedColorSwapchainFormats), std::end(SupportedColorSwapchainFormats),
					runtimeFormats.begin(), runtimeFormats.end());
			if (swapchainFormatIt == std::end(SupportedColorSwapchainFormats)) {
				THROW("No runtime swapchain format supported for color swapchain");
			}

			return *swapchainFormatIt;
		}



		void renderView(const XrCompositionLayerProjectionView& layerView, const XrSwapchainImageBaseHeader* swapchainImage,
			int64_t swapchainFormat) {
			if (swapchainFormat) {
			}
			// Shared
			CHECK(layerView.subImage.imageArrayIndex == 0);  // Texture arrays not supported.
			ID3D11Texture2D* const colorTexture = reinterpret_cast<const XrSwapchainImageD3D11KHR*>(swapchainImage)->texture;

			// BGFX
			counter++;
			bgfx::ViewId view = 0;
			bgfx::setViewName(view, "standard view");
			bgfx::setViewRect(view, (uint16_t)layerView.subImage.imageRect.offset.x, (uint16_t)layerView.subImage.imageRect.offset.y,
				(uint16_t)layerView.subImage.imageRect.extent.width,
				(uint16_t)layerView.subImage.imageRect.extent.height);
			bgfx::setViewClear(view, BGFX_CLEAR_COLOR | BGFX_CLEAR_DEPTH, counter < 500 ? 0xff3030ff : 0xff30FFff, 1.0f, 0);
			//std::cout << counter << "\n";

			auto frameId = (uintptr_t)colorTexture;
			if (textures.find(frameId) == textures.end()) {
				D3D11_TEXTURE2D_DESC colorDesc;
				colorTexture->GetDesc(&colorDesc);

				textures.insert(std::make_pair(
					frameId, bgfx::createTexture2D((uint16_t)1, (uint16_t)1, false, 1, bgfx::TextureFormat::RGBA8, BGFX_TEXTURE_RT)));

				//std::cout << "New frame added \n";
			}

			bgfx::overrideInternal(textures.at(frameId), (uintptr_t)colorTexture);
			if (framebuffers.find(frameId) != framebuffers.end()) {
				auto fb = framebuffers.at(frameId);
				// TODO: this is weird that I need to delete the framebuffer every frame and create a new one
				bgfx::destroy(fb);
			}

			framebuffers.erase(frameId);
			framebuffers.insert(std::make_pair(frameId, bgfx::createFrameBuffer(1, &textures.at(frameId))));

			/*  bgfx::overrideInternal(textures.at(frameId), (uintptr_t)colorTexture);

					  framebuffers.erase(frameId);
					  framebuffers.insert(std::make_pair(frameId, bgfx::createFrameBuffer(1, &textures.at(frameId))));*/

			bgfx::FrameBufferHandle frameBuffer = framebuffers.at(frameId);

			auto q = bx::Quaternion();
			q.x = layerView.pose.orientation.x;
			q.y = layerView.pose.orientation.y;
			q.z = layerView.pose.orientation.z;
			q.w = layerView.pose.orientation.w;

			// layerView.pose.orientation.
			const bx::Vec3 at = { 0.0f, 0.0f, 1.0f };

			// float(-counter)

			const bx::Vec3 eye = { layerView.pose.position.x, layerView.pose.position.y, layerView.pose.position.z };

			auto lookAt = bx::add(eye, bx::mul(at, q));

			float m_width = (float)layerView.subImage.imageRect.extent.width;
			float m_height = (float)layerView.subImage.imageRect.extent.height;
			// Set view and projection matrix for view 0.
			{
				// float viewMatA[16];
				float viewMat[16];
				// bx::mtxLookAt(viewMat, eye, lookAt);
				bx::mtxQuatTranslation(viewMat, q, eye);

				// bx::mtxInverse(viewMat, viewMatA);

				float proj[16];
				bx::mtxProj(proj, 60.0f, float(m_width) / float(m_height), 0.1f, 1000.0f, bgfx::getCaps()->homogeneousDepth);

				XrMatrix4x4f projectionMatrix;
				XrMatrix4x4f_CreateProjectionFov(&projectionMatrix, GRAPHICS_D3D, layerView.fov, 0.05f, 100.0f);
				for (uint16_t j = 0; j < 16; j++) {
					proj[j] = projectionMatrix.m[j];
				}

				bgfx::setViewTransform(view, viewMat, proj);

				// Set view 0 default viewport.
				// bgfx::setViewRect(view, 0, 0, uint16_t(m_width), uint16_t(m_height));
			}

			bgfx::setViewFrameBuffer(view, frameBuffer);
			bgfx::touch(view);

			// m_pt = 2;
			// bgfx::IndexBufferHandle ibh = m_ibh[m_pt];
			// uint64_t state = 0 | (m_r ? BGFX_STATE_WRITE_R : 0) | (m_g ? BGFX_STATE_WRITE_G : 0) | (m_b ? BGFX_STATE_WRITE_B : 0) |
			//                 (m_a ? BGFX_STATE_WRITE_A : 0) | BGFX_STATE_WRITE_Z | BGFX_STATE_DEPTH_TEST_LESS | BGFX_STATE_CULL_CW |
			//                 BGFX_STATE_MSAA | s_ptState[m_pt];

			//// Submit 11x11 cubes.
			// for (uint32_t yy = 0; yy < 11; ++yy) {
			//    for (uint32_t xx = 0; xx < 11; ++xx) {
			//        float mtx[16];
			//        bx::mtxRotateXY(mtx, 0, 0);
			//        mtx[12] = -15.0f + float(xx) * 3.0f;
			//        mtx[13] = -15.0f + float(yy) * 3.0f;
			//        mtx[14] = 0.0f;

			//        // Set model matrix for rendering.
			//        bgfx::setTransform(mtx);

			//        // Set vertex and index buffer.
			//        bgfx::setVertexBuffer(0, m_vbh);
			//        bgfx::setIndexBuffer(ibh);

			//        // Set render states.
			//        bgfx::setState(state);

			//        // Submit primitive for rendering to view 0.
			//        bgfx::submit(0, m_program);
			//    }
			//}
			if (counter > 6) {
				counter += 0;
			}
			bgfx::frame();
		}

		XrGraphicsBindingD3D11KHR m_graphicsBinding{ XR_TYPE_GRAPHICS_BINDING_D3D11_KHR };


		std::unordered_map<uintptr_t, bgfx::FrameBufferHandle> framebuffers;
		std::unordered_map<uintptr_t, bgfx::TextureHandle> textures;
		uint16_t counter = 0;
	};

	struct SwapchainImageXR {
		XrSwapchainImageBaseHeader* image;
		XrSwapchain handle;
	};

	struct FrameInfo {
		XrFrameState frameState;
		std::vector<XrCompositionLayerBaseHeader*> layers;
		XrCompositionLayerProjection layer{ XR_TYPE_COMPOSITION_LAYER_PROJECTION };
		std::vector<XrCompositionLayerProjectionView> projectionLayerViews;
		std::vector<SwapchainImageXR> images;
		XrViewState viewState{ XR_TYPE_VIEW_STATE };
		uint32_t viewCapacityInput;
		uint32_t viewCountOutput;
		XrViewLocateInfo viewLocateInfo{ XR_TYPE_VIEW_LOCATE_INFO };
		bool shouldRender = false;
	};

	class OpenXRLib {
	public:
		OpenXRLib() {}

		void init(const std::vector<std::string> platformExtensions = {},
			const std::vector<std::string> graphicsExtensions = { XR_KHR_D3D11_ENABLE_EXTENSION_NAME },
			XrBaseInStructure* createInstanceExtension = nullptr, XrFormFactor formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY,
			// XrViewConfigurationType configurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO,
			// XrEnvironmentBlendMode blendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE,
			const char* xrApplicationName = "XR App") {
			// TODO enumerate layers and extensions if needed

			CHECK(m_instance == XR_NULL_HANDLE);

			// Create union of extensions required by platform and graphics plugins.
			std::vector<const char*> extensions;

			// Transform platform and graphics extension std::strings to C strings.
			std::transform(platformExtensions.begin(), platformExtensions.end(), std::back_inserter(extensions),
				[](const std::string& ext) { return ext.c_str(); });
			std::transform(graphicsExtensions.begin(), graphicsExtensions.end(), std::back_inserter(extensions),
				[](const std::string& ext) { return ext.c_str(); });

			XrInstanceCreateInfo createInfo{ XR_TYPE_INSTANCE_CREATE_INFO };
			createInfo.next = createInstanceExtension;
			createInfo.enabledExtensionCount = (uint32_t)extensions.size();
			createInfo.enabledExtensionNames = extensions.data();

			strcpy(createInfo.applicationInfo.applicationName, xrApplicationName);
			createInfo.applicationInfo.apiVersion = XR_CURRENT_API_VERSION;

			CHECK_XRCMD(xrCreateInstance(&createInfo, &m_instance));

			// Log instance info
			CHECK(m_instance != XR_NULL_HANDLE);

			XrInstanceProperties instanceProperties{ XR_TYPE_INSTANCE_PROPERTIES };
			CHECK_XRCMD(xrGetInstanceProperties(m_instance, &instanceProperties));

			Log::Write(Log::Level::Info, Fmt("Instance RuntimeName=%s RuntimeVersion=%s", instanceProperties.runtimeName,
				GetXrVersionString(instanceProperties.runtimeVersion).c_str()));

			// Initialize system
			XrSystemGetInfo systemInfo{ XR_TYPE_SYSTEM_GET_INFO };
			systemInfo.formFactor = formFactor;
			CHECK_XRCMD(xrGetSystem(m_instance, &systemInfo, &m_systemId));
			CHECK(m_instance != XR_NULL_HANDLE);
			CHECK(m_systemId != XR_NULL_SYSTEM_ID);

			// TODO Init gfx device now using m_instance and m_systemId
		}

		bool isSessionRunning() {
			switch (m_sessionState) {
			case XR_SESSION_STATE_RUNNING:
			case XR_SESSION_STATE_VISIBLE:
			case XR_SESSION_STATE_FOCUSED:
				return true;
			}
			return false;
		}

		void initializeSession(void* graphicsBinding) {
			CHECK(m_instance != XR_NULL_HANDLE);
			CHECK(m_session == XR_NULL_HANDLE);

			// Create session
			{
				Log::Write(Log::Level::Verbose, Fmt("Creating session..."));

				XrSessionCreateInfo createInfo{ XR_TYPE_SESSION_CREATE_INFO };


				createInfo.next = graphicsBinding;

				createInfo.systemId = m_systemId;
				CHECK_XRCMD(xrCreateSession(m_instance, &createInfo, &m_session));
			}

			// Log reference space
			CHECK(m_session != XR_NULL_HANDLE);
			uint32_t spaceCount;
			CHECK_XRCMD(xrEnumerateReferenceSpaces(m_session, 0, &spaceCount, nullptr));
			std::vector<XrReferenceSpaceType> spaces(spaceCount);
			CHECK_XRCMD(xrEnumerateReferenceSpaces(m_session, spaceCount, &spaceCount, spaces.data()));
			Log::Write(Log::Level::Info, Fmt("Available reference spaces: %d", spaceCount));
			for (XrReferenceSpaceType space : spaces) {
				Log::Write(Log::Level::Verbose, Fmt("  Name: %s", GetXrReferenceSpaceTypeString(space).c_str()));
			}

			// CreateVisualizedSpaces
			CHECK(m_session != XR_NULL_HANDLE);

			XrReferenceSpaceType visualizedSpaces[] = { XR_REFERENCE_SPACE_TYPE_VIEW, XR_REFERENCE_SPACE_TYPE_LOCAL,
													   XR_REFERENCE_SPACE_TYPE_STAGE };

			for (auto visualizedSpace : visualizedSpaces) {
				XrReferenceSpaceCreateInfo referenceSpaceCreateInfo{ XR_TYPE_REFERENCE_SPACE_CREATE_INFO };
				XrPosef t{};
				t.orientation.w = 1;
				referenceSpaceCreateInfo.poseInReferenceSpace = t;
				referenceSpaceCreateInfo.referenceSpaceType = visualizedSpace;

				XrSpace space;
				XrResult res = xrCreateReferenceSpace(m_session, &referenceSpaceCreateInfo, &space);
				if (XR_SUCCEEDED(res)) {
					m_visualizedSpaces.push_back(space);
				}
				else {
					Log::Write(Log::Level::Warning,
						Fmt("Failed to create one of the reference spaces with error %d for visualization", res));
				}
			}

			// Set the app space
			{
				XrReferenceSpaceCreateInfo referenceSpaceCreateInfo{ XR_TYPE_REFERENCE_SPACE_CREATE_INFO };
				XrPosef t{};
				t.orientation.w = 1;
				referenceSpaceCreateInfo.poseInReferenceSpace = t;
				referenceSpaceCreateInfo.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
				CHECK_XRCMD(xrCreateReferenceSpace(m_session, &referenceSpaceCreateInfo, &m_appSpace));
			}
		}

		void createSwapchains(BGFXApp& app, XrStructureType swapchainType) {
			CHECK(m_session != XR_NULL_HANDLE);
			CHECK(m_swapchains.size() == 0);
			CHECK(m_configViews.empty());

			// Read graphics properties for preferred swapchain length and logging.
			XrSystemProperties systemProperties{ XR_TYPE_SYSTEM_PROPERTIES };
			CHECK_XRCMD(xrGetSystemProperties(m_instance, m_systemId, &systemProperties));

			// Log system properties.
			Log::Write(Log::Level::Info,
				Fmt("System Properties: Name=%s VendorId=%d", systemProperties.systemName, systemProperties.vendorId));
			Log::Write(Log::Level::Info, Fmt("System Graphics Properties: MaxWidth=%d MaxHeight=%d MaxViews=%d",
				systemProperties.graphicsProperties.maxSwapchainImageWidth,
				systemProperties.graphicsProperties.maxSwapchainImageHeight,
				systemProperties.graphicsProperties.maxViewCount));
			Log::Write(Log::Level::Info, Fmt("System Tracking Properties: OrientationTracking=%s PositionTracking=%s",
				systemProperties.trackingProperties.orientationTracking ? "True" : "False",
				systemProperties.trackingProperties.positionTracking ? "True" : "False"));

			// Note: No other view configurations exist at the time this code was written. If this condition
			// is not met, the project will need to be audited to see how support should be added.
			CHECK_MSG(m_viewConfigType == XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, "Unsupported view configuration type");

			// Query and cache view configuration views.
			uint32_t viewCount;
			CHECK_XRCMD(xrEnumerateViewConfigurationViews(m_instance, m_systemId, m_viewConfigType, 0, &viewCount, nullptr));
			m_configViews.resize(viewCount, { XR_TYPE_VIEW_CONFIGURATION_VIEW });
			CHECK_XRCMD(xrEnumerateViewConfigurationViews(m_instance, m_systemId, m_viewConfigType, viewCount, &viewCount,
				m_configViews.data()));

			// Create and cache view buffer for xrLocateViews later.
			m_views.resize(viewCount, { XR_TYPE_VIEW });

			// Create the swapchain and get the images.
			if (viewCount > 0) {
				// Select a swapchain format.
				uint32_t swapchainFormatCount;
				CHECK_XRCMD(xrEnumerateSwapchainFormats(m_session, 0, &swapchainFormatCount, nullptr));
				std::vector<int64_t> swapchainFormats(swapchainFormatCount);
				CHECK_XRCMD(xrEnumerateSwapchainFormats(m_session, (uint32_t)swapchainFormats.size(), &swapchainFormatCount,
					swapchainFormats.data()));
				CHECK(swapchainFormatCount == swapchainFormats.size());
				m_colorSwapchainFormat = app.selectColorSwapchainFormat(swapchainFormats);

				// Print swapchain formats and the selected one.
				{
					std::string swapchainFormatsString;
					for (int64_t format : swapchainFormats) {
						const bool selected = format == m_colorSwapchainFormat;
						swapchainFormatsString += " ";
						if (selected) swapchainFormatsString += "[";
						swapchainFormatsString += std::to_string(format);
						if (selected) swapchainFormatsString += "]";
					}
					Log::Write(Log::Level::Verbose, Fmt("Swapchain Formats:%s", swapchainFormatsString.c_str()));
				}

				// Create a swapchain for each view.
				for (uint32_t i = 0; i < viewCount; i++) {
					const XrViewConfigurationView& vp = m_configViews[i];
					Log::Write(Log::Level::Info,
						Fmt("Creating swapchain for view %d with dimensions Width=%d Height=%d SampleCount=%d", i,
							vp.recommendedImageRectWidth, vp.recommendedImageRectHeight, vp.recommendedSwapchainSampleCount));

					// Create the swapchain.
					XrSwapchainCreateInfo swapchainCreateInfo{ XR_TYPE_SWAPCHAIN_CREATE_INFO };
					swapchainCreateInfo.arraySize = 1;
					swapchainCreateInfo.format = m_colorSwapchainFormat;
					swapchainCreateInfo.width = vp.recommendedImageRectWidth;
					swapchainCreateInfo.height = vp.recommendedImageRectHeight;
					swapchainCreateInfo.mipCount = 1;
					swapchainCreateInfo.faceCount = 1;
					swapchainCreateInfo.sampleCount = vp.recommendedSwapchainSampleCount;
					swapchainCreateInfo.usageFlags = XR_SWAPCHAIN_USAGE_SAMPLED_BIT | XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT;
					Swapchain swapchain;
					swapchain.width = swapchainCreateInfo.width;
					swapchain.height = swapchainCreateInfo.height;
					CHECK_XRCMD(xrCreateSwapchain(m_session, &swapchainCreateInfo, &swapchain.handle));

					m_swapchains.push_back(swapchain);

					uint32_t imageCount;
					CHECK_XRCMD(xrEnumerateSwapchainImages(swapchain.handle, 0, &imageCount, nullptr));
					// XXX This should really just return XrSwapchainImageBaseHeader*
					std::vector<XrSwapchainImageBaseHeader*> swapchainImages =
						allocateSwapchainImageStructs(imageCount, swapchainCreateInfo, swapchainType);
					CHECK_XRCMD(xrEnumerateSwapchainImages(swapchain.handle, imageCount, &imageCount, swapchainImages[0]));

					m_swapchainImages.insert(std::make_pair(swapchain.handle, std::move(swapchainImages)));
				}
			}
		}

		std::vector<XrSwapchainImageBaseHeader*> allocateSwapchainImageStructs(uint32_t capacity,
			const XrSwapchainCreateInfo& /*swapchainCreateInfo*/,
			XrStructureType swapchainType) {
			if (swapchainType == XR_TYPE_SWAPCHAIN_IMAGE_D3D11_KHR) {
				// Allocate and initialize the buffer of image structs (must be sequential in memory for
				// xrEnumerateSwapchainImages).
				// Return back an array of pointers to each swapchain image struct so the consumer doesn't need to know the
				// type/size.
				std::vector<XrSwapchainImageD3D11KHR> swapchainImageBuffer(capacity);
				std::vector<XrSwapchainImageBaseHeader*> swapchainImageBase;
				for (XrSwapchainImageD3D11KHR& image : swapchainImageBuffer) {
					image.type = XR_TYPE_SWAPCHAIN_IMAGE_D3D11_KHR;
					swapchainImageBase.push_back(reinterpret_cast<XrSwapchainImageBaseHeader*>(&image));
				}

				// Keep the buffer alive by moving it into the list of buffers.
				m_swapchainImageBuffers.push_back(std::move(swapchainImageBuffer));

				return swapchainImageBase;
			}
			else {
				throw "Invalid swapchain type for allocateSwapchainImageStructs";
			}

		}

		// Return event if one is available, otherwise return null.
		const XrEventDataBaseHeader* TryReadNextEvent() {
			// It is sufficient to clear the just the XrEventDataBuffer header to XR_TYPE_EVENT_DATA_BUFFER
			XrEventDataBaseHeader* baseHeader = reinterpret_cast<XrEventDataBaseHeader*>(&m_eventDataBuffer);
			*baseHeader = { XR_TYPE_EVENT_DATA_BUFFER };
			const XrResult xr = xrPollEvent(m_instance, &m_eventDataBuffer);
			if (xr == XR_SUCCESS) {
				if (baseHeader->type == XR_TYPE_EVENT_DATA_EVENTS_LOST) {
					const XrEventDataEventsLost* const eventsLost = reinterpret_cast<const XrEventDataEventsLost*>(baseHeader);
					Log::Write(Log::Level::Warning, Fmt("%d events lost", eventsLost));
				}

				return baseHeader;
			}
			else if (xr == XR_EVENT_UNAVAILABLE) {
				return nullptr;
			}
			else {
				throw "xrPollEvent";
				//THROW_XR(xr, "xrPollEvent");
			}
		}

		void ManageSession(const XrEventDataSessionStateChanged& lifecycle, bool* exitRenderLoop, bool* requestRestart) {
			static std::map<XrSessionState, const std::string> stateName = {
				{XR_SESSION_STATE_UNKNOWN, "UNKNOWN"},   {XR_SESSION_STATE_IDLE, "IDLE"},
				{XR_SESSION_STATE_READY, "READY"},       {XR_SESSION_STATE_RUNNING, "RUNNING"},
				{XR_SESSION_STATE_VISIBLE, "VISIBLE"},   {XR_SESSION_STATE_FOCUSED, "FOCUSED"},
				{XR_SESSION_STATE_STOPPING, "STOPPING"}, {XR_SESSION_STATE_LOSS_PENDING, "LOSS_PENDING"},
				{XR_SESSION_STATE_EXITING, "EXITING"},
			};

			XrSessionState oldState = m_sessionState;
			m_sessionState = lifecycle.state;

			const std::string& oldStateName = stateName[oldState];
			const std::string& newStateName = stateName[m_sessionState];
			Log::Write(Log::Level::Info, Fmt("XrEventDataSessionStateChanged: state %s->%s session=%lld time=%lld",
				oldStateName.c_str(), newStateName.c_str(), lifecycle.session, lifecycle.time));

			if (lifecycle.session && (lifecycle.session != m_session)) {
				Log::Write(Log::Level::Error, "XrEventDataSessionStateChanged for unknown session");
				return;
			}

			switch (m_sessionState) {
			case XR_SESSION_STATE_READY: {
				CHECK(m_session != XR_NULL_HANDLE);
				XrSessionBeginInfo sessionBeginInfo{ XR_TYPE_SESSION_BEGIN_INFO };
				sessionBeginInfo.primaryViewConfigurationType = m_viewConfigType;
				XrResult res = xrBeginSession(m_session, &sessionBeginInfo);
				if (res == XR_SESSION_VISIBILITY_UNAVAILABLE) {
					Log::Write(Log::Level::Warning, "xrBeginSession returned XR_SESSION_VISIBILITY_UNAVAILABLE");
				}
				else {
					CHECK_XRRESULT(res, "xrBeginSession");
				}
				break;
			}
			case XR_SESSION_STATE_STOPPING: {
				CHECK(m_session != XR_NULL_HANDLE);
				CHECK_XRCMD(xrEndSession(m_session))
					break;
			}
			case XR_SESSION_STATE_EXITING: {
				*exitRenderLoop = true;
				// Do not attempt to restart because user closed this session.
				*requestRestart = false;
				break;
			}
			case XR_SESSION_STATE_LOSS_PENDING: {
				*exitRenderLoop = true;
				// Poll for a new instance
				*requestRestart = true;
				break;
			}
			}
		}

		void pollEvents(bool* exitRenderLoop, bool* requestRestart) {
			*exitRenderLoop = *requestRestart = false;

			// Process all pending messages.
			while (const XrEventDataBaseHeader* event = TryReadNextEvent()) {
				switch (event->type) {
				case XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING: {
					const auto& instanceLossPending = *reinterpret_cast<const XrEventDataInstanceLossPending*>(event);
					Log::Write(Log::Level::Warning, Fmt("XrEventDataInstanceLossPending by %lld", instanceLossPending.lossTime));
					*exitRenderLoop = true;
					*requestRestart = true;
					return;
				}
				case XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED: {
					ManageSession(*reinterpret_cast<const XrEventDataSessionStateChanged*>(event), exitRenderLoop, requestRestart);
					break;
				}
				case XR_TYPE_EVENT_DATA_REFERENCE_SPACE_CHANGE_PENDING:
				case XR_TYPE_EVENT_DATA_INTERACTION_PROFILE_CHANGED:
				default: {
					Log::Write(Log::Level::Verbose, Fmt("Ignoring event type %d", event->type));
					break;
				}
				}
			}
		}

		bool isSessionVisible() {
			switch (m_sessionState) {
			case XR_SESSION_STATE_VISIBLE:
			case XR_SESSION_STATE_FOCUSED:
				return true;
			}
			return false;
		}






		void aquireImage(FrameInfo& frame, uint16_t i) {
			// Each view has a separate swapchain which is acquired, rendered to, and released.
			const Swapchain viewSwapchain = m_swapchains[i];
			frame.images[i].handle = viewSwapchain.handle;

			XrSwapchainImageAcquireInfo acquireInfo{ XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO };

			uint32_t swapchainImageIndex;
			CHECK_XRCMD(xrAcquireSwapchainImage(viewSwapchain.handle, &acquireInfo, &swapchainImageIndex));

			XrSwapchainImageWaitInfo waitInfo{ XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO };
			waitInfo.timeout = XR_INFINITE_DURATION;
			CHECK_XRCMD(xrWaitSwapchainImage(viewSwapchain.handle, &waitInfo));

			frame.projectionLayerViews[i] = { XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW };
			frame.projectionLayerViews[i].pose = m_views[i].pose;
			frame.projectionLayerViews[i].fov = m_views[i].fov;
			frame.projectionLayerViews[i].subImage.swapchain = viewSwapchain.handle;
			frame.projectionLayerViews[i].subImage.imageRect.offset = { 0, 0 };
			frame.projectionLayerViews[i].subImage.imageRect.extent = { viewSwapchain.width, viewSwapchain.height };
			frame.images[i].image = m_swapchainImages[viewSwapchain.handle][swapchainImageIndex];
		}

		void releaseImage(FrameInfo& frame, uint16_t i) {
			XrSwapchainImageReleaseInfo releaseInfo{ XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO };
			CHECK_XRCMD(xrReleaseSwapchainImage(frame.images[i].handle, &releaseInfo));
		}

		FrameInfo aquireFrame() {
			XrResult res;
			FrameInfo frame;

			// Sync with frame and begin frame
			CHECK(m_session != XR_NULL_HANDLE);
			XrFrameWaitInfo frameWaitInfo{ XR_TYPE_FRAME_WAIT_INFO };
			XrFrameState frameState{ XR_TYPE_FRAME_STATE };
			CHECK_XRCMD(xrWaitFrame(m_session, &frameWaitInfo, &frameState));
			XrFrameBeginInfo frameBeginInfo{ XR_TYPE_FRAME_BEGIN_INFO };
			CHECK_XRCMD(xrBeginFrame(m_session, &frameBeginInfo));

			frame.frameState = frameState;
			frame.viewCapacityInput = (uint32_t)m_views.size();
			frame.viewLocateInfo.displayTime = frame.frameState.predictedDisplayTime;
			frame.viewLocateInfo.space = m_appSpace;

			res = xrLocateViews(m_session, &frame.viewLocateInfo, &frame.viewState, frame.viewCapacityInput, &frame.viewCountOutput,
				m_views.data());
			CHECK_XRRESULT(res, "xrLocateViews");

			frame.shouldRender = XR_UNQUALIFIED_SUCCESS(res) && isSessionVisible();
			CHECK(frame.viewCountOutput == frame.viewCapacityInput);
			CHECK(frame.viewCountOutput == m_configViews.size());
			CHECK(frame.viewCountOutput == m_swapchains.size());

			frame.projectionLayerViews.resize(frame.viewCountOutput);
			frame.images.resize(frame.viewCountOutput);


			return std::move(frame);
		}

		void submitFrame(FrameInfo frame) {
			if (frame.shouldRender) {
				frame.layer.space = m_appSpace;
				frame.layer.viewCount = (uint32_t)frame.projectionLayerViews.size();
				frame.layer.views = frame.projectionLayerViews.data();

				frame.layers.push_back(reinterpret_cast<XrCompositionLayerBaseHeader*>(&frame.layer));
			}

			XrFrameEndInfo frameEndInfo{ XR_TYPE_FRAME_END_INFO };
			frameEndInfo.displayTime = frame.frameState.predictedDisplayTime;
			frameEndInfo.environmentBlendMode = m_environmentBlendMode;
			frameEndInfo.layerCount = (uint32_t)frame.layers.size();
			frameEndInfo.layers = frame.layers.data();
			CHECK_XRCMD(xrEndFrame(m_session, &frameEndInfo));
		}

		static std::string version() { return "1"; }

		inline std::string GetXrReferenceSpaceTypeString(XrReferenceSpaceType referenceSpaceType) {
			if (referenceSpaceType == XR_REFERENCE_SPACE_TYPE_VIEW)
				return "View";
			else if (referenceSpaceType == XR_REFERENCE_SPACE_TYPE_LOCAL)
				return "Local";
			else if (referenceSpaceType == XR_REFERENCE_SPACE_TYPE_STAGE)
				return "Stage";
			return "Unknown";
		}

		inline std::string GetXrVersionString(uint32_t ver) {
			return Fmt("%d.%d.%d", XR_VERSION_MAJOR(ver), XR_VERSION_MINOR(ver), XR_VERSION_PATCH(ver));
		}


		XrSession m_session{ XR_NULL_HANDLE };
		XrSystemId m_systemId{ XR_NULL_SYSTEM_ID };
		XrInstance m_instance{ XR_NULL_HANDLE };
		XrSpace m_appSpace;
		XrEventDataBuffer m_eventDataBuffer;
		std::vector<XrSpace> m_visualizedSpaces;
		struct Swapchain {
			XrSwapchain handle;
			int32_t width;
			int32_t height;
		};
		std::vector<Swapchain> m_swapchains;
		std::vector<XrViewConfigurationView> m_configViews;
		std::vector<XrView> m_views;
		int64_t m_colorSwapchainFormat{ -1 };
		std::map<XrSwapchain, std::vector<XrSwapchainImageBaseHeader*>> m_swapchainImages;
		XrViewConfigurationType m_viewConfigType{ XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO };
		// Application's current lifecycle state according to the runtime
		XrSessionState m_sessionState{ XR_SESSION_STATE_UNKNOWN };
		XrEnvironmentBlendMode m_environmentBlendMode{ XR_ENVIRONMENT_BLEND_MODE_OPAQUE };
		std::list<std::vector<XrSwapchainImageD3D11KHR>> m_swapchainImageBuffers;

	};

    class NativeEngine::Impl final
    {
    public:
        Impl(void* nativeWindowPtr, RuntimeImpl& runtimeImpl);

        void Initialize(Napi::Env& env);
        void UpdateSize(float width, float height);
        void UpdateRenderTarget();
        void Suspend();

    private:
        using EngineDefiner = NativeEngineDefiner<NativeEngine::Impl>;
        friend EngineDefiner;

        struct VertexArray final
        {
            struct IndexBuffer
            {
                bgfx::IndexBufferHandle handle;
            };

            IndexBuffer indexBuffer;

            struct VertexBuffer
            {
                bgfx::VertexBufferHandle handle;
                uint32_t startVertex;
                bgfx::VertexDeclHandle declHandle;
            };

            std::vector<VertexBuffer> vertexBuffers;
        };

        enum BlendMode {}; // TODO DEBUG
        enum class Filter {}; // TODO DEBUG
        enum class AddressMode {}; // TODO DEBUG

        struct TextureData final
        {
            ~TextureData()
            {
                bgfx::destroy(Texture);

                for (auto image : Images)
                {
                    bimg::imageFree(image);
                }
            }

            std::vector<bimg::ImageContainer*> Images{};
            bgfx::TextureHandle Texture{ bgfx::kInvalidHandle };
        };

        struct ProgramData final
        {
            ~ProgramData()
            {
                bgfx::destroy(Program);
            }

            std::unordered_map<std::string, uint32_t> AttributeLocations{};
            std::unordered_map<std::string, UniformInfo> VertexUniformNameToInfo{};
            std::unordered_map<std::string, UniformInfo> FragmentUniformNameToInfo{};

            bgfx::ProgramHandle Program{};

            struct UniformValue
            {
                std::vector<float> Data{};
                uint16_t ElementLength{};
            };

            std::unordered_map<uint16_t, UniformValue> Uniforms{};

            void SetUniform(bgfx::UniformHandle handle, gsl::span<const float> data, size_t elementLength = 1)
            {
                UniformValue& value = Uniforms[handle.idx];
                value.Data.assign(data.begin(), data.end());
                value.ElementLength = static_cast<uint16_t>(elementLength);
            }
        };

        void RequestAnimationFrame(const Napi::CallbackInfo& info);
        Napi::Value CreateVertexArray(const Napi::CallbackInfo& info);
        void DeleteVertexArray(const Napi::CallbackInfo& info);
        void BindVertexArray(const Napi::CallbackInfo& info);
        Napi::Value CreateIndexBuffer(const Napi::CallbackInfo& info);
        void DeleteIndexBuffer(const Napi::CallbackInfo& info);
        void RecordIndexBuffer(const Napi::CallbackInfo& info);
        Napi::Value CreateVertexBuffer(const Napi::CallbackInfo& info);
        void DeleteVertexBuffer(const Napi::CallbackInfo& info);
        void RecordVertexBuffer(const Napi::CallbackInfo& info);
        Napi::Value CreateProgram(const Napi::CallbackInfo& info);
        Napi::Value GetUniforms(const Napi::CallbackInfo& info);
        Napi::Value GetAttributes(const Napi::CallbackInfo& info);
        void SetProgram(const Napi::CallbackInfo& info);
        void SetState(const Napi::CallbackInfo& info);
        void SetZOffset(const Napi::CallbackInfo& info);
        Napi::Value GetZOffset(const Napi::CallbackInfo& info);
        void SetDepthTest(const Napi::CallbackInfo& info);
        Napi::Value GetDepthWrite(const Napi::CallbackInfo& info);
        void SetDepthWrite(const Napi::CallbackInfo& info);
        void SetColorWrite(const Napi::CallbackInfo& info);
        void SetBlendMode(const Napi::CallbackInfo& info);
        void SetMatrix(const Napi::CallbackInfo& info);
        void SetIntArray(const Napi::CallbackInfo& info);
        void SetIntArray2(const Napi::CallbackInfo& info);
        void SetIntArray3(const Napi::CallbackInfo& info);
        void SetIntArray4(const Napi::CallbackInfo& info);
        void SetFloatArray(const Napi::CallbackInfo& info);
        void SetFloatArray2(const Napi::CallbackInfo& info);
        void SetFloatArray3(const Napi::CallbackInfo& info);
        void SetFloatArray4(const Napi::CallbackInfo& info);
        void SetMatrices(const Napi::CallbackInfo& info);
        void SetMatrix3x3(const Napi::CallbackInfo& info);
        void SetMatrix2x2(const Napi::CallbackInfo& info);
        void SetFloat(const Napi::CallbackInfo& info);
        void SetFloat2(const Napi::CallbackInfo& info);
        void SetFloat3(const Napi::CallbackInfo& info);
        void SetFloat4(const Napi::CallbackInfo& info);
        void SetBool(const Napi::CallbackInfo& info);
        Napi::Value CreateTexture(const Napi::CallbackInfo& info);
        void LoadTexture(const Napi::CallbackInfo& info);
        void LoadCubeTexture(const Napi::CallbackInfo& info);
        Napi::Value GetTextureWidth(const Napi::CallbackInfo& info);
        Napi::Value GetTextureHeight(const Napi::CallbackInfo& info);
        void SetTextureSampling(const Napi::CallbackInfo& info);
        void SetTextureWrapMode(const Napi::CallbackInfo& info);
        void SetTextureAnisotropicLevel(const Napi::CallbackInfo& info);
        void SetTexture(const Napi::CallbackInfo& info);
        void DeleteTexture(const Napi::CallbackInfo& info);
        void DrawIndexed(const Napi::CallbackInfo& info);
        void Draw(const Napi::CallbackInfo& info);
        void Clear(const Napi::CallbackInfo& info);
        Napi::Value GetRenderWidth(const Napi::CallbackInfo& info);
        Napi::Value GetRenderHeight(const Napi::CallbackInfo& info);

        void DispatchAnimationFrameAsync(Napi::FunctionReference callback);

        ShaderCompiler m_shaderCompiler;

        ProgramData* m_currentProgram;

        RuntimeImpl& m_runtimeImpl;

        struct
        {
            uint32_t Width{};
            uint32_t Height{};
        } m_size;

        bx::DefaultAllocator m_allocator;
        uint64_t m_engineState;

        // Scratch vector used for data alignment.
        std::vector<float> m_scratch;
    };

    NativeEngine::Impl::Impl(void* nativeWindowPtr, RuntimeImpl& runtimeImpl)
        : m_runtimeImpl{ runtimeImpl }
        , m_currentProgram{ nullptr }
        , m_size{ 1024, 768 }
        , m_engineState{ BGFX_STATE_DEFAULT }
    {
		auto xr = OpenXRLib();

		auto bgfxApp = BGFXApp();
		xr.init();
		bgfxApp.initGraphicsDevice(xr.m_instance, xr.m_systemId);
		xr.initializeSession(&bgfxApp.m_graphicsBinding);
		xr.createSwapchains(bgfxApp, XR_TYPE_SWAPCHAIN_IMAGE_D3D11_KHR);

		bool exit = false;
		bool restart = false;
		while (1) {
			xr.pollEvents(&exit, &restart);
			auto frame = xr.aquireFrame();
			if (frame.shouldRender) {
				for (uint16_t i = 0; i < frame.viewCountOutput; i++) {
					xr.aquireImage(frame, i);
					bgfxApp.renderView(frame.projectionLayerViews[i], frame.images[i].image,
						xr.m_colorSwapchainFormat);
					xr.releaseImage(frame, i);
				}

				OutputDebugStringA("new frame!");
			}
			else {
				// Throttle loop since xrWaitFrame won't be called.
				std::this_thread::sleep_for(std::chrono::milliseconds(250));
			}
			xr.submitFrame(std::move(frame));
		}

        bgfx::Init init{};
        init.platformData.nwh = nativeWindowPtr;
        bgfx::setPlatformData(init.platformData);

        init.type = bgfx::RendererType::Direct3D11;
        init.resolution.width = m_size.Width;
        init.resolution.height = m_size.Height;
        init.resolution.reset = BGFX_RESET_VSYNC;
        bgfx::init(init);

        bgfx::setViewClear(0, BGFX_CLEAR_COLOR | BGFX_CLEAR_DEPTH, 0x443355FF, 1.0f, 0);
        bgfx::setViewRect(0, 0, 0, m_size.Width, m_size.Height);
    }

    void NativeEngine::Impl::Initialize(Napi::Env& env)
    {
        EngineDefiner::Define(env, this);
    }

    void NativeEngine::Impl::UpdateSize(float width, float height)
    {
        auto w = static_cast<uint32_t>(width);
        auto h = static_cast<uint32_t>(height);

        if (w != m_size.Width || h != m_size.Height)
        {
            m_size = { w, h };
            UpdateRenderTarget();
        }
    }

    void NativeEngine::Impl::UpdateRenderTarget()
    {
        bgfx::reset(m_size.Width, m_size.Height, BGFX_RESET_VSYNC | BGFX_RESET_MSAA_X4);
        bgfx::setViewRect(0, 0, 0, m_size.Width, m_size.Height);
    }

    void NativeEngine::Impl::Suspend()
    {
        // TODO: Figure out what this is supposed to do.
    }

    // NativeEngine definitions

    void NativeEngine::Impl::RequestAnimationFrame(const Napi::CallbackInfo& info)
    {
        DispatchAnimationFrameAsync(Napi::Persistent(info[0].As<Napi::Function>()));
    }

    Napi::Value NativeEngine::Impl::CreateVertexArray(const Napi::CallbackInfo& info)
    {
        return Napi::External<VertexArray>::New(info.Env(), new VertexArray{});
    }

    void NativeEngine::Impl::DeleteVertexArray(const Napi::CallbackInfo& info)
    {
        delete info[0].As<Napi::External<VertexArray>>().Data();
    }

    void NativeEngine::Impl::BindVertexArray(const Napi::CallbackInfo& info)
    {
        const auto& vertexArray = *(info[0].As<Napi::External<VertexArray>>().Data());

        bgfx::setIndexBuffer(vertexArray.indexBuffer.handle);

        const auto& vertexBuffers = vertexArray.vertexBuffers;
        for (uint8_t index = 0; index < vertexBuffers.size(); ++index)
        {
            const auto& vertexBuffer = vertexBuffers[index];
            bgfx::setVertexBuffer(index, vertexBuffer.handle, vertexBuffer.startVertex, UINT32_MAX, vertexBuffer.declHandle);
        }
    }

    Napi::Value NativeEngine::Impl::CreateIndexBuffer(const Napi::CallbackInfo& info)
    {
        const Napi::TypedArray data = info[0].As<Napi::TypedArray>();
        const bgfx::Memory* ref = bgfx::makeRef(data.As<Napi::Uint8Array>().Data(), static_cast<uint32_t>(data.ByteLength()));
        const uint16_t flags = data.TypedArrayType() == napi_typedarray_type::napi_uint16_array ? 0 : BGFX_BUFFER_INDEX32;
        const bgfx::IndexBufferHandle handle = bgfx::createIndexBuffer(ref, flags);
        return Napi::Value::From(info.Env(), static_cast<uint32_t>(handle.idx));
    }

    void NativeEngine::Impl::DeleteIndexBuffer(const Napi::CallbackInfo& info)
    {
        const bgfx::IndexBufferHandle handle{ static_cast<uint16_t>(info[0].As<Napi::Number>().Uint32Value()) };
        bgfx::destroy(handle);
    }

    void NativeEngine::Impl::RecordIndexBuffer(const Napi::CallbackInfo& info)
    {
        VertexArray& vertexArray = *(info[0].As<Napi::External<VertexArray>>().Data());
        const bgfx::IndexBufferHandle handle{ static_cast<uint16_t>(info[1].As<Napi::Number>().Uint32Value()) };
        vertexArray.indexBuffer.handle = handle;
    }

    Napi::Value NativeEngine::Impl::CreateVertexBuffer(const Napi::CallbackInfo& info)
    {
        const Napi::Uint8Array data = info[0].As<Napi::Uint8Array>();

        // HACK: Create an empty valid vertex decl which will never be used. Consider fixing in bgfx.
        bgfx::VertexDecl decl;
        decl.begin();
        decl.m_stride = 1;
        decl.end();

        const bgfx::Memory* ref = bgfx::copy(data.Data(), static_cast<uint32_t>(data.ByteLength()));
        const bgfx::VertexBufferHandle handle = bgfx::createVertexBuffer(ref, decl);
        return Napi::Value::From(info.Env(), static_cast<uint32_t>(handle.idx));
    }

    void NativeEngine::Impl::DeleteVertexBuffer(const Napi::CallbackInfo& info)
    {
        const bgfx::VertexBufferHandle handle{ static_cast<uint16_t>(info[0].As<Napi::Number>().Uint32Value()) };
        bgfx::destroy(handle);
    }

    void NativeEngine::Impl::RecordVertexBuffer(const Napi::CallbackInfo& info)
    {
        VertexArray& vertexArray = *(info[0].As<Napi::External<VertexArray>>().Data());
        const bgfx::VertexBufferHandle handle{ static_cast<uint16_t>(info[1].As<Napi::Number>().Uint32Value()) };
        const uint32_t location = info[2].As<Napi::Number>().Uint32Value();
        const uint32_t byteOffset = info[3].As<Napi::Number>().Uint32Value();
        const uint32_t byteStride = info[4].As<Napi::Number>().Uint32Value();
        const uint32_t numElements = info[5].As<Napi::Number>().Uint32Value();
        const uint32_t type = info[6].As<Napi::Number>().Uint32Value();
        const bool normalized = info[7].As<Napi::Boolean>().Value();

        bgfx::VertexDecl decl;
        decl.begin();
        const bgfx::Attrib::Enum attrib = static_cast<bgfx::Attrib::Enum>(location);
        const bgfx::AttribType::Enum attribType = ConvertAttribType(static_cast<WebGLAttribType>(type));
        decl.add(attrib, numElements, attribType, normalized);
        decl.m_stride = static_cast<uint16_t>(byteStride);
        decl.end();

        vertexArray.vertexBuffers.push_back({ std::move(handle), byteOffset / byteStride, bgfx::createVertexDecl(decl) });
    }

    Napi::Value NativeEngine::Impl::CreateProgram(const Napi::CallbackInfo& info)
    {
        const auto vertexSource = info[0].As<Napi::String>().Utf8Value();
        // TODO: This is a HACK to account for the fact that DirectX and OpenGL disagree about the vertical orientation of screen space.
        // Remove this ASAP when we have a more long-term plan to account for this behavior.
        const auto fragmentSource = std::regex_replace(info[1].As<Napi::String>().Utf8Value(), std::regex("dFdy\\("), "-dFdy(");

        auto programData = new ProgramData();

        std::vector<uint8_t> vertexBytes{};
        std::vector<uint8_t> fragmentBytes{};
        std::unordered_map<std::string, uint32_t> attributeLocations;

        m_shaderCompiler.Compile(vertexSource, fragmentSource, [&](ShaderCompiler::ShaderInfo vertexShaderInfo, ShaderCompiler::ShaderInfo fragmentShaderInfo)
        {
            constexpr uint8_t BGFX_SHADER_BIN_VERSION = 6;

            // These hashes are generated internally by BGFX's custom shader compilation pipeline,
            // which we don't have access to.  Fortunately, however, they aren't used for anything
            // crucial; they just have to match.
            constexpr uint32_t vertexOutputsHash = 0xBAD1DEA;
            constexpr uint32_t fragmentInputsHash = vertexOutputsHash;

            {
                const spirv_cross::Compiler& compiler = *vertexShaderInfo.Compiler;
                const spirv_cross::ShaderResources resources = compiler.get_shader_resources();
                assert(resources.uniform_buffers.size() == 1);
                const spirv_cross::Resource& uniformBuffer = resources.uniform_buffers[0];
                const spirv_cross::SmallVector<spirv_cross::Resource>& samplers = resources.separate_samplers;
                size_t numUniforms = compiler.get_type(uniformBuffer.base_type_id).member_types.size() + samplers.size();

                AppendBytes(vertexBytes, BX_MAKEFOURCC('V', 'S', 'H', BGFX_SHADER_BIN_VERSION));
                AppendBytes(vertexBytes, vertexOutputsHash);
                AppendBytes(vertexBytes, fragmentInputsHash);

                AppendBytes(vertexBytes, static_cast<uint16_t>(numUniforms));
                AppendUniformBuffer(vertexBytes, compiler, uniformBuffer, false);
                AppendSamplers(vertexBytes, compiler, samplers, false, programData->VertexUniformNameToInfo);

                AppendBytes(vertexBytes, static_cast<uint32_t>(vertexShaderInfo.Bytes.size()));
                AppendBytes(vertexBytes, vertexShaderInfo.Bytes);
                AppendBytes(vertexBytes, static_cast<uint8_t>(0));

                AppendBytes(vertexBytes, static_cast<uint8_t>(resources.stage_inputs.size()));
                for (const spirv_cross::Resource& stageInput : resources.stage_inputs)
                {
                    const uint32_t location = compiler.get_decoration(stageInput.id, spv::DecorationLocation);
                    AppendBytes(vertexBytes, bgfx::attribToId(static_cast<bgfx::Attrib::Enum>(location)));
                    attributeLocations[stageInput.name] = location;
                }

                AppendBytes(vertexBytes, static_cast<uint16_t>(compiler.get_declared_struct_size(compiler.get_type(uniformBuffer.base_type_id))));
            }

            {
                const spirv_cross::Compiler& compiler = *fragmentShaderInfo.Compiler;
                const spirv_cross::ShaderResources resources = compiler.get_shader_resources();
                assert(resources.uniform_buffers.size() == 1);
                const spirv_cross::Resource& uniformBuffer = resources.uniform_buffers[0];
                const spirv_cross::SmallVector<spirv_cross::Resource>& samplers = resources.separate_samplers;
                size_t numUniforms = compiler.get_type(uniformBuffer.base_type_id).member_types.size() + samplers.size();

                AppendBytes(fragmentBytes, BX_MAKEFOURCC('F', 'S', 'H', BGFX_SHADER_BIN_VERSION));
                AppendBytes(fragmentBytes, vertexOutputsHash);
                AppendBytes(fragmentBytes, fragmentInputsHash);

                AppendBytes(fragmentBytes, static_cast<uint16_t>(numUniforms));
                AppendUniformBuffer(fragmentBytes, compiler, uniformBuffer, true);
                AppendSamplers(fragmentBytes, compiler, samplers, true, programData->FragmentUniformNameToInfo);

                AppendBytes(fragmentBytes, static_cast<uint32_t>(fragmentShaderInfo.Bytes.size()));
                AppendBytes(fragmentBytes, fragmentShaderInfo.Bytes);
                AppendBytes(fragmentBytes, static_cast<uint8_t>(0));

                // Fragment shaders don't have attributes.
                AppendBytes(fragmentBytes, static_cast<uint8_t>(0));

                AppendBytes(fragmentBytes, static_cast<uint16_t>(compiler.get_declared_struct_size(compiler.get_type(uniformBuffer.base_type_id))));
            }
        });

        auto vertexShader = bgfx::createShader(bgfx::copy(vertexBytes.data(), static_cast<uint32_t>(vertexBytes.size())));
        CacheUniformHandles(vertexShader, programData->VertexUniformNameToInfo);
        programData->AttributeLocations = std::move(attributeLocations);

        auto fragmentShader = bgfx::createShader(bgfx::copy(fragmentBytes.data(), static_cast<uint32_t>(fragmentBytes.size())));
        CacheUniformHandles(fragmentShader, programData->FragmentUniformNameToInfo);

        programData->Program = bgfx::createProgram(vertexShader, fragmentShader, true);

        auto finalizer = [](Napi::Env, ProgramData* data)
        {
            delete data;
        };

        return Napi::External<ProgramData>::New(info.Env(), programData, finalizer);
    }

    Napi::Value NativeEngine::Impl::GetUniforms(const Napi::CallbackInfo& info)
    {
        const auto program = info[0].As<Napi::External<ProgramData>>().Data();
        const auto names = info[1].As<Napi::Array>();

        auto length = names.Length();
        auto uniforms = Napi::Array::New(info.Env(), length);
        for (uint32_t index = 0; index < length; ++index)
        {
            const auto name = names[index].As<Napi::String>().Utf8Value();

            auto vertexFound = program->VertexUniformNameToInfo.find(name);
            auto fragmentFound = program->FragmentUniformNameToInfo.find(name);

            if (vertexFound != program->VertexUniformNameToInfo.end())
            {
                uniforms[index] = Napi::External<UniformInfo>::New(info.Env(), &vertexFound->second);
            }
            else if (fragmentFound != program->FragmentUniformNameToInfo.end())
            {
                uniforms[index] = Napi::External<UniformInfo>::New(info.Env(), &fragmentFound->second);
            }
            else
            {
                uniforms[index] = info.Env().Null();
            }
        }

        return uniforms;
    }

    Napi::Value NativeEngine::Impl::GetAttributes(const Napi::CallbackInfo& info)
    {
        const auto program = info[0].As<Napi::External<ProgramData>>().Data();
        const auto names = info[1].As<Napi::Array>();

        const auto& attributeLocations = program->AttributeLocations;

        auto length = names.Length();
        auto attributes = Napi::Array::New(info.Env(), length);
        for (uint32_t index = 0; index < length; ++index)
        {
            const auto name = names[index].As<Napi::String>().Utf8Value();
            const auto it = attributeLocations.find(name);
            int location = (it == attributeLocations.end() ? -1 : gsl::narrow_cast<int>(it->second));
            attributes[index] = Napi::Value::From(info.Env(), location);
        }

        return attributes;
    }

    void NativeEngine::Impl::SetProgram(const Napi::CallbackInfo& info)
    {
        auto program = info[0].As<Napi::External<ProgramData>>().Data();
        m_currentProgram = program;
    }

    void NativeEngine::Impl::SetState(const Napi::CallbackInfo& info)
    {
        const auto culling = info[0].As<Napi::Boolean>().Value();
        const auto reverseSide = info[2].As<Napi::Boolean>().Value();

        m_engineState &= ~BGFX_STATE_CULL_MASK;
        if (reverseSide)
        {
            m_engineState &= ~BGFX_STATE_FRONT_CCW;

            if (culling)
            {
                m_engineState |= BGFX_STATE_CULL_CW;
            }
        }
        else
        {
            m_engineState |= BGFX_STATE_FRONT_CCW;

            if (culling)
            {
                m_engineState |= BGFX_STATE_CULL_CCW;
            }
        }

        // TODO: zOffset
        const auto zOffset = info[1].As<Napi::Number>().FloatValue();

        bgfx::setState(m_engineState);
    }

    void NativeEngine::Impl::SetZOffset(const Napi::CallbackInfo& info)
    {
        const auto zOffset = info[0].As<Napi::Number>().FloatValue();

        // STUB: Stub.
    }

    Napi::Value NativeEngine::Impl::GetZOffset(const Napi::CallbackInfo& info)
    {
        // STUB: Stub.
        return{};
    }

    void NativeEngine::Impl::SetDepthTest(const Napi::CallbackInfo& info)
    {
        const auto enable = info[0].As<Napi::Boolean>().Value();

        // STUB: Stub.
    }

    Napi::Value NativeEngine::Impl::GetDepthWrite(const Napi::CallbackInfo& info)
    {
        // STUB: Stub.
        return{};
    }

    void NativeEngine::Impl::SetDepthWrite(const Napi::CallbackInfo& info)
    {
        const auto enable = info[0].As<Napi::Boolean>().Value();

        // STUB: Stub.
    }

    void NativeEngine::Impl::SetColorWrite(const Napi::CallbackInfo& info)
    {
        const auto enable = info[0].As<Napi::Boolean>().Value();

        // STUB: Stub.
    }

    void NativeEngine::Impl::SetBlendMode(const Napi::CallbackInfo& info)
    {
        const auto blendMode = static_cast<BlendMode>(info[0].As<Napi::Number>().Int32Value());

        m_engineState &= ~BGFX_STATE_BLEND_MASK;
        m_engineState |= ALPHA_MODE[blendMode];

        bgfx::setState(m_engineState);
    }

    void NativeEngine::Impl::SetMatrix(const Napi::CallbackInfo& info)
    {
        const auto uniformData = info[0].As<Napi::External<UniformInfo>>().Data();
        const auto matrix = info[1].As<Napi::Float32Array>();

        const size_t elementLength = matrix.ElementLength();
        assert(elementLength == 16);

        m_currentProgram->SetUniform(uniformData->Handle, gsl::make_span(matrix.Data(), elementLength));
    }

    void NativeEngine::Impl::SetIntArray(const Napi::CallbackInfo& info)
    {
        // args: ShaderProperty property, gsl::span<const int> array

        assert(false);
    }

    void NativeEngine::Impl::SetIntArray2(const Napi::CallbackInfo& info)
    {
        // args: ShaderProperty property, gsl::span<const int> array

        assert(false);
    }

    void NativeEngine::Impl::SetIntArray3(const Napi::CallbackInfo& info)
    {
        // args: ShaderProperty property, gsl::span<const int> array

        assert(false);
    }

    void NativeEngine::Impl::SetIntArray4(const Napi::CallbackInfo& info)
    {
        // args: ShaderProperty property, gsl::span<const int> array

        assert(false);
    }

    void NativeEngine::Impl::SetFloatArray(const Napi::CallbackInfo& info)
    {
        const auto uniformData = info[0].As<Napi::External<UniformInfo>>().Data();
        const auto array = info[1].As<Napi::Float32Array>();

        size_t elementLength = array.ElementLength();

        m_scratch.clear();
        for (size_t index = 0; index < elementLength; ++index)
        {
            const float values[] = { array[index], 0.0f, 0.0f, 0.0f };
            m_scratch.insert(m_scratch.end(), values, values + 4);
        }

        m_currentProgram->SetUniform(uniformData->Handle, m_scratch, elementLength);
    }

    void NativeEngine::Impl::SetFloatArray2(const Napi::CallbackInfo& info)
    {
        // args: ShaderProperty property, gsl::span<const float> array

        assert(false);
    }

    void NativeEngine::Impl::SetFloatArray3(const Napi::CallbackInfo& info)
    {
        // args: ShaderProperty property, gsl::span<const float> array

        assert(false);
    }

    void NativeEngine::Impl::SetFloatArray4(const Napi::CallbackInfo& info)
    {
        // args: ShaderProperty property, gsl::span<const float> array

        assert(false);
    }

    void NativeEngine::Impl::SetMatrices(const Napi::CallbackInfo& info)
    {
        const auto uniformData = info[0].As<Napi::External<UniformInfo>>().Data();
        const auto matricesArray = info[1].As<Napi::Float32Array>();

        const size_t elementLength = matricesArray.ElementLength();
        assert(elementLength % 16 == 0);

        m_currentProgram->SetUniform(uniformData->Handle, gsl::span(matricesArray.Data(), elementLength), elementLength / 16);
    }

    void NativeEngine::Impl::SetMatrix3x3(const Napi::CallbackInfo& info)
    {
        // args: ShaderProperty property, gsl::span<const float> matrix

        assert(false);
    }

    void NativeEngine::Impl::SetMatrix2x2(const Napi::CallbackInfo& info)
    {
        // args: ShaderProperty property, gsl::span<const float> matrix

        assert(false);
    }

    void NativeEngine::Impl::SetFloat(const Napi::CallbackInfo& info)
    {
        const auto uniformData = info[0].As<Napi::External<UniformInfo>>().Data();
        const float values[] =
        {
            info[1].As<Napi::Number>().FloatValue(),
            0.0f,
            0.0f,
            0.0f
        };

        m_currentProgram->SetUniform(uniformData->Handle, values);
    }

    void NativeEngine::Impl::SetFloat2(const Napi::CallbackInfo& info)
    {
        const auto uniformData = info[0].As<Napi::External<UniformInfo>>().Data();
        const float values[] =
        {
            info[1].As<Napi::Number>().FloatValue(),
            info[2].As<Napi::Number>().FloatValue(),
            0.0f,
            0.0f
        };

        m_currentProgram->SetUniform(uniformData->Handle, values);
    }

    void NativeEngine::Impl::SetFloat3(const Napi::CallbackInfo& info)
    {
        const auto uniformData = info[0].As<Napi::External<UniformInfo>>().Data();
        const float values[] =
        {
            info[1].As<Napi::Number>().FloatValue(),
            info[2].As<Napi::Number>().FloatValue(),
            info[3].As<Napi::Number>().FloatValue(),
            0.0f
        };

        m_currentProgram->SetUniform(uniformData->Handle, values);
    }

    void NativeEngine::Impl::SetFloat4(const Napi::CallbackInfo& info)
    {
        const auto uniformData = info[0].As<Napi::External<UniformInfo>>().Data();
        const float values[] =
        {
            info[1].As<Napi::Number>().FloatValue(),
            info[2].As<Napi::Number>().FloatValue(),
            info[3].As<Napi::Number>().FloatValue(),
            info[4].As<Napi::Number>().FloatValue()
        };

        m_currentProgram->SetUniform(uniformData->Handle, values);
    }

    void NativeEngine::Impl::SetBool(const Napi::CallbackInfo& info)
    {
        // args: ShaderProperty property, bool value

        assert(false);
    }

    Napi::Value NativeEngine::Impl::CreateTexture(const Napi::CallbackInfo& info)
    {
        return Napi::External<TextureData>::New(info.Env(), new TextureData());
    }

    void NativeEngine::Impl::LoadTexture(const Napi::CallbackInfo& info)
    {
        const auto textureData = info[0].As<Napi::External<TextureData>>().Data();
        const auto buffer = info[1].As<Napi::ArrayBuffer>();
        const auto mipMap = info[2].As<Napi::Boolean>().Value();

        textureData->Images.push_back(bimg::imageParse(&m_allocator, buffer.Data(), static_cast<uint32_t>(buffer.ByteLength())));
        auto& image = *textureData->Images.front();

        textureData->Texture = bgfx::createTexture2D(
            image.m_width,
            image.m_height,
            false, // TODO: generate mipmaps when requested
            1,
            static_cast<bgfx::TextureFormat::Enum>(image.m_format),
            0,
            bgfx::makeRef(image.m_data, image.m_size));
    }

    void NativeEngine::Impl::LoadCubeTexture(const Napi::CallbackInfo& info)
    {
        const auto textureData = info[0].As<Napi::External<TextureData>>().Data();
        const auto mipLevelsArray = info[1].As<Napi::Array>();
        const auto flipY = info[2].As<Napi::Boolean>().Value();

        std::vector<std::vector<bimg::ImageContainer*>> images{};
        images.reserve(mipLevelsArray.Length());

        uint32_t totalSize = 0;

        for (uint32_t mipLevel = 0; mipLevel < mipLevelsArray.Length(); mipLevel++)
        {
            const auto facesArray = mipLevelsArray[mipLevel].As<Napi::Array>();

            images.emplace_back().reserve(facesArray.Length());

            for (uint32_t face = 0; face < facesArray.Length(); face++)
            {
                const auto image = facesArray[face].As<Napi::TypedArray>();
                auto buffer = gsl::make_span(static_cast<uint8_t*>(image.ArrayBuffer().Data()) + image.ByteOffset(), image.ByteLength());

                textureData->Images.push_back(bimg::imageParse(&m_allocator, buffer.data(), static_cast<uint32_t>(buffer.size())));
                images.back().push_back(textureData->Images.back());
                totalSize += static_cast<uint32_t>(images.back().back()->m_size);
            }
        }

        auto allPixels = bgfx::alloc(totalSize);

        auto ptr = allPixels->data;
        for (uint32_t face = 0; face < images.front().size(); face++)
        {
            for (uint32_t mipLevel = 0; mipLevel < images.size(); mipLevel++)
            {
                const auto image = images[mipLevel][face];

                std::memcpy(ptr, image->m_data, image->m_size);

                if (flipY)
                {
                    FlipYInImageBytes(gsl::make_span(ptr, image->m_size), image->m_height, image->m_size / image->m_height);
                }

                ptr += image->m_size;
            }
        }

        bgfx::TextureFormat::Enum format{};
        switch (images.front().front()->m_format)
        {
            case bimg::TextureFormat::RGBA8:
            {
                format = bgfx::TextureFormat::RGBA8;
                break;
            }
            case bimg::TextureFormat::RGB8:
            {
                format = bgfx::TextureFormat::RGB8;
                break;
            }
            default:
            {
                throw std::exception("Unexpected texture format.");
            }
        }

        textureData->Texture = bgfx::createTextureCube(
            images.front().front()->m_width,         // Side size
            true,                                           // Has mips
            1,                                              // Number of layers
            format,                                         // Self-explanatory
            0x0,                                            // Flags
            allPixels);                                     // Memory
    }

    Napi::Value NativeEngine::Impl::GetTextureWidth(const Napi::CallbackInfo& info)
    {
        const auto textureData = info[0].As<Napi::External<TextureData>>().Data();
        assert(textureData->Images.size() > 0 && !textureData->Images.front()->m_cubeMap);
        return Napi::Value::From(info.Env(), textureData->Images.front()->m_width);
    }

    Napi::Value NativeEngine::Impl::GetTextureHeight(const Napi::CallbackInfo& info)
    {
        const auto textureData = info[0].As<Napi::External<TextureData>>().Data();
        assert(textureData->Images.size() > 0 && !textureData->Images.front()->m_cubeMap);
        return Napi::Value::From(info.Env(), textureData->Images.front()->m_width);
    }

    void NativeEngine::Impl::SetTextureSampling(const Napi::CallbackInfo& info)
    {
        const auto textureData = info[0].As<Napi::External<TextureData>>().Data();
        const auto filter = static_cast<Filter>(info[1].As<Napi::Number>().Uint32Value());

        // STUB: Stub.
    }

    void NativeEngine::Impl::SetTextureWrapMode(const Napi::CallbackInfo& info)
    {
        const auto textureData = info[0].As<Napi::External<TextureData>>().Data();
        const auto addressModeU = static_cast<AddressMode>(info[1].As<Napi::Number>().Uint32Value());
        const auto addressModeV = static_cast<AddressMode>(info[2].As<Napi::Number>().Uint32Value());
        const auto addressModeW = static_cast<AddressMode>(info[3].As<Napi::Number>().Uint32Value());

        // STUB: Stub.
    }

    void NativeEngine::Impl::SetTextureAnisotropicLevel(const Napi::CallbackInfo& info)
    {
        const auto textureData = info[0].As<Napi::External<TextureData>>().Data();
        const auto value = info[1].As<Napi::Number>().Uint32Value();

        // STUB: Stub.
    }

    void NativeEngine::Impl::SetTexture(const Napi::CallbackInfo& info)
    {
        const auto uniformData = info[0].As<Napi::External<UniformInfo>>().Data();
        const auto textureData = info[1].As<Napi::External<TextureData>>().Data();

        bgfx::setTexture(uniformData->Stage, uniformData->Handle, textureData->Texture);
    }

    void NativeEngine::Impl::DeleteTexture(const Napi::CallbackInfo& info)
    {
        const auto textureData = info[0].As<Napi::External<TextureData>>().Data();
        delete textureData;
    }

    void NativeEngine::Impl::DrawIndexed(const Napi::CallbackInfo& info)
    {
        const auto fillMode = info[0].As<Napi::Number>().Int32Value();
        const auto elementStart = info[1].As<Napi::Number>().Int32Value();
        const auto elementCount = info[2].As<Napi::Number>().Int32Value();

        // TODO: handle viewport

        for (const auto& it : m_currentProgram->Uniforms)
        {
            const ProgramData::UniformValue& value = it.second;
            bgfx::setUniform({ it.first }, value.Data.data(), value.ElementLength);
        }

        bgfx::submit(0, m_currentProgram->Program, 0, true);
    }

    void NativeEngine::Impl::Draw(const Napi::CallbackInfo& info)
    {
        const auto fillMode = info[0].As<Napi::Number>().Int32Value();
        const auto elementStart = info[1].As<Napi::Number>().Int32Value();
        const auto elementCount = info[2].As<Napi::Number>().Int32Value();

        // STUB: Stub.
        // bgfx::submit(), right?  Which means we have to preserve here the state of
        // which program is being worked on.
    }

    void NativeEngine::Impl::Clear(const Napi::CallbackInfo& info)
    {
        auto r = info[0].As<Napi::Number>().FloatValue();
        auto g = info[1].As<Napi::Number>().FloatValue();
        auto b = info[2].As<Napi::Number>().FloatValue();
        auto a = info[3].IsUndefined() ? 1.f : info[3].As<Napi::Number>().FloatValue();
        auto backBuffer = info[4].IsUndefined() ? true : info[4].As<Napi::Boolean>().Value();
        auto depth = info[5].IsUndefined() ? true : info[5].As<Napi::Boolean>().Value();
        auto stencil = info[6].IsUndefined() ? true : info[6].As<Napi::Boolean>().Value();

        // TODO CHECK: Does this have meaning for BGFX?  BGFX seems to call clear()
        // on its own, depending on the settings.
    }

    Napi::Value NativeEngine::Impl::GetRenderWidth(const Napi::CallbackInfo& info)
    {
        // TODO CHECK: Is this not just the size?  What is this?
        return Napi::Value::From(info.Env(), m_size.Width);
    }

    Napi::Value NativeEngine::Impl::GetRenderHeight(const Napi::CallbackInfo& info)
    {
        // TODO CHECK: Is this not just the size?  What is this?
        return Napi::Value::From(info.Env(), m_size.Height);
    }

    void NativeEngine::Impl::DispatchAnimationFrameAsync(Napi::FunctionReference callback)
    {
        // The purpose of encapsulating the callbackPtr in a std::shared_ptr is because, under the hood, the lambda is
        // put into a kind of function which requires a copy constructor for all of its captured variables.  Because
        // the Napi::FunctionReference is not copyable, this breaks when trying to capture the callback directly, so we
        // wrap it in a std::shared_ptr to allow the capture to function correctly.
        m_runtimeImpl.Execute([this, callbackPtr = std::make_shared<Napi::FunctionReference>(std::move(callback))](auto&)
        {
            //bgfx_test(static_cast<uint16_t>(m_size.Width), static_cast<uint16_t>(m_size.Height));

            callbackPtr->Call({});
            bgfx::frame();
        });
    }

    // NativeEngine exterior definitions.

    NativeEngine::NativeEngine(void* nativeWindowPtr, RuntimeImpl& runtimeImpl)
        : m_impl{ std::make_unique<NativeEngine::Impl>(nativeWindowPtr, runtimeImpl) }
    {
    }

    NativeEngine::~NativeEngine()
    {
    }

    void NativeEngine::Initialize(Napi::Env& env)
    {
        m_impl->Initialize(env);
    }

    void NativeEngine::UpdateSize(float width, float height)
    {
        m_impl->UpdateSize(width, height);
    }

    void NativeEngine::UpdateRenderTarget()
    {
        m_impl->UpdateRenderTarget();
    }

    void NativeEngine::Suspend()
    {
        m_impl->Suspend();
    }
}
