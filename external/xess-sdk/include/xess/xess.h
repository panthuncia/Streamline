/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef XESS_H
#define XESS_H

#ifdef XESS_SHARED_LIB
#ifdef _WIN32
#ifdef XESS_EXPORT_API
#define XESS_API __declspec(dllexport)
#else
#define XESS_API __declspec(dllimport)
#endif
#else
#define XESS_API __attribute__((visibility("default")))
#endif
#else
#define XESS_API
#endif

#if !defined _MSC_VER || (_MSC_VER >= 1929)
#define XESS_PRAGMA(x) _Pragma(#x)
#else
#define XESS_PRAGMA(x) __pragma(x)
#endif
#define XESS_PACK_B_X(x) XESS_PRAGMA(pack(push, x))
#define XESS_PACK_E() XESS_PRAGMA(pack(pop))
#define XESS_PACK_B() XESS_PACK_B_X(8)

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _xess_context_handle_t* xess_context_handle_t;

XESS_PACK_B()
typedef struct _xess_version_t
{
	uint16_t major;
	uint16_t minor;
	uint16_t patch;
	uint16_t reserved;
} xess_version_t;
XESS_PACK_E()

XESS_PACK_B()
typedef struct _xess_2d_t
{
	uint32_t x;
	uint32_t y;
} xess_2d_t;
XESS_PACK_E()

typedef xess_2d_t xess_coord_t;

typedef enum _xess_quality_settings_t
{
	XESS_QUALITY_SETTING_ULTRA_PERFORMANCE = 100,
	XESS_QUALITY_SETTING_PERFORMANCE = 101,
	XESS_QUALITY_SETTING_BALANCED = 102,
	XESS_QUALITY_SETTING_QUALITY = 103,
	XESS_QUALITY_SETTING_ULTRA_QUALITY = 104,
	XESS_QUALITY_SETTING_ULTRA_QUALITY_PLUS = 105,
	XESS_QUALITY_SETTING_AA = 106,
} xess_quality_settings_t;

typedef enum _xess_init_flags_t
{
	XESS_INIT_FLAG_NONE = 0,
	XESS_INIT_FLAG_HIGH_RES_MV = 1 << 0,
	XESS_INIT_FLAG_INVERTED_DEPTH = 1 << 1,
	XESS_INIT_FLAG_EXPOSURE_SCALE_TEXTURE = 1 << 2,
	XESS_INIT_FLAG_RESPONSIVE_PIXEL_MASK = 1 << 3,
	XESS_INIT_FLAG_USE_NDC_VELOCITY = 1 << 4,
	XESS_INIT_FLAG_EXTERNAL_DESCRIPTOR_HEAP = 1 << 5,
	XESS_INIT_FLAG_LDR_INPUT_COLOR = 1 << 6,
	XESS_INIT_FLAG_JITTERED_MV = 1 << 7,
	XESS_INIT_FLAG_ENABLE_AUTOEXPOSURE = 1 << 8
} xess_init_flags_t;

XESS_PACK_B()
typedef struct _xess_properties_t
{
	uint32_t requiredDescriptorCount;
	uint64_t tempBufferHeapSize;
	uint64_t tempTextureHeapSize;
} xess_properties_t;
XESS_PACK_E()

typedef enum _xess_result_t
{
	XESS_RESULT_WARNING_NONEXISTING_FOLDER = 1,
	XESS_RESULT_WARNING_OLD_DRIVER = 2,
	XESS_RESULT_SUCCESS = 0,
	XESS_RESULT_ERROR_UNSUPPORTED_DEVICE = -1,
	XESS_RESULT_ERROR_UNSUPPORTED_DRIVER = -2,
	XESS_RESULT_ERROR_UNINITIALIZED = -3,
	XESS_RESULT_ERROR_INVALID_ARGUMENT = -4,
	XESS_RESULT_ERROR_DEVICE_OUT_OF_MEMORY = -5,
	XESS_RESULT_ERROR_DEVICE = -6,
	XESS_RESULT_ERROR_NOT_IMPLEMENTED = -7,
	XESS_RESULT_ERROR_INVALID_CONTEXT = -8,
	XESS_RESULT_ERROR_OPERATION_IN_PROGRESS = -9,
	XESS_RESULT_ERROR_UNSUPPORTED = -10,
	XESS_RESULT_ERROR_CANT_LOAD_LIBRARY = -11,
	XESS_RESULT_ERROR_WRONG_CALL_ORDER = -12,
	XESS_RESULT_ERROR_UNKNOWN = -1000,
} xess_result_t;

typedef enum _xess_logging_level_t
{
	XESS_LOGGING_LEVEL_DEBUG = 0,
	XESS_LOGGING_LEVEL_INFO = 1,
	XESS_LOGGING_LEVEL_WARNING = 2,
	XESS_LOGGING_LEVEL_ERROR = 3
} xess_logging_level_t;

typedef void (*xess_app_log_callback_t)(const char* message, xess_logging_level_t loggingLevel);

#ifndef XESS_TYPES_ONLY

XESS_API xess_result_t xessGetVersion(xess_version_t* pVersion);

XESS_API xess_result_t xessGetIntelXeFXVersion(xess_context_handle_t hContext,
	xess_version_t* pVersion);

XESS_API xess_result_t xessGetProperties(xess_context_handle_t hContext,
	const xess_2d_t* pOutputResolution, xess_properties_t* pBindingProperties);

XESS_API xess_result_t xessGetInputResolution(xess_context_handle_t hContext,
	const xess_2d_t* pOutputResolution, xess_quality_settings_t qualitySettings,
	xess_2d_t* pInputResolution);

XESS_API xess_result_t xessGetOptimalInputResolution(xess_context_handle_t hContext,
	const xess_2d_t* pOutputResolution, xess_quality_settings_t qualitySettings,
	xess_2d_t* pInputResolutionOptimal, xess_2d_t* pInputResolutionMin, xess_2d_t* pInputResolutionMax);

XESS_API xess_result_t xessGetJitterScale(xess_context_handle_t hContext, float* pX, float* pY);
XESS_API xess_result_t xessGetVelocityScale(xess_context_handle_t hContext, float* pX, float* pY);
XESS_API xess_result_t xessDestroyContext(xess_context_handle_t hContext);
XESS_API xess_result_t xessSetJitterScale(xess_context_handle_t hContext, float x, float y);
XESS_API xess_result_t xessSetVelocityScale(xess_context_handle_t hContext, float x, float y);
XESS_API xess_result_t xessSetExposureMultiplier(xess_context_handle_t hContext, float scale);
XESS_API xess_result_t xessGetExposureMultiplier(xess_context_handle_t hContext, float* pScale);
XESS_API xess_result_t xessSetLoggingCallback(xess_context_handle_t hContext,
	xess_logging_level_t loggingLevel, xess_app_log_callback_t loggingCallback);
XESS_API xess_result_t xessIsOptimalDriver(xess_context_handle_t hContext);
XESS_API xess_result_t xessForceLegacyScaleFactors(xess_context_handle_t hContext, bool force);
XESS_API xess_result_t xessGetPipelineBuildStatus(xess_context_handle_t hContext);

#endif

#ifdef __cplusplus
}
#endif

#endif
