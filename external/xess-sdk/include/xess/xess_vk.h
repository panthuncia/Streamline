/*******************************************************************************
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef XESS_VK_H
#define XESS_VK_H

#include <vulkan/vulkan.h>

#include "xess.h"

#ifdef __cplusplus
extern "C" {
#endif

XESS_PACK_B()
typedef struct _xess_vk_image_view_info
{
	VkImageView imageView;
	VkImage image;
	VkImageSubresourceRange subresourceRange;
	VkFormat format;
	unsigned int width;
	unsigned int height;
} xess_vk_image_view_info;
XESS_PACK_E()

XESS_PACK_B()
typedef struct _xess_vk_execute_params_t
{
	xess_vk_image_view_info colorTexture;
	xess_vk_image_view_info velocityTexture;
	xess_vk_image_view_info depthTexture;
	xess_vk_image_view_info exposureScaleTexture;
	xess_vk_image_view_info responsivePixelMaskTexture;
	xess_vk_image_view_info outputTexture;

	float jitterOffsetX;
	float jitterOffsetY;
	float exposureScale;
	uint32_t resetHistory;
	uint32_t inputWidth;
	uint32_t inputHeight;
	xess_coord_t inputColorBase;
	xess_coord_t inputMotionVectorBase;
	xess_coord_t inputDepthBase;
	xess_coord_t inputResponsiveMaskBase;
	xess_coord_t reserved0;
	xess_coord_t outputColorBase;
} xess_vk_execute_params_t;
XESS_PACK_E()

XESS_PACK_B()
typedef struct _xess_vk_init_params_t
{
	xess_2d_t outputResolution;
	xess_quality_settings_t qualitySetting;
	uint32_t initFlags;
	uint32_t creationNodeMask;
	uint32_t visibleNodeMask;
	VkDeviceMemory tempBufferHeap;
	uint64_t bufferHeapOffset;
	VkDeviceMemory tempTextureHeap;
	uint64_t textureHeapOffset;
	VkPipelineCache pipelineCache;
} xess_vk_init_params_t;
XESS_PACK_E()

XESS_API xess_result_t xessVKGetRequiredInstanceExtensions(uint32_t* instanceExtensionsCount,
	const char* const** instanceExtensions, uint32_t* minVkApiVersion);

XESS_API xess_result_t xessVKGetRequiredDeviceExtensions(VkInstance instance,
	VkPhysicalDevice physicalDevice, uint32_t* deviceExtensionsCount,
	const char* const** deviceExtensions);

XESS_API xess_result_t xessVKGetRequiredDeviceFeatures(VkInstance instance,
	VkPhysicalDevice physicalDevice, void** features);

XESS_API xess_result_t xessVKCreateContext(VkInstance instance, VkPhysicalDevice physicalDevice,
	VkDevice device, xess_context_handle_t* phContext);

XESS_API xess_result_t xessVKBuildPipelines(xess_context_handle_t hContext,
	VkPipelineCache pipelineCache, bool blocking, uint32_t initFlags);

XESS_API xess_result_t xessVKInit(
	xess_context_handle_t hContext, const xess_vk_init_params_t* pInitParams);

XESS_API xess_result_t xessVKGetInitParams(
	xess_context_handle_t hContext, xess_vk_init_params_t* pInitParams);

XESS_API xess_result_t xessVKExecute(xess_context_handle_t hContext,
	VkCommandBuffer commandBuffer, const xess_vk_execute_params_t* pExecParams);

#ifdef __cplusplus
}
#endif

#endif
