//	Copyright (c) 2020, Michael Kunz. All rights reserved.
//	https://github.com/kunzmi/ImageStackAlignator
//
//	This file is part of ImageStackAlignator.
//
//	ImageStackAlignator is free software: you can redistribute it and/or modify
//	it under the terms of the GNU Lesser General Public License as 
//	published by the Free Software Foundation, version 3.
//
//	ImageStackAlignator is distributed in the hope that it will be useful,
//	but WITHOUT ANY WARRANTY; without even the implied warranty of
//	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//	GNU Lesser General Public License for more details.
//
//	You should have received a copy of the GNU Lesser General Public
//	License along with this library; if not, write to the Free Software
//	Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
//	MA 02110-1301  USA, http://www.gnu.org/licenses/.

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cfloat>

#include <math.h>

extern "C"
__global__ void ComputeRobustnessMask(
	const float3* __restrict__ rawImgRef,
	const float3* __restrict__ rawImgMoved,
	float4* __restrict__ robustnessMask,
	cudaTextureObject_t texUV,
	int imgWidth,
	int imgHeight,
	int imgPitch,
	int maskPitch,
	float alpha,
	float beta,
	float thresholdM)
{
	int pxX = blockIdx.x * blockDim.x + threadIdx.x;
	int pxY = blockIdx.y * blockDim.y + threadIdx.y;

	extern __shared__ float3 pixelsRef[];
	int sharedOffset = 3 * 3 * (threadIdx.y * blockDim.x + threadIdx.x);

	if (pxX >= imgWidth - 1|| pxY >= imgHeight - 1 || pxX < 1 || pxY < 1)
		return;

	float3 meanRef = make_float3(0, 0, 0);
	float3 meanMoved = make_float3(0, 0, 0);
	float3 stdRef = make_float3(0, 0, 0);
	float3 stdMoved = make_float3(0, 0, 0);
	float3 dist = make_float3(0, 0, 0);
	float3 sigma = make_float3(0, 0, 0);

	float2 shiftf = tex2D<float2>(texUV, ((float)pxX + 0.5f) / (float)imgWidth, ((float)pxY + 0.5f) / (float)imgHeight);
	float2 maxShift = shiftf;
	float2 minShift = shiftf;
	
	for (int y = -2; y <= 2; y++)
	{
		for (int x = -2; x <= 2; x++)
		{
			float2 s = tex2D<float2>(texUV, ((float)pxX + x + 0.5f) / (float)imgWidth, ((float)pxY + y + 0.5f) / (float)imgHeight);
			maxShift.x = fmaxf(s.x, shiftf.x);
			maxShift.y = fmaxf(s.y, shiftf.y);
			minShift.x = fminf(s.x, shiftf.x);
			minShift.y = fminf(s.y, shiftf.y);
		}
	}

	int2 shift;
	//half resolution image:
	shift.x = roundf(shiftf.x * 0.5f);
	shift.y = roundf(shiftf.y * 0.5f);

	for (int y = -1; y <= 1; y++)
	{
		for (int x = -1; x <= 1; x++)
		{
			float3 p = *(((float3*)((char*)rawImgRef + imgPitch * (pxY + y))) + pxX + x);
			pixelsRef[sharedOffset + (y + 1) * 3 + (x + 1)] = p;

			meanRef.x += p.x;
			meanRef.y += p.y;
			meanRef.z += p.z;

			int ppy = min(max(pxY + shift.y + y, 0), imgHeight - 1);
			int ppx = min(max(pxX + shift.x + x, 0), imgWidth - 1);
			p = *(((float3*)((char*)rawImgMoved + imgPitch * (ppy))) + ppx);
			meanMoved.x += p.x;
			meanMoved.y += p.y;
			meanMoved.z += p.z;
		}
	}
	meanRef.x /= 9.0f;
	meanRef.y /= 9.0f;
	meanRef.z /= 9.0f;
	meanMoved.x /= 9.0f;
	meanMoved.y /= 9.0f;
	meanMoved.z /= 9.0f;

	float meandist = fabs(meanRef.x - meanMoved.x) + fabs(meanRef.y - meanMoved.y) + fabs(meanRef.z - meanMoved.z);
	meandist /= 3.0f;
	maxShift.x *= 0.5f * meandist;
	maxShift.y *= 0.5f * meandist;
	minShift.x *= 0.5f * meandist;
	minShift.y *= 0.5f * meandist;

	float M = sqrtf((maxShift.x - minShift.x) * (maxShift.x - minShift.x) + (maxShift.y - minShift.y) * (maxShift.y - minShift.y));

	for (int y = -1; y <= 1; y++)
	{
		for (int x = -1; x <= 1; x++)
		{
			int p = sharedOffset + (y + 1) * 3 + (x + 1);
			stdRef.x += (pixelsRef[p].x - meanRef.x) * (pixelsRef[p].x - meanRef.x);
			stdRef.y += (pixelsRef[p].y - meanRef.y) * (pixelsRef[p].y - meanRef.y);
			stdRef.z += (pixelsRef[p].z - meanRef.z) * (pixelsRef[p].z - meanRef.z);
		}
	}

	stdRef.x = sqrtf(stdRef.x / 9.0f);
	stdRef.y = sqrtf(stdRef.y / 9.0f);
	stdRef.z = sqrtf(stdRef.z / 9.0f);

	float3 sigmaMD;
	sigmaMD.x = sqrtf(alpha * meanRef.x + beta);
	sigmaMD.y = sqrtf(alpha * meanRef.y + beta) / sqrtf(2.0f); //we have two green pixels averaged --> devide by sqrtf(2);
	sigmaMD.z = sqrtf(alpha * meanRef.z + beta);

	dist.x = fabs(meanRef.x - meanMoved.x);
	dist.y = fabs(meanRef.y - meanMoved.y);
	dist.z = fabs(meanRef.z - meanMoved.z);

	sigma.x = fmaxf(sigmaMD.x, stdRef.x);
	sigma.y = fmaxf(sigmaMD.y, stdRef.y);
	sigma.z = fmaxf(sigmaMD.z, stdRef.z);
	
	dist.x = dist.x * (stdRef.x * stdRef.x / (stdRef.x * stdRef.x + sigmaMD.x * sigmaMD.x));
	dist.y = dist.y * (stdRef.y * stdRef.y / (stdRef.y * stdRef.y + sigmaMD.y * sigmaMD.y));
	dist.z = dist.z * (stdRef.z * stdRef.z / (stdRef.z * stdRef.z + sigmaMD.z * sigmaMD.z));/**/
	   
	float4 mask;
	float s = 1.5f;
	if (M > thresholdM)
		s = 0;

	const float t = 0.12f;
	mask.x = fmaxf(fminf(s * exp(-dist.x * dist.x / (sigma.x * sigma.x)) - t, 1.0f), 0.0f);
	mask.y = fmaxf(fminf(s * exp(-dist.y * dist.y / (sigma.y * sigma.y)) - t, 1.0f), 0.0f);
	mask.z = fmaxf(fminf(s * exp(-dist.z * dist.z / (sigma.z * sigma.z)) - t, 1.0f), 0.0f);
	mask.w = M;

	*(((float4*)((char*)robustnessMask + maskPitch * pxY)) + pxX) = mask;
}