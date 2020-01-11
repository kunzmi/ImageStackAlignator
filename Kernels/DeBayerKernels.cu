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
#include <cmath>


enum BayerColor : int
{
	Red = 0,
	Green = 1,
	Blue = 2,
	Cyan = 3,
	Magenta = 4,
	Yellow = 5,
	White = 6
};


extern "C"
__device__ __constant__ BayerColor c_cfaPattern[2][2];

#define RAW(xx, yy) (*(((float*)((char*)imgIn + (yy) * strideIn)) + (xx)))
#define RAWR(xx, yy) ((RAW(xx, yy) - blackPoint.x) * scale.x)
#define RAWG(xx, yy) ((RAW(xx, yy) - blackPoint.y) * scale.y)
#define RAWB(xx, yy) ((RAW(xx, yy) - blackPoint.z) * scale.z)

#define RED(xx, yy) ((*(((float3*)((char*)outImage + (yy) * strideOut)) + (xx))).x)
#define GREEN(xx, yy) ((*(((float3*)((char*)outImage + (yy) * strideOut)) + (xx))).y)
#define BLUE(xx, yy) ((*(((float3*)((char*)outImage + (yy) * strideOut)) + (xx))).z)


//Simple gradient and laplacian supported weighted interpolation of green channel. See e.g. Wu and Zhang
extern "C"
__global__ void deBayerGreenKernel(const int width, const int height, const float* __restrict__ imgIn, int strideIn, float3* outImage, int strideOut, float3 blackPoint, float3 scale)
{
	// integer pixel coordinates	
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width - 2 || x < 2) return;
	if (y >= height - 2 || y < 2) return;

	BayerColor thisPixel = c_cfaPattern[y % 2][x % 2];

	float g = 0;
	float p;
	float xMinus2;
	float xMinus1;
	float xPlus1;
	float xPlus2;

	float yMinus2;
	float yMinus1;
	float yPlus1;
	float yPlus2;
	float gradientX;
	float gradientY;

	float laplaceX;
	float laplaceY;

	float interpolX;
	float interpolY;

	float weight;

	switch (thisPixel)
	{
	case Green:
		g = RAWG(x, y);
		break;

	case Red:
		p = RAWR(x, y);
		xMinus2 = RAWR(x - 2, y);
		xMinus1 = RAWG(x - 1, y);
		xPlus1 = RAWG(x + 1, y);
		xPlus2 = RAWR(x + 2, y);

		yMinus2 = RAWR(x, y - 2);
		yMinus1 = RAWG(x, y - 1);
		yPlus1 = RAWG(x, y + 1);
		yPlus2 = RAWR(x, y + 2);

		gradientX = 0.5f * fabs(xPlus1 - xMinus1);
		gradientY = 0.5f * fabs(yPlus1 - yMinus1);

		laplaceX = 0.25f * fabs(2.0f * p - xMinus2 - xPlus2);
		laplaceY = 0.25f * fabs(2.0f * p - yMinus2 - yPlus2);

		interpolX = 0.125f * (-xMinus2 + 4.0f * xMinus1 + 2.0f * p + 4.0f * xPlus1 - xPlus2);
		interpolY = 0.125f * (-yMinus2 + 4.0f * yMinus1 + 2.0f * p + 4.0f * yPlus1 - yPlus2);

		weight = (gradientY + laplaceY) / (gradientX + gradientY + laplaceX + laplaceY + 0.000000001f);

		g = weight * interpolX + (1.0f - weight) * interpolY;

		break;
	case Blue:
		p = RAWB(x, y);
		xMinus2 = RAWB(x - 2, y);
		xMinus1 = RAWG(x - 1, y);
		xPlus1 = RAWG(x + 1, y);
		xPlus2 = RAWB(x + 2, y);

		yMinus2 = RAWB(x, y - 2);
		yMinus1 = RAWG(x, y - 1);
		yPlus1 = RAWG(x, y + 1);
		yPlus2 = RAWB(x, y + 2);

		gradientX = 0.5f * fabs(xPlus1 - xMinus1);
		gradientY = 0.5f * fabs(yPlus1 - yMinus1);

		laplaceX = 0.25f * fabs(2.0f * p - xMinus2 - xPlus2);
		laplaceY = 0.25f * fabs(2.0f * p - yMinus2 - yPlus2);

		interpolX = 0.125f * (-xMinus2 + 4.0f * xMinus1 + 2.0f * p + 4.0f * xPlus1 - xPlus2);
		interpolY = 0.125f * (-yMinus2 + 4.0f * yMinus1 + 2.0f * p + 4.0f * yPlus1 - yPlus2);

		weight = (gradientY + laplaceY) / (gradientX + gradientY + laplaceX + laplaceY + 0.000000001f);

		g = weight * interpolX + (1.0f - weight) * interpolY;


		break;
	}
	GREEN(x, y) = g;
}

//interpolate color difference to green channel
extern "C"
__global__ void deBayerRedBlueKernel(const int width, const int height, const float* __restrict__ imgIn, int strideIn, float3* outImage, int strideOut, float3 blackPoint, float3 scale)
{
	// integer pixel coordinates	
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width - 2 || x < 2) return;
	if (y >= height - 2 || y < 2) return;

	BayerColor thisPixel = c_cfaPattern[y % 2][x % 2];
	BayerColor thisRow = c_cfaPattern[y % 2][(x + 1) % 2];
	float r, b;
	float g = GREEN(x, y);

	switch (thisPixel)
	{
	case Green:
		if (thisRow == Red)
		{
			float xMinus1r = RAWR(x - 1, y);
			float xPlus1r = RAWR(x + 1, y);
			float xMinus1g = GREEN(x - 1, y);
			float xPlus1g = GREEN(x + 1, y);
			r = g + 0.5f * ((xMinus1r - xMinus1g) + (xPlus1r - xPlus1g));

			float yMinus1b = RAWB(x, y - 1);
			float yPlus1b = RAWB(x, y + 1);
			float yMinus1g = GREEN(x, y - 1);
			float yPlus1g = GREEN(x, y + 1);
			b = g + 0.5f * ((yMinus1b - yMinus1g) + (yPlus1b - yPlus1g));
		}
		else
		{
			float xMinus1b = RAWB(x - 1, y);
			float xPlus1b = RAWB(x + 1, y);
			float xMinus1g = GREEN(x - 1, y);
			float xPlus1g = GREEN(x + 1, y);
			b = g + 0.5f * ((xMinus1b - xMinus1g) + (xPlus1b - xPlus1g));

			float yMinus1r = RAWR(x, y - 1);
			float yPlus1r = RAWR(x, y + 1);
			float yMinus1g = GREEN(x, y - 1);
			float yPlus1g = GREEN(x, y + 1);
			r = g + 0.5f * ((yMinus1r - yMinus1g) + (yPlus1r - yPlus1g));
		}
		break;
	case Red:
		r = RAWR(x, y);
		{
			float mmB = RAWB(x - 1, y - 1);
			float pmB = RAWB(x + 1, y - 1);
			float ppB = RAWB(x + 1, y + 1);
			float mpB = RAWB(x - 1, y + 1);
			float mmG = GREEN(x - 1, y - 1);
			float pmG = GREEN(x + 1, y - 1);
			float ppG = GREEN(x + 1, y + 1);
			float mpG = GREEN(x - 1, y + 1);
			b = g + 0.25f * ((mmB - mmG) + (pmB - pmG) + (ppB - ppG) + (mpB - mpG));
		}
		break;
	case Blue:
		b = RAWB(x, y);
		{
			float mmR = RAWR(x - 1, y - 1);
			float pmR = RAWR(x + 1, y - 1);
			float ppR = RAWR(x + 1, y + 1);
			float mpR = RAWR(x - 1, y + 1);
			float mmG = GREEN(x - 1, y - 1);
			float pmG = GREEN(x + 1, y - 1);
			float ppG = GREEN(x + 1, y + 1);
			float mpG = GREEN(x - 1, y + 1);
			r = g + 0.25f * ((mmR - mmG) + (pmR - pmG) + (ppR - ppG) + (mpR - mpG));
		}
		break;

	}
	RED(x, y) = r;
	BLUE(x, y) = b;
}

#undef RAW
#undef RAWR
#undef RAWG
#undef RAWB

#undef RED
#undef GREEN
#undef BLUE

#define RAW2(x, y) ((float)dataIn[(y) * dimX * 2 + (x)])
extern "C"
__global__ void deBayersSubSample3(unsigned short* dataIn, float3* imgOut, float maxVal, int dimX, int dimY, int strideOut)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= dimX || y >= dimY)
		return;



	float3 pixel = make_float3(0, 0, 0);

	float3* lineOut = (float3*)((char*)imgOut + y * strideOut);
	float factor = 1.0f / maxVal;

	for (int ix = 0; ix < 2; ix++)
	{
		for (int iy = 0; iy < 2; iy++)
		{
			BayerColor thisPixel = c_cfaPattern[iy][ix];

			if (thisPixel == Green)
			{
				pixel.y += RAW2(2 * x + ix, 2 * y + iy) * factor * 0.5f; //we have two green pixels per bayer
			}
			else
				if (thisPixel == Red)
				{
					pixel.x = RAW2(2 * x + ix, 2 * y + iy) * factor;
				}
				else
					if (thisPixel == Blue)
					{
						pixel.z = RAW2(2 * x + ix, 2 * y + iy) * factor;
					}
		}
	}

	lineOut[x] = pixel;
}

#undef RAW2


#define RAW(x, y) ((float)dataIn[(y) * dimX + (x)])
extern "C"
__global__ void accumulateImages(
	unsigned short* __restrict__ dataIn,
	float3 * __restrict__ imgOut,
	float3 * __restrict__ totalWeights,
	const float4 * __restrict__ certaintyMask,
	const float3* __restrict__ kernelParam,
	const float2* __restrict__ shifts,
	float maxVal, int dimX, int dimY, int strideOut, int strideMask, int strideShift)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < 1 || y < 1 || x >= dimX - 1 || y >= dimY - 1)
		return;


	float3 pixel = *(((float3*)((char*)imgOut + y * strideOut)) + x);
	float3 totalWeight = *(((float3*)((char*)totalWeights + y * strideOut)) + x);
	float3 kernel = *(((const float3*)((const char*)kernelParam + y * strideOut)) + x);
	float2 shift = *(((const float2*)((const char*)shifts + y * strideShift)) + x);
	shift.x = roundf(shift.x);
	shift.y = roundf(shift.y);
	int sx = shift.x;
	int sy = shift.y;

	float factor = 1.0f / maxVal;

	for (int py = -2; py <= 2; py++)
	{
		for (int px = -2; px <= 2; px++)
		{
			int ppsx = x + px + sx;
			int ppsy = y + py + sy;

			int ppx = x + px;
			int ppy = y + py;
			
			ppsx = min(max(ppsx, 0), dimX - 1);
			ppsy = min(max(ppsy, 0), dimY - 1);

			ppx = min(max(ppx, 0), dimX - 1);
			ppy = min(max(ppy, 0), dimY - 1);

			BayerColor thisPixel = c_cfaPattern[(ppsy) % 2][(ppsx) % 2];

			float w = px * px * kernel.x + 2 * px * py * kernel.z + py * py * kernel.y;
			w = exp(-0.5f * w);
			if (!isfinite(w))
				w = px*py == 0 ? 1 : 0;

			float raw = RAW(ppsx, ppsy) * factor;

			if (thisPixel == Green)
			{
				float certainty = (((const float4*)((const char*)certaintyMask + (ppy / 2) * strideMask)) + (ppx / 2))->y;
				if (!isfinite(certainty))
					certainty = 0;
				pixel.y += raw * w * certainty;
				totalWeight.y += w * certainty;
			}
			else
				if (thisPixel == Red)
				{
					float certainty = (((const float4*)((const char*)certaintyMask + (ppy / 2) * strideMask)) + (ppx / 2))->x;
					if (!isfinite(certainty))
						certainty = 0;
					pixel.x += raw * w * certainty;
					totalWeight.x += w * certainty;
				}
				else
					if (thisPixel == Blue)
					{
						float certainty = (((const float4*)((const char*)certaintyMask + (ppy / 2) * strideMask)) + (ppx / 2))->z;
						if (!isfinite(certainty))
							certainty = 0;
						pixel.z += raw * w * certainty;
						totalWeight.z += w * certainty;
					}
		}
	}

	*(((float3*)((char*)imgOut + y * strideOut)) + x) = pixel;
	*(((float3*)((char*)totalWeights + y * strideOut)) + x) = totalWeight;
}

extern "C"
__global__ void accumulateImagesSuperRes(
	unsigned short* __restrict__ dataIn,
	float3 * __restrict__ imgOut,
	float3 * __restrict__ totalWeights,
	const float4 * __restrict__ certaintyMask,
	cudaTextureObject_t kernelParam,
	cudaTextureObject_t shifts,
	float maxVal, int dimX, int dimY, int strideOut, int strideMask, int strideKernelParam, int strideShift)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < 1 || y < 1 || x >= dimX - 1 || y >= dimY - 1)
		return;


	float3 pixel = *(((float3*)((char*)imgOut + (size_t)y * (size_t)strideOut)) + (size_t)x);
	float3 totalWeight = *(((float3*)((char*)totalWeights + (size_t)y * (size_t)strideOut)) + (size_t)x);

	float posX = ((float)x + 0.5f + dimX / 2) / 2.0f / dimX;
	float posY = ((float)y + 0.5f + dimY / 2) / 2.0f / dimY;

	float4 kernel = tex2D<float4>(kernelParam, posX, posY);// *(((const float3*)((const char*)kernelParam + (y / 2 + dimY / 4) * strideKernelParam)) + (x / 2 + dimX / 4));
	float2 shift = tex2D<float2>(shifts, posX, posY);// *(((const float2*)((const char*)shifts + (y / 2 + dimY / 4) * strideShift)) + (x / 2 + dimX / 4));
	shift.x = roundf(shift.x*2);
	shift.y = roundf(shift.y*2);
	int sx = shift.x;
	int sy = shift.y;

	float factor = 1.0f / maxVal;

	for (int py = -2; py <= 2; py++)
	{
		for (int px = -2; px <= 2; px++)
		{
			int ppsx = x + px + sx + dimX / 2;
			int ppsy = y + py + sy + dimY / 2;
			int ppx = x + px + dimX / 2;
			int ppy = y + py + dimY / 2;

			ppsx = min(max(ppsx/2, 0 + dimX / 4), dimX/2 - 1 + dimX / 4);
			ppsy = min(max(ppsy/2, 0 + dimY / 4), dimY/2 - 1 + dimY / 4);

			ppx = min(max(ppx / 2, 0 + dimX / 4), dimX / 2 - 1 + dimX / 4);
			ppy = min(max(ppy / 2, 0 + dimY / 4), dimY / 2 - 1 + dimY / 4);

			BayerColor thisPixel = c_cfaPattern[(ppsy) % 2][(ppsx) % 2];

			float w = px * px * kernel.x + 2 * px * py * kernel.z + py * py * kernel.y;
			w = exp(-0.5f * w);
			if (!isfinite(w))
				w = px * py == 0 ? 1 : 0;

			float raw = RAW(ppsx, ppsy) * factor;

			if (thisPixel == Green)
			{
				float certainty = (((const float4*)((const char*)certaintyMask + (ppy / 2) * strideMask)) + (ppx / 2))->y;
				if (!isfinite(certainty))
					certainty = 0;
				pixel.y += raw * w * certainty;
				totalWeight.y += w * certainty;
			}
			else
				if (thisPixel == Red)
				{
					float certainty = (((const float4*)((const char*)certaintyMask + (ppy / 2) * strideMask)) + (ppx / 2))->x;
					if (!isfinite(certainty))
						certainty = 0;
					pixel.x += raw * w * certainty;
					totalWeight.x += w * certainty;
				}
				else
					if (thisPixel == Blue)
					{
						float certainty = (((const float4*)((const char*)certaintyMask + (ppy / 2) * strideMask)) + (ppx / 2))->z;
						if (!isfinite(certainty))
							certainty = 0;
						pixel.z += raw * w * certainty;
						totalWeight.z += w * certainty;
					}
		}
	}

	*(((float3*)((char*)imgOut + (size_t)y * (size_t)strideOut)) + (size_t)x) = pixel;
	*(((float3*)((char*)totalWeights + (size_t)y * (size_t)strideOut)) + (size_t)x) = totalWeight;
}
#undef RAW