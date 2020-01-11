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


//squared sum of a tile without the border
extern "C"
__global__ void squaredSum(
	const float* __restrict__ inTiles,
	float* __restrict__ outValues,
	int maxShift,
	int tileSize,
	int tileCount)
{
	int tileIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (tileIdx >= tileCount)
		return;

	float sum = 0;
	int tileArray = tileIdx * (tileSize + maxShift * 2) * (tileSize + maxShift * 2);
	for (int y = 0; y < tileSize; y++)
	{
		int yShift = (y + maxShift) * (tileSize + maxShift * 2);
		for (int x = 0; x < tileSize; x++)
		{
			int pxInTilesArray = tileArray + yShift + x + maxShift;
			float pixel = inTiles[pxInTilesArray];
			sum += pixel * pixel;
		}
	}
	outValues[tileIdx] = sum;
}

//Boxfilter ignoring the border parts
//blockDim.X must be tileSize + 2 * maxShift
//blockDim.Y must be 1
extern "C"
__global__ void boxFilterWithBorderX(
	const float* __restrict__ inTiles,
	float* __restrict__ outTiles,
	int maxShift,
	int tileSize,
	int tileCount)
{
	int pxX = blockIdx.x * blockDim.x + threadIdx.x;
	int pxY = blockIdx.y * blockDim.y + threadIdx.y;
	int tileIdx = blockIdx.z * blockDim.z + threadIdx.z;

	extern __shared__ float shared[]; //sizeof blockDim.z * (2 * maxShift + tileSize)

	if (pxX >= tileSize + 2 * maxShift || pxY >= tileSize + 2 * maxShift || tileIdx >= tileCount)
		return;

	int pxInTilesArray = tileIdx * (tileSize + maxShift * 2) * (tileSize + maxShift * 2) + pxY * (tileSize + maxShift * 2) + pxX;
	//copy lines of block to shared memory
	int zOffset = threadIdx.z * (tileSize + 2 * maxShift);
	shared[zOffset + pxX] = inTiles[pxInTilesArray];

	__syncthreads();

	float outVal = 0;
	if (pxX >= tileSize / 2 && pxX <= maxShift * 2 + tileSize / 2)
	{
		for (int shift = -tileSize / 2; shift < tileSize / 2; shift++)
		{ 
			outVal += shared[zOffset + pxX + shift] * shared[zOffset + pxX + shift];
		}
	}
	outTiles[pxInTilesArray] = outVal;
}
//Boxfilter ignoring the border parts
//blockDim.Y must be tileSize + 2 * maxShift
//blockDim.X must be 1
extern "C"
__global__ void boxFilterWithBorderY(
	const float* __restrict__ inTiles,
	float* __restrict__ outTiles,
	int maxShift,
	int tileSize,
	int tileCount)
{
	int pxX = blockIdx.x * blockDim.x + threadIdx.x;
	int pxY = blockIdx.y * blockDim.y + threadIdx.y;
	int tileIdx = blockIdx.z * blockDim.z + threadIdx.z;

	extern __shared__ float shared[]; //sizeof blockDim.z * (2 * maxShift + tileSize)

	if (pxX >= tileSize + 2 * maxShift || pxY >= tileSize + 2 * maxShift || tileIdx >= tileCount)
		return;

	int pxInTilesArray = tileIdx * (tileSize + maxShift * 2) * (tileSize + maxShift * 2) + pxY * (tileSize + maxShift * 2) + pxX;
	//copy lines of block to shared memory
	int zOffset = threadIdx.z * (tileSize + 2 * maxShift);
	shared[zOffset + pxY] = inTiles[pxInTilesArray];

	__syncthreads();

	float outVal = 0;
	if (pxY >= tileSize / 2 && pxY <= maxShift * 2 + tileSize / 2)
	{
		for (int shift = -tileSize / 2; shift < tileSize / 2; shift++)
		{
			outVal += shared[zOffset + pxY + shift];// *shared[zOffset + pxY + shift];
		}
	}
	outTiles[pxInTilesArray] = outVal;
}


//Computed the normalized CC values out of the different input data
//Cross correlation is fft shifted
//blockDim.X must be 2 * maxShift
//blockDim.Y must be 2 * maxShift
//blockDim.Z must be nr of tiles
extern "C"
__global__ void normalizedCC(
	const float* __restrict__ ccImage,
	const float* __restrict__ squaredTemplate,
	const float* __restrict__ boxFilteredImage,
	float* __restrict__ shiftImage,
	int maxShift,
	int tileSize,
	int tileCount)
{
	int pxX = blockIdx.x * blockDim.x + threadIdx.x;
	int pxY = blockIdx.y * blockDim.y + threadIdx.y;
	int tileIdx = blockIdx.z * blockDim.z + threadIdx.z;

	if (pxX > 2 * maxShift || pxY > 2 * maxShift || tileIdx >= tileCount)
		return;

	int shiftX = pxX - maxShift;
	int shiftY = pxY - maxShift;
	int fftShiftX = shiftX;
	int fftShiftY = shiftY;

	if (fftShiftX < 0)
		fftShiftX = tileSize + 2 * maxShift + shiftX; //fftShift
	if (fftShiftY < 0)
		fftShiftY = tileSize + 2 * maxShift + shiftY; //fftShift


	int pxInCCArray = tileIdx * (tileSize + maxShift * 2) * (tileSize + maxShift * 2) + fftShiftY * (tileSize + maxShift * 2) + fftShiftX;
	int pxInBoxFilter = tileIdx * (tileSize + maxShift * 2) * (tileSize + maxShift * 2) + ((tileSize + maxShift * 2) / 2 + shiftY) * (tileSize + maxShift * 2) + ((tileSize + maxShift * 2) / 2 + shiftX);
	int pxOut = tileIdx * ((maxShift * 2 + 1) * (maxShift * 2 + 1)) + pxY * (maxShift * 2 + 1) + pxX;

	shiftImage[pxOut] = squaredTemplate[tileIdx] + boxFilteredImage[pxInBoxFilter] - 2 * ccImage[pxInCCArray];
}

//Convert a tiled image into consecutive tiles for FFT
//input img has a pitch, output tiles are consecutive
//output tiles overlap by maxShift is filled by zero
extern "C"
__global__ void convertToTilesOverlapBorder(
	const float* __restrict__ inImg,
	float* __restrict__ outTiles,
	int imgWidth,
	int imgHeight,
	int imgPitch,
	int maxShift,
	int tileSize,
	int tileCountX,
	int tileCountY,
	float2 baseShift,
	float baseRotation)
{
	int pxX = blockIdx.x * blockDim.x + threadIdx.x;
	int pxY = blockIdx.y * blockDim.y + threadIdx.y;
	int tileIdx = blockIdx.z * blockDim.z + threadIdx.z;

	if (pxX >= 2 * maxShift + tileSize || pxY >= 2 * maxShift + tileSize || tileIdx >= tileCountX * tileCountY)
		return;

	int pxInTilesArray = tileIdx * (tileSize + maxShift * 2) * (tileSize + maxShift * 2) + pxY * (tileSize + maxShift * 2) + pxX;
	if (pxX < maxShift || pxY < maxShift || pxX >= tileSize + maxShift || pxY >= tileSize + maxShift)
	{
		outTiles[pxInTilesArray] = 0;
		return;
	}

	int tileIdxY = tileIdx / tileCountX; //floor integer division
	int tileIdxX = tileIdx - (tileIdxY * tileCountX);

	//add base shift and rotation
	float2 shift;
	/*shift.x = -baseShift.x;
	shift.y = -baseShift.y;*/
	float sf = sinf(baseRotation);
	float cf = cosf(baseRotation);
	shift.x = cf * -baseShift.x - sf * -baseShift.y;
	shift.y = sf * -baseShift.x + cf * -baseShift.y;

	float patchCenterX = tileIdxX * tileSize + tileSize / 2 - imgWidth / 2; //in pixels
	float patchCenterY = tileIdxY * tileSize + tileSize / 2 - imgHeight / 2;

	shift.x += cf * patchCenterX - sf * patchCenterY - patchCenterX;
	shift.y += sf * patchCenterX + cf * patchCenterY - patchCenterY;

	int pxInImgX = tileIdxX * tileSize + pxX + (int)roundf(shift.x);
	int pxInImgY = tileIdxY * tileSize + pxY + (int)roundf(shift.y);
	pxInImgX = fminf(fmaxf(pxInImgX, 0), imgWidth - 1);
	pxInImgY = fminf(fmaxf(pxInImgY, 0), imgHeight - 1);

	const float* line = (const float*)((const char*)inImg + imgPitch * pxInImgY);
	float pixel = line[pxInImgX];
	outTiles[pxInTilesArray] = pixel;
}

//Convert a tiled image into consecutive tiles for FFT
//input img has a pitch, output tiles are consecutive
//output tiles overlap by maxShift on each side
extern "C"
__global__ void convertToTilesOverlapPreShift(
	const float* __restrict__ inImg,
	float* __restrict__ outTiles,
	const float2* __restrict__ preShift,
	int preShiftPitch,
	int imgWidth,
	int imgHeight,
	int imgPitch,
	int maxShift,
	int tileSize,
	int tileCountX,
	int tileCountY,
	float2 baseShift,
	float baseRotation)
{
	int pxX = blockIdx.x * blockDim.x + threadIdx.x;
	int pxY = blockIdx.y * blockDim.y + threadIdx.y;
	int tileIdx = blockIdx.z * blockDim.z + threadIdx.z;

	if (pxX >= 2 * maxShift + tileSize || pxY >= 2 * maxShift + tileSize || tileIdx >= tileCountX * tileCountY)
		return;


	int tileIdxY = tileIdx / tileCountX; //floor integer division
	int tileIdxX = tileIdx - (tileIdxY * tileCountX);

	const float2* lineShift = (const float2*)((const char*)preShift + preShiftPitch * tileIdxY);
	float2 shift = lineShift[tileIdxX];
	/*shift.x = -shift.x;
	shift.y = -shift.y;*/

	//add base shift and rotation
	/*shift.x -= baseShift.x;
	shift.y -= baseShift.y;*/
	float sf = sinf(baseRotation);
	float cf = cosf(baseRotation);
	shift.x += cf * -baseShift.x - sf * -baseShift.y;
	shift.y += sf * -baseShift.x + cf * -baseShift.y;

	float patchCenterX = tileIdxX * tileSize + tileSize / 2 - imgWidth / 2; //in pixels
	float patchCenterY = tileIdxY * tileSize + tileSize / 2 - imgHeight / 2;

	shift.x += cf * patchCenterX - sf * patchCenterY - patchCenterX;
	shift.y += sf * patchCenterX + cf * patchCenterY - patchCenterY;

	int pxInImgX = tileIdxX * tileSize + pxX + (int)roundf(shift.x);
	int pxInImgY = tileIdxY * tileSize + pxY + (int)roundf(shift.y);
	pxInImgX = fminf(fmaxf(pxInImgX, 0), imgWidth - 1);
	pxInImgY = fminf(fmaxf(pxInImgY, 0), imgHeight - 1);

	const float* line = (const float*)((const char*)inImg + imgPitch * pxInImgY);
	float pixel = line[pxInImgX];
	int pxInTilesArray = tileIdx * (tileSize + maxShift * 2) * (tileSize + maxShift * 2) + pxY * (tileSize + maxShift * 2) + pxX;
	outTiles[pxInTilesArray] = pixel;
}

__device__ float applysRGBGamma(float valIn)
{
	if (valIn <= 0.0031308f)
	{
		return 12.92f * valIn;
	}
	else
	{
		return (1.0f + 0.055f) * powf(valIn, 1.0f / 2.4f) - 0.055f;
	}
}

extern "C"
__global__ void GammasRGB(
	float3 * __restrict__ inOutImg,
	int imgWidth,
	int imgHeight,
	int imgPitch)
{
	int pxX = blockIdx.x * blockDim.x + threadIdx.x;
	int pxY = blockIdx.y * blockDim.y + threadIdx.y;

	if (pxX >= imgWidth || pxY >= imgHeight)
		return;

	float3 val = *(((float3*)((char*)inOutImg + imgPitch * pxY)) + pxX);
	//apply gamma:
	if (isnan(val.x))
		val.x = 0;
	if (isnan(val.y))
		val.y = 0;
	if (isnan(val.z))
		val.z = 0;
	
	val.x = fmaxf(fminf(val.x, 1.0f), 0.0f);
	val.y = fmaxf(fminf(val.y, 1.0f), 0.0f);
	val.z = fmaxf(fminf(val.z, 1.0f), 0.0f);

	val.x = applysRGBGamma(val.x);
	val.y = applysRGBGamma(val.y);
	val.z = applysRGBGamma(val.z);
	*(((float3*)((char*)inOutImg + imgPitch * pxY)) + pxX) = val;
}


extern "C"
__global__ void ApplyWeighting(
	float3 * __restrict__ inOutImg,
	const float3 * __restrict__ finalImg,
	const float3 * __restrict__ weight,
	int imgWidth,
	int imgHeight,
	int imgPitch,
	float threshold)
{
	int pxX = blockIdx.x * blockDim.x + threadIdx.x;
	int pxY = blockIdx.y * blockDim.y + threadIdx.y;

	if (pxX >= imgWidth || pxY >= imgHeight)
		return;


	float3 inout = *(((float3*)((char*)inOutImg + imgPitch * pxY)) + pxX);
	float3 val = *(((float3*)((char*)finalImg + imgPitch * pxY)) + pxX);
	float3 w = *(((float3*)((char*)weight + imgPitch * pxY)) + pxX);

	
	if (w.x < threshold)
	{
		val.x += inout.x;
		w.x += 1;
	}
	inout.x = 0;
	if (w.x != 0)
	{
		inout.x = val.x / w.x;
	}
	
	if (w.y < threshold)
	{
		val.y += inout.y;
		w.y += 1;
	}
	inout.y = 0;
	if (w.y != 0)
	{
		inout.y = val.y / w.y;
	}
	
	if (w.z < threshold)
	{
		val.z += inout.z;
		w.z += 1;
	}
	inout.z = 0;
	if (w.z != 0)
	{
		inout.z = val.z / w.z;
	}
	
	*(((float3*)((char*)inOutImg + imgPitch * pxY)) + pxX) = inout;
}


extern "C"
__global__ void conjugateComplexMulKernel(const float2* __restrict__ aIn, float2* __restrict__ bInOut, int maxElem)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= maxElem)
		return;

	float2 valA = aIn[idx];
	//conjugate complex
	valA.y = -valA.y;

	//multiplication:
	float2 valB = bInOut[idx];
	float2 res;
	res.x = valA.x * valB.x - valA.y * valB.y;
	res.y = valA.x * valB.y + valA.y * valB.x;
	bInOut[idx] = res;
}

__constant__ float FA11[]{ 1.0f / 4.0f, -2.0f / 4.0f, 1.0f / 4.0f, 2.0f / 4.0f, -4.0f / 4.0f, 2.0f / 4.0f, 1.0f / 4.0f, -2.0f / 4.0f, 1.0f / 4.0f };
__constant__ float FA22[]{ 1.0f / 4.0f, 2.0f / 4.0f, 1.0f / 4.0f, -2.0f / 4.0f, -4.0f / 4.0f, -2.0f / 4.0f, 1.0f / 4.0f, 2.0f / 4.0f, 1.0f / 4.0f };
__constant__ float FA12[]{ 1.0f / 4.0f, 0.0f / 4.0f, -1.0f / 4.0f, 0.0f / 4.0f, 0.0f / 4.0f, 0.0f / 4.0f, -1.0f / 4.0f, 0.0f / 4.0f, 1.0f / 4.0f };
__constant__ float Fb1[]{ -1.0f / 8.0f, 0.0f / 8.0f, 1.0f / 8.0f, -2.0f / 8.0f, 0.0f / 8.0f, 2.0f / 8.0f, -1.0f / 8.0f, 0.0f / 8.0f, 1.0f / 8.0f };
__constant__ float Fb2[]{ -1.0f / 8.0f, -2.0f / 8.0f, -1.0f / 8.0f, 0.0f / 8.0f, 0.0f / 8.0f, 0.0f / 8.0f, 1.0f / 8.0f, 2.0f / 8.0f, 1.0f / 8.0f };

//finds minimum peak in shift image and returns the interpolated coordinate.
//blockDim.X must be number of Tiles
extern "C"
__global__ void findMinimum(
	const float* __restrict__ shiftImage,
	float2* __restrict__ coordinates,
	int coordinatesPitch,
	int maxShift,
	int tileCount,
	int tileCountX,
	float threshold)
{
	int tileIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (tileIdx >= tileCount)
		return;
		
	int pixelsInTile = (2 * maxShift + 1) * (2 * maxShift + 1);
	int zOffset = tileIdx * pixelsInTile;

	float minVal = FLT_MAX;
	float maxVal = -FLT_MAX;
	int minIdx = -1;

	for (int i = 0; i < pixelsInTile; i++)
	{
		float val = shiftImage[zOffset + i];
		maxVal = fmaxf(maxVal, val);
		if (val < minVal)
		{
			minVal = val;
			minIdx = i;
		}
	}

	float2 coord;
	coord.y = (int)minIdx / (int)(2 * maxShift + 1);
	coord.x = minIdx - coord.y * (2 * maxShift + 1);


	if (coord.x < 1 || coord.y < 1 || coord.x >= 2 * maxShift || coord.y >= 2 * maxShift)
	{
		//we can't interpolate at the border :(
		coord.x = 0;
		coord.y = 0;
	}
	else
	{
		//Interpolate:

		float A11 = 0;
		float A22 = 0;
		float A12 = 0;
		float b1 = 0;
		float b2 = 0;

		for (int i = 0; i < 3; i++)
		{
			float img = shiftImage[zOffset + minIdx + i - 1 - (2 * maxShift + 1)];
			A11 += FA11[i] * img;
			A22 += FA22[i] * img;
			A12 += FA12[i] * img;
			b1 += Fb1[i] * img;
			b2 += Fb2[i] * img;
		}
		for (int i = 3; i < 6; i++)
		{
			float img = shiftImage[zOffset + minIdx + i - 4];
			A11 += FA11[i] * img;
			A22 += FA22[i] * img;
			A12 += FA12[i] * img;
			b1 += Fb1[i] * img;
			b2 += Fb2[i] * img;
		}
		for (int i = 6; i < 9; i++)
		{
			float img = shiftImage[zOffset + minIdx + i - 7 + (2 * maxShift + 1)];
			A11 += FA11[i] * img;
			A22 += FA22[i] * img;
			A12 += FA12[i] * img;
			b1 += Fb1[i] * img;
			b2 += Fb2[i] * img;
		}

		A11 = fmaxf(A11, 0.0f);
		A22 = fmaxf(A22, 0.0f);

		float detA = A11 * A22 - A12 * A12;
		if (detA < 0)
		{
			A12 = 0;
			detA = A11 * A22;
		}

		if (detA != 0)
		{
			float muX = (A22 * b1 - A12 * b2) / detA;
			float muY = (A11 * b2 - A12 * b1) / detA;

			if (fabs(muX) > 1)
			{
				muX = 0;
			}
			if (fabs(muY) > 1)
			{
				muY = 0;
			}

			coord.x -= muX;
			coord.y -= muY;
		}

		coord.x -= maxShift;
		coord.y -= maxShift;
	}
	int tileIdxY = tileIdx / tileCountX; //floor integer division
	int tileIdxX = tileIdx - (tileIdxY * tileCountX);

	//float2 shift = lineShift[tileIdxX];


	if (threshold + minVal > maxVal)
	{
		coord.x = 0;
		coord.y = 0;
	}

	*(((float2*)((char*)coordinates + coordinatesPitch * tileIdxY)) + tileIdxX) = coord;	
}




extern "C"
__global__ void UpSampleShifts(
	const float2* __restrict__ inShift,
	float2* __restrict__ outShift, int inPitch, int outPitch, int oldLevel, int newLevel, int oldCountX, int oldCountY, int newCountX, int newCountY, int oldTileSize, int newTileSize)
{
	int newBlockX = blockIdx.x * blockDim.x + threadIdx.x;
	int newBlockY = blockIdx.y * blockDim.y + threadIdx.y;

	if (newBlockX >= newCountX || newBlockY >= newCountY)
		return;

	float factor = (float)oldLevel * oldTileSize / (float)(newLevel * newTileSize);

	
	int newIdx = newBlockY * newCountX + newBlockX;

	float oldX = newBlockX / factor;
	float oldY = newBlockY / factor;


	int oldXMin = floor(oldX);
	int oldXMax = ceil(oldX);
	int oldYMin = floor(oldY);
	int oldYMax = ceil(oldY);
	oldXMin = min(oldXMin, oldCountX - 1);
	oldXMax = min(oldXMax, oldCountX - 1);
	oldYMin = min(oldYMin, oldCountY - 1);
	oldYMax = min(oldYMax, oldCountY - 1);
	//int oldIdx = oldY * oldCountX + oldX;

	float2 oldMinMin = *(((const float2*)((const char*)inShift + inPitch * oldYMin)) + oldXMin);
	float2 oldMaxMin = *(((const float2*)((const char*)inShift + inPitch * oldYMin)) + oldXMax);
	float2 oldMinMax = *(((const float2*)((const char*)inShift + inPitch * oldYMax)) + oldXMin);
	float2 oldMaxMax = *(((const float2*)((const char*)inShift + inPitch * oldYMax)) + oldXMax);

	float temp1 = oldMinMin.x + (oldMaxMin.x - oldMinMin.x) * (1.0f - (oldXMax - oldX));
	float temp2 = oldMinMax.x + (oldMaxMax.x - oldMinMax.x) * (1.0f - (oldXMax - oldX));
	float2 old;
	old.x = temp1 + (temp2 - temp1) * (1.0f - (oldYMax - oldY));
	temp1 = oldMinMin.y + (oldMaxMin.y - oldMinMin.y) * (1.0f - (oldXMax - oldX));
	temp2 = oldMinMax.y + (oldMaxMax.y - oldMinMax.y) * (1.0f - (oldXMax - oldX));
	old.y = temp1 + (temp2 - temp1) * (1.0f - (oldYMax - oldY));

	old.x *= oldLevel / (float)newLevel;
	old.y *= oldLevel / (float)newLevel;
	float2* lineOutShift = (float2*)((char*)outShift + outPitch * newBlockY);
	lineOutShift[newBlockX] = old;
}

extern "C"
__global__ void ComputeStructureTensor(
	const float* __restrict__ imgDx,
	const float* __restrict__ imgDy,
	float3* __restrict__ outImg,
	int imgWidth,
	int imgHeight,
	int imgDxDyPitch,
	int imgOutPitch)
{
	int pxX = blockIdx.x * blockDim.x + threadIdx.x;
	int pxY = blockIdx.y * blockDim.y + threadIdx.y;

	if (pxX >= imgWidth || pxY >= imgHeight)
		return;

	

	float dx = *(((const float*)((const char*)imgDx + imgDxDyPitch * pxY)) + pxX);
	float dy = *(((const float*)((const char*)imgDy + imgDxDyPitch * pxY)) + pxX);
	float3 val;
	val.x = dx * dx;
	val.y = dy * dy;
	val.z = dx * dy;
	*(((float3*)((char*)outImg + imgOutPitch * pxY)) + pxX) = val;
}

extern "C"
__global__ void ComputeKernelParam(
	float3* __restrict__ kernelImg,
	int imgWidth,
	int imgHeight,
	int imgOutPitch,
	float Dth,
	float Dtr,
	float kDetail,
	float kDenoise,
	float kStretch,
	float kShrink)
{
	int pxX = blockIdx.x * blockDim.x + threadIdx.x;
	int pxY = blockIdx.y * blockDim.y + threadIdx.y;

	if (pxX >= imgWidth || pxY >= imgHeight)
		return;

	float3 grad = *(((float3*)((char*)kernelImg + imgOutPitch * pxY)) + pxX);
	float a11 = grad.x;
	float a22 = grad.y;
	float a12 = grad.z;

	float help = sqrtf((a22 - a11) * (a22 - a11) + 4.0f * a12 * a12);
	float c = 2.0f * a12;
	float s = a22 - a11 + help;

	float norm = sqrtf(c * c + s * s);
	if (norm > 0)
	{
		c /= norm;
		s /= norm;
	}
	else
	{
		c = 1;
		s = 0;
	}

	float lam1 = (a11 + a22 + help) / 2.0f;
	float lam2 = (a11 + a22 - help) / 2.0f;


	float A = 1 + sqrtf((lam1 - lam2) * (lam1 - lam2) / ((lam1 + lam2) * (lam1 + lam2)));
	float D = 1 - sqrtf(lam1) / Dtr + Dth;

	D = fmaxf(fminf(1.0f, D), 0.0f);

	float k1h = kDetail * kStretch * A;
	float k2h = kDetail / kShrink * A;

	float k1 = ((1.0f - D)*k1h + D*kDetail*kDenoise);
	float k2 = ((1.0f - D)*k2h + D*kDetail*kDenoise);
	k1 *= k1;
	k2 *= k2;

	float x2 = c;
	float y2 = s;
	float x1 = s;
	float y1 = -c;

	float b11 = k1*x1*x1 + x2*x2*k2;
	float b12 = k1*x1*y1 + x2*y2*k2;
	float b22 = k1*y1*y1 + y2*y2*k2;

	float det = b11*b22 - b12*b12 + 0.0000000001f;

	float3 kernel;
	kernel.x = b22 / det;
	kernel.y = b11 / det;
	kernel.z = -b12 / det;
	*(((float3*)((char*)kernelImg + imgOutPitch * pxY)) + pxX) = kernel;
}

extern "C"
__global__
void fourierFilter(float2 * img, size_t stride, int width, int height, float lp, float hp, float lps, float hps, int clearAxis)
{
	//compute x,y indices 
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width / 2 + 1) return;
	if (y >= height) return;
	//if (x == 0 && y == 0) return; //don't change mean value

	float mx = (float)x;
	float my = (float)y;
	if (my > height * 0.5f)
		my = (height - my) * -1.0f;

	mx /= width;
	my /= height;

	float dist = sqrtf(mx * mx + my * my);
	float fil = 0;

	lp = lp - lps;
	hp = hp + hps;

	//Low pass
	if (lp > 0)
	{
		if (dist <= lp) fil = 1;
	}
	else
	{
		if (dist <=  1.0f) fil = 1;
	}
	//Gauss
	if (lps > 0)
	{
		float fil2;
		if (dist < lp) fil2 = 1;
		else fil2 = 0;

		fil2 = (-fil + 1.0f) * (float)expf(-((dist - lp) * (dist - lp) / (2 * lps * lps)));
		if (fil2 > 0.001f)
			fil = fil2;
	}

	if (lps > 0 && lp == 0 && hp == 0 && hps == 0)
		fil = (float)expf(-((dist - lp) * (dist - lp) / (2 * lps * lps)));

	if (hp > 0)
	{
		float fil2 = 0;
		if (dist >= hp) fil2 = 1;

		fil *= fil2;

		if (hps > 0)
		{
			float fil3 = 0;
			if (dist < hp) fil3 = 1;
			fil3 = (-fil2 + 1.0f) * (float)expf(-((dist - hp) * (dist - hp) / (2 * hps * hps)));
			if (fil3 > 0.001f)
				fil = fil3;
		}
	}

	float2* row = (float2*)((char*)img + stride * y);
	float2 erg = row[x];
	erg.x *= fil;
	erg.y *= fil;
	if (x < clearAxis || fabs(my)*height < clearAxis)
	{
		erg.x = 0;
		erg.y = 0;
	}
	row[x] = erg;
}

extern "C"
__global__
void fftshift(float2 * fft, int width, int height)
{
	//compute x,y indiced 
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width) return;
	if (y >= height) return;

	int mx = x - width / 2;
	int my = y - height / 2;

	float a = 1.0f - 2 * (((mx + my) & 1));

	float2 erg = fft[y * width + x];
	erg.x *= a;
	erg.y *= a;
	fft[y * width + x] = erg;
}