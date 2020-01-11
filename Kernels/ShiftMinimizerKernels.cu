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


extern "C"
__global__ void copyShiftMatrix(float* __restrict__ matrices, int tileCount, int imageCount, int shiftCount)
{
	int tile = blockIdx.x * blockDim.x + threadIdx.x;

	if (tile >= tileCount)
		return;

	//we copy the matrix of first tile to all other tiles...
	if (tile == 0)
		return;

	int matrixSize = (imageCount - 1) * shiftCount;

	int offset = (matrixSize) * tile;

	for (int elem = 0; elem < matrixSize; elem++)
	{
		matrices[offset + elem] = matrices[elem];
	}
}

extern "C"
__global__ void setPointers(float** __restrict__ shiftMatrixArray, float** __restrict__ shiftMatrixSafeArray, float** __restrict__ matrixSquareArray,
	float** __restrict__ matrixInvertedArray, float** __restrict__ solvedMatrixArray,
	float2**__restrict__ shiftOneToOneArray, float2** __restrict__ shiftMeasuredArray, float2** __restrict__ shiftOptimArray,
	float* shiftMatrices, float* shiftSafeMatrices, float* matricesSquared, float* matricesInverted, float* solvedMatrices,
	float2* shiftsOneToOne, float2* shiftsMeasured, float2* shiftsOptim, int tileCount, int imageCount, int shiftCount)
{
	int tile = blockIdx.x * blockDim.x + threadIdx.x;

	if (tile >= tileCount)
		return;

	int n1 = imageCount - 1;
	int m = shiftCount;

	int sizeShiftMatrix = n1 * m;
	int sizeSquared = n1 * n1;

	shiftMatrixArray[tile] = shiftMatrices + tile * sizeShiftMatrix;
	shiftMatrixSafeArray[tile] = shiftSafeMatrices + tile * sizeShiftMatrix;
	matrixSquareArray[tile] = matricesSquared + tile * sizeSquared;
	matrixInvertedArray[tile] = matricesInverted + tile * sizeSquared;
	solvedMatrixArray[tile] = solvedMatrices + tile * sizeShiftMatrix;
	shiftOneToOneArray[tile] = shiftsOneToOne + tile * n1;
	shiftOptimArray[tile] = shiftsOptim + tile * m;
	shiftMeasuredArray[tile] = shiftsMeasured + tile * m;
}



extern "C"
__global__ void checkForOutliers(
	float2* __restrict__ measuredShifts,
	const float* __restrict__ optimShiftsT,
	float* __restrict__ shiftMatrix,
	int* __restrict__ status,
	int* __restrict__ inversionInfo,
	int tileCount, int imageCount, int shiftCount)
{
	int tile = blockIdx.x * blockDim.x + threadIdx.x;

	if (tile >= tileCount)
		return;

	if (status[tile] < 0)
		return;

	if (inversionInfo[tile] != 0)
	{
		status[tile] = -1;
		return;
	}

	int n1 = imageCount - 1;
	int m = shiftCount; 
	int offsetMatrix = (n1 * m) * tile;
	int offsetAllVec = m * tile;
	

	float max = 1;
	int idxMax = -1;

	for (int i = 0; i < m; i++)
	{
		float distx = measuredShifts[offsetAllVec + i].x - optimShiftsT[2*offsetAllVec + i];
		float disty = measuredShifts[offsetAllVec + i].y - optimShiftsT[2*offsetAllVec + i + m];

		float dist = distx * distx + disty * disty;
		if (dist > max)
		{
			idxMax = i;
			max = dist;
		}
	}

	status[tile] = idxMax;
	//success: we found a meaningful minimum
	if (idxMax == -1)
	{
		return;
	}

	//remove the largest outlier
	measuredShifts[offsetAllVec + idxMax].x = 0;
	measuredShifts[offsetAllVec + idxMax].y = 0;
	for (int col = 0; col < n1; col++)
	{
		shiftMatrix[offsetMatrix + idxMax + col * m] = 0;
	}
}


extern "C"
__global__ void transposeShifts(
	float2 * __restrict__ measuredShifts,
	const float* __restrict__ measuredShiftsT,
	const float* __restrict__ shiftsOneToOneT,
	float2* __restrict__ shiftsOneToOne,
	int tileCount, int imageCount, int shiftCount)
{
	int tile = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if (tile >= tileCount)
		return;

	int n1 = imageCount - 1;
	int m = shiftCount;

	if (i >= m)
		return;

	int offsetAllVec = m * tile;

	float2 shift;
	shift.x = measuredShiftsT[2 * offsetAllVec + i];
	shift.y = measuredShiftsT[2 * offsetAllVec + i + m];
	measuredShifts[offsetAllVec + i] = shift;

	if (i >= n1)
		return;
	int offsetOneToOne = n1 * tile;
	float2 temp;
	temp.x = shiftsOneToOneT[2 * offsetOneToOne + i];
	temp.y = shiftsOneToOneT[2 * offsetOneToOne + i + n1];
	shiftsOneToOne[offsetOneToOne + i] = temp;
}

extern "C"
__global__ void getOptimalShifts(
	float2 * __restrict__ optimalShifts,
	const float2 * __restrict__ bestShifts,
	int imageCount,
	int tileCountX,
	int tileCountY,
	int optimalShiftsPitch,
	int referenceImage,
	int imageToTrack)
{
	int tileIdxX = blockIdx.x * blockDim.x + threadIdx.x;
	int tileIdxY = blockIdx.y * blockDim.y + threadIdx.y;

	if (tileIdxX >= tileCountX || tileIdxY >= tileCountY)
		return;

	int n1 = imageCount - 1;

	const float2* r = &bestShifts[(tileIdxX + tileIdxY * tileCountX) * n1];

	float2 totalShift = make_float2(0, 0);
	if (referenceImage < imageToTrack)
	{
		for (int i = referenceImage; i < imageToTrack; i++)
		{
			totalShift.x += r[i].x;
			totalShift.y += r[i].y;
		}
	}
	else if(imageToTrack < referenceImage)
	{
		for (int i = imageToTrack; i < referenceImage; i++)
		{
			totalShift.x -= r[i].x;
			totalShift.y -= r[i].y;
		}
	}

	*(((float2*)((char*)(optimalShifts) +optimalShiftsPitch * tileIdxY)) + tileIdxX) = totalShift;
}



extern "C"
__global__ void concatenateShifts(
	const float2* __restrict__ const* __restrict__ shiftIn,
	int* __restrict__ shiftInPitch,
	float2* __restrict__ shiftOut,
	int shiftCount,
	int tileCountX, int tileCountY)
{
	int shift = blockIdx.x * blockDim.x + threadIdx.x;
	int tileX = blockIdx.y * blockDim.y + threadIdx.y;
	int tileY = blockIdx.z * blockDim.z + threadIdx.z;

	if (tileX >= tileCountX || tileY >= tileCountY || shift >= shiftCount)
		return;

	const float2* line = (const float2*)((const char*)(shiftIn[shift]) + shiftInPitch[shift] * tileY);
	shiftOut[(tileX + tileY * tileCountX) * shiftCount + shift] = line[tileX];
}

extern "C"
__global__ void separateShifts(
	const float2* __restrict__ shiftIn,
	float2* __restrict__ const * __restrict__ shiftOut,
	int* __restrict__ shiftOutPitch,
	int shiftCount,
	int tileCountX, int tileCountY)
{
	int shift = blockIdx.x * blockDim.x + threadIdx.x;
	int tileX = blockIdx.y * blockDim.y + threadIdx.y;
	int tileY = blockIdx.z * blockDim.z + threadIdx.z;

	if (tileX >= tileCountX || tileY >= tileCountY|| shift >= shiftCount)
		return;

	float2* line = (float2*)((char*)(shiftOut[shift]) + shiftOutPitch[shift] * tileY);
	line[tileX] = shiftIn[(tileX + tileY * tileCountX) * shiftCount + shift];
}