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
__global__ void WarpingKernel(int width, int height, int stride,
	cudaTextureObject_t texUV, float* __restrict__ out, cudaTextureObject_t texToWarp)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;
	
	if (ix >= width || iy >= height) return;

	float2 shift = tex2D<float2>(texUV, ((float)ix + 0.5f) / (float)width, ((float)iy + 0.5f) / (float)height);

	float x = ((float)ix + 0.5f + shift.x) / (float)width;
	float y = ((float)iy + 0.5f + shift.y) / (float)height;

	float pixel = tex2D<float>(texToWarp, x, y);

	*(((float*)((char*)out + stride * (iy))) + (ix)) = pixel;
}


extern "C"
__global__ void CreateFlowFieldFromTiles(
	float2* __restrict__ outImg,
	cudaTextureObject_t texObjShiftXY,
	int tileSize,
	int tileCountX,
	int tileCountY,
	int imgWidth,
	int imgHeight,
	int imgPitch,
	float2 baseShift,
	float baseRotation)
{
	int pxX = blockIdx.x * blockDim.x + threadIdx.x;
	int pxY = blockIdx.y * blockDim.y + threadIdx.y;

	if (pxX >= imgWidth || pxY >= imgHeight)
		return;
	
	int tileX = pxX / tileSize;
	int tileY = pxY / tileSize;

	tileX = min(tileX, tileCountX - 1);
	tileY = min(tileY, tileCountY - 1);

	int tileIdx = tileX + tileY * tileCountX;
	float2 shift;
	shift.x = 0;
	shift.y = 0;
	
	//add base shift and rotation
	shift.x = cosf(baseRotation) * -baseShift.x - sinf(baseRotation) * -baseShift.y;
	shift.y = sinf(baseRotation) * -baseShift.x + cosf(baseRotation) * -baseShift.y;

	float patchCenterX = pxX - imgWidth / 2; //in pixels
	float patchCenterY = pxY - imgHeight / 2;

	shift.x += cosf(baseRotation) * patchCenterX - sinf(baseRotation) * patchCenterY - patchCenterX;
	shift.y += sinf(baseRotation) * patchCenterX + cosf(baseRotation) * patchCenterY - patchCenterY;


	float2 shiftPatch = tex2D<float2>(texObjShiftXY, (pxX + 0.5f) / (float)imgWidth, (pxY + 0.5f) / (float)imgHeight);
	shift.x += shiftPatch.x;
	shift.y += shiftPatch.y;
	
	*(((float2*)((char*)outImg + imgPitch * pxY)) + pxX) = shift;
}


extern "C"
__global__ void ComputeDerivativesKernel(int width, int height, int stride,
	float* Ix, float* Iy, float* Iz,
	cudaTextureObject_t texSource,
	cudaTextureObject_t texTarget)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;


	if (ix >= width || iy >= height) return;

	float dx = 1.0f / (float)width;
	float dy = 1.0f / (float)height;

	float x = ((float)ix + 0.5f) * dx;
	float y = ((float)iy + 0.5f) * dy;

	float t0, t1;
	// x derivative
	t0 = tex2D<float>(texSource, x + 2.0f * dx, y);
	t0 -= tex2D<float>(texSource, x + 1.0f * dx, y) * 8.0f;
	t0 += tex2D<float>(texSource, x - 1.0f * dx, y) * 8.0f;
	t0 -= tex2D<float>(texSource, x - 2.0f * dx, y);
	t0 /= 12.0f;

	t1 = tex2D<float>(texTarget, x + 2.0f * dx, y);
	t1 -= tex2D<float>(texTarget, x + 1.0f * dx, y) * 8.0f;
	t1 += tex2D<float>(texTarget, x - 1.0f * dx, y) * 8.0f;
	t1 -= tex2D<float>(texTarget, x - 2.0f * dx, y);
	t1 /= 12.0f;

	*(((float*)((char*)Ix + stride * iy)) + ix) = (t0 + t1) * 0.5f;

	// t derivative
	*(((float*)((char*)Iz + stride * iy)) + ix) = tex2D<float>(texSource, x, y) - tex2D<float>(texTarget, x, y);

	// y derivative
	t0 = tex2D<float>(texSource, x, y + 2.0f * dy);
	t0 -= tex2D<float>(texSource, x, y + 1.0f * dy) * 8.0f;
	t0 += tex2D<float>(texSource, x, y - 1.0f * dy) * 8.0f;
	t0 -= tex2D<float>(texSource, x, y - 2.0f * dy);
	t0 /= 12.0f;

	t1 = tex2D<float>(texTarget, x, y + 2.0f * dy);
	t1 -= tex2D<float>(texTarget, x, y + 1.0f * dy) * 8.0f;
	t1 += tex2D<float>(texTarget, x, y - 1.0f * dy) * 8.0f;
	t1 -= tex2D<float>(texTarget, x, y - 2.0f * dy);
	t1 /= 12.0f;

	*(((float*)((char*)Iy + stride * iy)) + ix) = (t0 + t1) * 0.5f;
}


extern "C"
__global__ void ComputeDerivatives2Kernel(int width, int height, int stride,
	float* Ix, float* Iy,
	cudaTextureObject_t tex)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;
	
	if (ix >= width || iy >= height) return;

	float dx = 1.0f / (float)width;
	float dy = 1.0f / (float)height;

	float x = ((float)ix + 0.5f) * dx;
	float y = ((float)iy + 0.5f) * dy;

	float t0, t1;
	// x derivative
	t0 = tex2D<float>(tex, x + 2.0f * dx, y);
	t0 -= tex2D<float>(tex, x + 1.0f * dx, y) * 8.0f;
	t0 += tex2D<float>(tex, x - 1.0f * dx, y) * 8.0f;
	t0 -= tex2D<float>(tex, x - 2.0f * dx, y);
	t0 /= 12.0f;

	*(((float*)((char*)Ix + stride * iy)) + ix) = t0;


	// y derivative
	t0 = tex2D<float>(tex, x, y + 2.0f * dy);
	t0 -= tex2D<float>(tex, x, y + 1.0f * dy) * 8.0f;
	t0 += tex2D<float>(tex, x, y - 1.0f * dy) * 8.0f;
	t0 -= tex2D<float>(tex, x, y - 2.0f * dy);
	t0 /= 12.0f;

	*(((float*)((char*)Iy + stride * iy)) + ix) = t0;
}



extern "C"
__global__ void lucasKanadeOptim(
	float2 * __restrict__ shifts,
	const float* __restrict__ imFx,
	const float* __restrict__ imFy,
	const float* __restrict__ imFt,
	int pitchShift,
	int pitchImg,
	int width,
	int height,
	int halfWindowSize,
	float minDet)
{
	int pxX = blockIdx.x * blockDim.x + threadIdx.x;
	int pxY = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (pxX < halfWindowSize || pxX >= width - halfWindowSize ||
		pxY < halfWindowSize || pxY >= height - halfWindowSize)
		return;

	int windowSize = halfWindowSize * 2 + 1;
	float matMul[4];
	float matMulInv[4];
	float UT[4];
	float S[4];
	float V[4];

	float UV[2];

	matMul[0] = matMul[1] = matMul[2] = matMul[3] = 0;
	for (int y = -halfWindowSize; y <= halfWindowSize; y++)
	{
		for (int x = -halfWindowSize; x <= halfWindowSize; x++)
		{
			int globalX = pxX + x;
			int globalY = pxY + y;
			
			float dx = *(((float*)(((char*)imFx) + globalY * pitchImg)) + globalX);
			float dy = *(((float*)(((char*)imFy) + globalY * pitchImg)) + globalX);

			matMul[0] += dx * dx;
			matMul[1] += dx * dy;
			matMul[3] += dy * dy;
		}
	}
	matMul[2] = matMul[1];

	//matrix pseudo inversion:
	float a = matMul[0];
	float b = matMul[1];
	float c = matMul[2];
	float d = matMul[3];

	float theta = 0.5f * atan2(2.0f * a * c + 2.0f * b * d, a * a + b * b - c * c - d * d);
	float ct = cos(theta);
	float st = sin(theta);
	UT[0] = ct;
	UT[2] = -st; //transposed
	UT[1] = st;  //transposed
	UT[3] = ct;

	float S1 = a * a + b * b + c * c + d * d;
	float S2 = sqrtf((a * a + b * b - c * c - d * d) * (a * a + b * b - c * c - d * d) + 4 * (a * c + b * d) * (a * c + b * d));
	float sigma1 = sqrt((S1 + S2) / 2);
	float sigma2 = sqrt((S1 - S2) / 2);

	float smin = fminf(sigma1, sigma1);
	if (smin < minDet)
		return;


	sigma1 = sigma1 != 0 ? 1.0f / sigma1 : 0;
	sigma2 = sigma2 != 0 ? 1.0f / sigma2 : 0;

	S[0] = sigma1;
	S[1] = 0;
	S[2] = 0;
	S[3] = sigma2;

	float epsilon = 0.5f * atan2(2.0f * a * b + 2.0f * c * d, a * a - b * b + c * c - d * d);

	float ce = cos(epsilon);
	float se = sin(epsilon);

	float s11 = (a * ct + c * st) * ce + (b * ct + d * st) * se;
	float s22 = (a * st - c * ct) * se + (-b * st + d * ct) * ce;

	s11 = s11 > 0.0f ? 1.0f : s11 < 0 ? -1.0f : 0.0f;
	s22 = s22 > 0.0f ? 1.0f : s22 < 0 ? -1.0f : 0.0f;

	V[0] = s11 * ce;
	V[1] = -s22 * se;
	V[2] = s11 * se;
	V[3] = s22 * ce;

	matMul[0] = S[0] * UT[0] + S[1] * UT[2];
	matMul[1] = S[0] * UT[1] + S[1] * UT[3];
	matMul[2] = S[2] * UT[0] + S[3] * UT[2];
	matMul[3] = S[2] * UT[1] + S[3] * UT[3];

	matMulInv[0] = V[0] * matMul[0] + V[1] * matMul[2];
	matMulInv[1] = V[0] * matMul[1] + V[1] * matMul[3];
	matMulInv[2] = V[2] * matMul[0] + V[3] * matMul[2];
	matMulInv[3] = V[2] * matMul[1] + V[3] * matMul[3];
	
	int ws2 = windowSize * windowSize;
	UV[0] = 0;
	UV[1] = 0;

	for (int i = 0; i < ws2; i++)
	{
		int y = i / windowSize;
		int x = i - (y * windowSize);

		int globalX = pxX + x - halfWindowSize;
		int globalY = pxY + y - halfWindowSize;

		float dx = *(((float*)(((char*)imFx) + globalY * pitchImg)) + globalX);
		float dy = *(((float*)(((char*)imFy) + globalY * pitchImg)) + globalX);

		float dt = *(((float*)(((char*)imFt) + globalY * pitchImg)) + globalX);

		UV[0] += (matMulInv[0] * dx + matMulInv[1] * dy) * dt;
		UV[1] += (matMulInv[2] * dx + matMulInv[3] * dy) * dt;
	}

	UV[0] = isnan(UV[0]) ? 0 : UV[0];
	UV[1] = isnan(UV[1]) ? 0 : UV[1];

	/*UV[0] = fmaxf(fminf(2.0f, UV[0]), -2.0f);
	UV[1] = fmaxf(fminf(2.0f, UV[1]), -2.0f);*/

	float2 shift = *(((float2*)(((char*)shifts) + pxY * pitchShift)) + pxX);
	shift.x += UV[0];
	shift.y += UV[1];
	*(((float2*)((char*)shifts + pxY * pitchShift)) + pxX) = shift;
}
