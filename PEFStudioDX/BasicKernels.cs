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
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using ManagedCuda.NPP;

namespace PEFStudioDX
{
    public class convertToTilesOverlapKernel : CudaKernel
    {
        const string kernelName = "convertToTilesOverlapPreShift";
        dim3 blockSize = new dim3(24, 1, 8);
        public convertToTilesOverlapKernel(CudaContext ctx, CUmodule module)
            : base(kernelName, module, ctx)
        {
            this.BlockDimensions = blockSize;
        }

        public float RunSafe(NPPImage_32fC1 inImg, CudaDeviceVariable<float> outTiles, NPPImage_32fC2 preShift, int tileSize, int maxShift, int tileCountX, int tileCountY, float2 baseShift, float baseRotation)
        {
            this.SetComputeSize((uint)(tileSize + 2 * maxShift), (uint)(tileSize + 2 * maxShift), (uint)(tileCountX * tileCountY));
            return this.Run(inImg.DevicePointerRoi, outTiles.DevicePointer, preShift.DevicePointer, preShift.Pitch, inImg.WidthRoi, inImg.HeightRoi, inImg.Pitch, maxShift, tileSize, tileCountX, tileCountY, baseShift, baseRotation);
        }
    }

    public class convertToTilesOverlapBorderKernel : CudaKernel
    {
        const string kernelName = "convertToTilesOverlapBorder";
        dim3 blockSize = new dim3(24, 1, 8);
        public convertToTilesOverlapBorderKernel(CudaContext ctx, CUmodule module)
            : base(kernelName, module, ctx)
        {
            this.BlockDimensions = blockSize;
        }

        public float RunSafe(NPPImage_32fC1 inImg, CudaDeviceVariable<float> outTiles, int tileSize, int maxShift, int tileCountX, int tileCountY, float2 baseShift, float baseRotation)
        {
            this.SetComputeSize((uint)(tileSize + 2 * maxShift), (uint)(tileSize + 2 * maxShift), (uint)(tileCountX * tileCountY));
            return this.Run(inImg.DevicePointerRoi, outTiles.DevicePointer, inImg.WidthRoi, inImg.HeightRoi, inImg.Pitch, maxShift, tileSize, tileCountX, tileCountY, baseShift, baseRotation);
        }
    }

    public class conjugateComplexMulKernel : CudaKernel
    {
        const string kernelName = "conjugateComplexMulKernel";
        dim3 blockSize = new dim3(128, 1, 1);
        public conjugateComplexMulKernel(CudaContext ctx, CUmodule module)
            : base(kernelName, module, ctx)
        {
            this.BlockDimensions = blockSize;
        }

        public float RunSafe(CudaDeviceVariable<float2> dataIn, CudaDeviceVariable<float2> dataInOut)
        {
            int size = Math.Min(dataIn.Size, dataInOut.Size);
            this.SetComputeSize((uint)(size));
            return this.Run(dataIn.DevicePointer, dataInOut.DevicePointer, size);
        }
    }

    public class fourierFilterKernel : CudaKernel
    {
        //void fourierFilter(float2 * img, size_t stride, int width, int height, float lp, float hp, float lps, float hps, int clearAxis)

        const string kernelName = "fourierFilter";
        dim3 blockSize = new dim3(32, 32, 1);
        public fourierFilterKernel(CudaContext ctx, CUmodule module)
            : base(kernelName, module, ctx)
        {
            this.BlockDimensions = blockSize;
        }

        public float RunSafe(CudaDeviceVariable<float2> dataInOut, int width, int height, int clearAxis, float lp, float hp, float lps, float hps)
        {
            int fftWidth = width / 2 + 1;
            this.SetComputeSize((uint)fftWidth, (uint)height);
            return this.Run(dataInOut.DevicePointer, fftWidth * (int)float2.SizeOf, width, height, lp, hp, lps, hps, clearAxis);
        }
    }

    public class fftshiftKernel : CudaKernel
    {
        //fftshift(float2 * fft, int width, int height)

        const string kernelName = "fftshift";
        dim3 blockSize = new dim3(32, 32, 1);
        public fftshiftKernel(CudaContext ctx, CUmodule module)
            : base(kernelName, module, ctx)
        {
            this.BlockDimensions = blockSize;
        }

        public float RunSafe(CudaDeviceVariable<float2> dataInOut, int width, int height)
        {
            int fftWidth = width / 2 + 1;
            this.SetComputeSize((uint)fftWidth, (uint)height);
            return this.Run(dataInOut.DevicePointer, fftWidth, height);
        }
    }

    public class squaredSumKernel : CudaKernel
    {
        const string kernelName = "squaredSum";
        dim3 blockSize = new dim3(128, 1, 1);
        public squaredSumKernel(CudaContext ctx, CUmodule module)
            : base(kernelName, module, ctx)
        {
            this.BlockDimensions = blockSize;
        }

        public float RunSafe(CudaDeviceVariable<float> inTiles, CudaDeviceVariable<float> outTiles, int maxShift, int tileSize, int tileCount)
        {
            this.SetComputeSize((uint)(tileCount));
            return this.Run(inTiles.DevicePointer, outTiles.DevicePointer, maxShift, tileSize, tileCount);
        }
    }

    public class boxFilterWithBorderXKernel : CudaKernel
    {
        const string kernelName = "boxFilterWithBorderX";
        public boxFilterWithBorderXKernel(CudaContext ctx, CUmodule module)
            : base(kernelName, module, ctx)
        {
        }

        public float RunSafe(CudaDeviceVariable<float> inTiles, CudaDeviceVariable<float> outTiles, int maxShift, int tileSize, int tileCount)
        {
            this.BlockDimensions = new dim3((uint)tileSize + 2 * (uint)maxShift, 1, 4);
            this.SetComputeSize((uint)(tileSize + 2 * maxShift), (uint)(tileSize + 2 * maxShift), (uint)(tileCount));
            this.DynamicSharedMemory = this.BlockDimensions.z * (uint)(2 * maxShift + tileSize) * sizeof(float);

            return this.Run(inTiles.DevicePointer, outTiles.DevicePointer, maxShift, tileSize, tileCount);
        }
    }

    public class boxFilterWithBorderYKernel : CudaKernel
    {
        const string kernelName = "boxFilterWithBorderY";
        public boxFilterWithBorderYKernel(CudaContext ctx, CUmodule module)
            : base(kernelName, module, ctx)
        {
        }

        public float RunSafe(CudaDeviceVariable<float> inTiles, CudaDeviceVariable<float> outTiles, int maxShift, int tileSize, int tileCount)
        {
            this.BlockDimensions = new dim3(1, (uint)tileSize + 2 * (uint)maxShift, 4);
            this.SetComputeSize((uint)(tileSize + 2 * maxShift), (uint)(tileSize + 2 * maxShift), (uint)(tileCount));
            this.DynamicSharedMemory = this.BlockDimensions.z * (uint)(2 * maxShift + tileSize) * sizeof(float);

            return this.Run(inTiles.DevicePointer, outTiles.DevicePointer, maxShift, tileSize, tileCount);
        }
    }

    public class normalizedCCKernel : CudaKernel
    {
        const string kernelName = "normalizedCC";
        public normalizedCCKernel(CudaContext ctx, CUmodule module)
            : base(kernelName, module, ctx)
        {
        }

        public float RunSafe(CudaDeviceVariable<float> ccImage, CudaDeviceVariable<float> squaredTemplate, CudaDeviceVariable<float> boxFilteredImage, CudaDeviceVariable<float> shiftImage,
            int maxShift, int tileSize, int tileCount)
        {

            if (2 * (uint)maxShift + 1 > 8)
            {
                this.BlockDimensions = new dim3(8, 8, 4);
            }
            else
            {
                this.BlockDimensions = new dim3(2 * (uint)maxShift + 1, 2 * (uint)maxShift + 1, 4);
            }
            this.SetComputeSize(2 * (uint)maxShift + 1, 2 * (uint)maxShift + 1, (uint)(tileCount));

            return this.Run(ccImage.DevicePointer, squaredTemplate.DevicePointer, boxFilteredImage.DevicePointer, shiftImage.DevicePointer, maxShift, tileSize, tileCount);
        }
    }

    public class findMinimumKernel : CudaKernel
    {
        const string kernelName = "findMinimum";
        public findMinimumKernel(CudaContext ctx, CUmodule module)
            : base(kernelName, module, ctx)
        {
        }

        public float RunSafe(CudaDeviceVariable<float> shiftImage, NPPImage_32fC2 coordinates, int maxShift, int tileCountX, int tileCountY, float threshold)
        {
            this.BlockDimensions = new dim3(128, 1, 1);
            this.SetComputeSize((uint)(tileCountX * tileCountY), 1, 1);

            return this.Run(shiftImage.DevicePointer, coordinates.DevicePointerRoi, coordinates.Pitch, maxShift, tileCountX * tileCountY, tileCountX, threshold);
        }
    }

    public class upsampleShiftsKernel : CudaKernel
    {
        const string kernelName = "UpSampleShifts";
        public upsampleShiftsKernel(CudaContext ctx, CUmodule module)
            : base(kernelName, module, ctx)
        {
        }

        public float RunSafe(NPPImage_32fC2 inShift, NPPImage_32fC2 outShift, int oldLevel, int newLevel, int oldCountX, int oldCountY, int newCountX, int newCountY, int oldTileSize, int newTileSize)
        {
            this.BlockDimensions = new dim3(16, 16, 1);
            this.SetComputeSize((uint)(newCountX), (uint)(newCountY), 1);

            return this.Run(inShift.DevicePointerRoi, outShift.DevicePointerRoi, inShift.Pitch, outShift.Pitch, oldLevel, newLevel, oldCountX, oldCountY, newCountX, newCountY, oldTileSize, newTileSize);
        }
    }

    /*GammasRGB(
	float3 * __restrict__ inOutImg,
	int imgWidth,
	int imgHeight,
	int imgPitch
     */

    public class GammasRGBKernel : CudaKernel
    {
        const string kernelName = "GammasRGB";
        public GammasRGBKernel(CudaContext ctx, CUmodule module)
            : base(kernelName, module, ctx)
        {
        }

        public float RunSafe(NPPImage_32fC3 img)
        {
            this.BlockDimensions = new dim3(16, 16, 1);
            this.SetComputeSize((uint)(img.WidthRoi), (uint)(img.HeightRoi), 1);

            return this.Run(img.DevicePointerRoi, img.WidthRoi, img.HeightRoi, img.Pitch);
        }
    }
    public class ApplyWeightingKernel : CudaKernel
    {
        const string kernelName = "ApplyWeighting";
        public ApplyWeightingKernel(CudaContext ctx, CUmodule module)
            : base(kernelName, module, ctx)
        {
        }

        /*
         
	float3 * __restrict__ inOutImg,
	const float3 * __restrict__ finalImg,
	const float3 * __restrict__ weight,
	int imgWidth,
	int imgHeight,
	int imgPitch,
	float threshold
             */

        public float RunSafe(NPPImage_32fC3 img, NPPImage_32fC3 finalImg, NPPImage_32fC3 weights, float threashold)
        {
            this.BlockDimensions = new dim3(16, 16, 1);
            this.SetComputeSize((uint)(img.WidthRoi), (uint)(img.HeightRoi), 1);

            return this.Run(img.DevicePointerRoi, finalImg.DevicePointerRoi, weights.DevicePointerRoi, img.WidthRoi, img.HeightRoi, img.Pitch, threashold);
        }
    }

    public class computeStructureTensorKernel : CudaKernel
    {
        const string kernelName = "ComputeStructureTensor";
        dim3 blockSize = new dim3(24, 8, 1);
        public computeStructureTensorKernel(CudaContext ctx, CUmodule module)
            : base(kernelName, module, ctx)
        {
            this.BlockDimensions = blockSize;
        }

        public float RunSafe(NPPImage_32fC1 imgDx, NPPImage_32fC1 imgDy, NPPImage_32fC3 outImg)
        {
            this.SetComputeSize((uint)(outImg.WidthRoi), (uint)(outImg.HeightRoi), 1);
            return this.Run(imgDx.DevicePointerRoi, imgDy.DevicePointerRoi, outImg.DevicePointerRoi, outImg.WidthRoi, outImg.HeightRoi, imgDx.Pitch, outImg.Pitch);
        }
    }

    public class computeKernelParamKernel : CudaKernel
    {
        const string kernelName = "ComputeKernelParam";
        dim3 blockSize = new dim3(24, 8, 1);
        public computeKernelParamKernel(CudaContext ctx, CUmodule module)
            : base(kernelName, module, ctx)
        {
            this.BlockDimensions = blockSize;
        }

        public float RunSafe(NPPImage_32fC3 kernelImg, float Dth, float Dtr, float kDetail, float kDenoise, float kStretch, float kShrink)
        {
            this.SetComputeSize((uint)(kernelImg.WidthRoi), (uint)(kernelImg.HeightRoi), 1);
            return this.Run(kernelImg.DevicePointerRoi, kernelImg.WidthRoi, kernelImg.HeightRoi, kernelImg.Pitch, Dth, Dtr, kDetail, kDenoise, kStretch, kShrink);
        }
    }


}

