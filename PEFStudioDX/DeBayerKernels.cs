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
    public class DeBayerGreenKernel : CudaKernel
    {
        public DeBayerGreenKernel(CUmodule module, CudaContext ctx)
            : base("deBayerGreenKernel", module, ctx)
        {
            BlockDimensions = new dim3(32, 16, 1);
            GridDimensions = new dim3(1, 1, 1);
        }

        public float RunSafe(NPPImage_32fC1 imgIn, NPPImage_32fC3 imgOut, float3 blackPoint, float3 scale)
        {
            //const int width, const int height, const float* __restrict__ imgIn, int strideIn, float3 *outImage, int strideOut, float3 blackPoint, float3 scale
            SetComputeSize((uint)imgIn.Width, (uint)imgIn.Height, 1);
            return base.Run(imgIn.Width, imgIn.Height, imgIn.DevicePointer, imgIn.Pitch, imgOut.DevicePointer, imgOut.Pitch, blackPoint, scale);
        }

        public float RunSafe(CudaDeviceVariable<float> imgIn, CudaDeviceVariable<float3> imgOut, int patchSize, float3 blackPoint, float3 scale)
        {
            SetComputeSize((uint)patchSize, (uint)patchSize, 1);
            return base.Run(patchSize, patchSize, imgIn.DevicePointer, patchSize * 4, imgOut.DevicePointer, patchSize * 12, blackPoint, scale);
        }

        public PentaxPefFile.RawFile.BayerColor[] BayerPattern
        {
            set
            {
                int[] temp = new int[value.Length];
                for (int i = 0; i < value.Length; i++)
                {
                    temp[i] = (int)value[i];
                }
                base.SetConstantVariable<int>("c_cfaPattern", temp);
            }
        }
    }

    public class DeBayerRedBlueKernel : CudaKernel
    {
        public DeBayerRedBlueKernel(CUmodule module, CudaContext ctx)
            : base("deBayerRedBlueKernel", module, ctx)
        {
            BlockDimensions = new dim3(32, 16, 1);
            GridDimensions = new dim3(1, 1, 1);
        }

        public float RunSafe(NPPImage_32fC1 imgIn, NPPImage_32fC3 imgOut, float3 blackPoint, float3 scale)
        {
            //const int width, const int height, const float* __restrict__ imgIn, int strideIn, float3 *outImage, int strideOut, float3 blackPoint, float3 scale
            SetComputeSize((uint)imgIn.Width, (uint)imgIn.Height, 1);
            return base.Run(imgIn.Width, imgIn.Height, imgIn.DevicePointer, imgIn.Pitch, imgOut.DevicePointer, imgOut.Pitch, blackPoint, scale);
        }

        public float RunSafe(CudaDeviceVariable<float> imgIn, CudaDeviceVariable<float3> imgOut, int patchSize, float3 blackPoint, float3 scale)
        {
            SetComputeSize((uint)patchSize, (uint)patchSize, 1);
            return base.Run(patchSize, patchSize, imgIn.DevicePointer, patchSize * 4, imgOut.DevicePointer, patchSize * 12, blackPoint, scale);
        }

        public PentaxPefFile.RawFile.BayerColor[] BayerPattern
        {
            set
            {
                int[] temp = new int[value.Length];
                for (int i = 0; i < value.Length; i++)
                {
                    temp[i] = (int)value[i];
                }
                base.SetConstantVariable<int>("c_cfaPattern", temp);
            }
        }
    }

    public class DeBayersSubSampleKernel : CudaKernel
    {
        private const uint BlockSizeX = 16;
        private const uint BlockSizeY = 16;

        public DeBayersSubSampleKernel(CudaContext ctx, CUmodule module)
            : base("deBayersSubSample3", module, ctx, BlockSizeX, BlockSizeY)
        {
            //deBayersSubSample(unsigned short* dataIn, float3* imgOut, int bitDepth, int dimX, int dimY, int strideOut)
        }

        public float RunSafe(CudaDeviceVariable<ushort> imgIn, NPPImage_32fC3 imgOut, float maxVal)
        {
            SetComputeSize((uint)imgOut.WidthRoi, (uint)imgOut.HeightRoi);
            return base.Run(imgIn.DevicePointer, imgOut.DevicePointer, maxVal, imgOut.WidthRoi, imgOut.HeightRoi, imgOut.Pitch);
        }

        public PentaxPefFile.RawFile.BayerColor[] BayerPattern
        {
            set
            {
                int[] temp = new int[value.Length];
                for (int i = 0; i < value.Length; i++)
                {
                    temp[i] = (int)value[i];
                }
                base.SetConstantVariable<int>("c_cfaPattern", temp);
            }
        }
    }

    public class AccumulateImagesKernel : CudaKernel
    {
        private const uint BlockSizeX = 16;
        private const uint BlockSizeY = 16;

        public AccumulateImagesKernel(CudaContext ctx, CUmodule module)
            : base("accumulateImages", module, ctx, BlockSizeX, BlockSizeY)
        {
            /*
             * accumulateImages(
	unsigned short* __restrict__ dataIn,
	float3 * __restrict__ imgOut,
	float3 * __restrict__ totalWeights,
	const float3 * __restrict__ certaintyMask,
	const float3* __restrict__ kernelParam,
	const float2* __restrict__ shifts,
	float maxVal, int dimX, int dimY, int strideOut)
             */
        }

        public float RunSafe(CudaDeviceVariable<ushort> dataIn, NPPImage_32fC3 imgOut, NPPImage_32fC3 totalWeights, NPPImage_32fC4 certaintyMask, NPPImage_32fC3 kernelParam, NPPImage_32fC2 shifts, float maxVal)
        {
            SetComputeSize((uint)imgOut.WidthRoi, (uint)imgOut.HeightRoi);

            return base.Run(dataIn.DevicePointer, imgOut.DevicePointerRoi, totalWeights.DevicePointerRoi,
                certaintyMask.DevicePointerRoi, kernelParam.DevicePointerRoi, shifts.DevicePointerRoi,
                maxVal, imgOut.WidthRoi, imgOut.HeightRoi, imgOut.Pitch, certaintyMask.Pitch, shifts.Pitch);
        }

        public PentaxPefFile.RawFile.BayerColor[] BayerPattern
        {
            set
            {
                int[] temp = new int[value.Length];
                for (int i = 0; i < value.Length; i++)
                {
                    temp[i] = (int)value[i];
                }
                base.SetConstantVariable<int>("c_cfaPattern", temp);
            }
        }
    }

    public class AccumulateImagesSuperResKernel : CudaKernel
    {
        private const uint BlockSizeX = 16;
        private const uint BlockSizeY = 16;

        public AccumulateImagesSuperResKernel(CudaContext ctx, CUmodule module)
            : base("accumulateImagesSuperRes", module, ctx, BlockSizeX, BlockSizeY)
        {
            /*
             * accumulateImages(
	unsigned short* __restrict__ dataIn,
	float3 * __restrict__ imgOut,
	float3 * __restrict__ totalWeights,
	const float3 * __restrict__ certaintyMask,
	const float3* __restrict__ kernelParam,
	const float2* __restrict__ shifts,
	float maxVal, int dimX, int dimY, int strideOut)
             */
        }

        public float RunSafe(CudaDeviceVariable<ushort> dataIn, NPPImage_32fC3 imgOut, NPPImage_32fC3 totalWeights, NPPImage_32fC4 certaintyMask, NPPImage_32fC4 kernelParam, NPPImage_32fC2 shifts, float maxVal)
        {
            SetComputeSize((uint)imgOut.WidthRoi, (uint)imgOut.HeightRoi);

            CudaResourceDesc descKernel = new CudaResourceDesc(kernelParam);
            CudaTextureDescriptor texDescKernel = new CudaTextureDescriptor(CUAddressMode.Clamp, CUFilterMode.Linear, CUTexRefSetFlags.NormalizedCoordinates);
            CudaTexObject texKernel = new CudaTexObject(descKernel, texDescKernel);

            CudaResourceDesc descShift = new CudaResourceDesc(shifts);
            CudaTextureDescriptor texDescShift = new CudaTextureDescriptor(CUAddressMode.Mirror, CUFilterMode.Linear, CUTexRefSetFlags.NormalizedCoordinates);
            CudaTexObject texShift = new CudaTexObject(descShift, texDescShift);

            float t = base.Run(dataIn.DevicePointer, imgOut.DevicePointerRoi, totalWeights.DevicePointerRoi,
                certaintyMask.DevicePointerRoi, texKernel.TexObject, texShift.TexObject,
                maxVal, imgOut.WidthRoi, imgOut.HeightRoi, imgOut.Pitch, certaintyMask.Pitch, kernelParam.Pitch, shifts.Pitch);

            texShift.Dispose();
            texKernel.Dispose();
            return t;
        }

        public PentaxPefFile.RawFile.BayerColor[] BayerPattern
        {
            set
            {
                int[] temp = new int[value.Length];
                for (int i = 0; i < value.Length; i++)
                {
                    temp[i] = (int)value[i];
                }
                base.SetConstantVariable<int>("c_cfaPattern", temp);
            }
        }
    }
}
