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

    public class WarpingKernel : CudaKernel
    {
        const string kernelName = "WarpingKernel";
        public WarpingKernel(CudaContext ctx, CUmodule module)
            : base(kernelName, module, ctx)
        {
        }

        //int width, int height, int stride,
        //cudaTextureObject_t texUV, float* __restrict__ out, cudaTextureObject_t texToWarp
        public float RunSafe(NPPImage_32fC1 inImg, NPPImage_32fC1 outImg, NPPImage_32fC2 flow)
        {
            this.BlockDimensions = new dim3(32, 6, 1);
            this.SetComputeSize((uint)(outImg.WidthRoi), (uint)(outImg.HeightRoi), 1);

            CudaResourceDesc descImg = new CudaResourceDesc(inImg);
            CudaTextureDescriptor texDescImg = new CudaTextureDescriptor(CUAddressMode.Mirror, CUFilterMode.Linear, CUTexRefSetFlags.NormalizedCoordinates);
            CudaTexObject texImg = new CudaTexObject(descImg, texDescImg);

            CudaResourceDesc descFlow = new CudaResourceDesc(flow);
            CudaTextureDescriptor texDescFlow = new CudaTextureDescriptor(CUAddressMode.Clamp, CUFilterMode.Point, CUTexRefSetFlags.NormalizedCoordinates);
            CudaTexObject texFlow = new CudaTexObject(descFlow, texDescFlow);

            return this.Run(outImg.WidthRoi, outImg.HeightRoi, outImg.Pitch, texFlow.TexObject, outImg.DevicePointerRoi, texImg.TexObject);
        }
    }
    public class CreateFlowFieldFromTiles : CudaKernel
    {
        const string kernelName = "CreateFlowFieldFromTiles";
        public CreateFlowFieldFromTiles(CudaContext ctx, CUmodule module)
            : base(kernelName, module, ctx)
        {
        }

        //float2* __restrict__ outImg,
        //cudaTextureObject_t texObjShiftXY,
        //int imgWidth,
        //int imgHeight,
        //int imgPitch
        public float RunSafe(NPPImage_32fC2 inFlow, NPPImage_32fC2 outFlow, float2 baseShift, float baseRotation, int tileSize, int tileCountX, int tileCountY)
        {
            this.BlockDimensions = new dim3(32, 6, 1);
            this.SetComputeSize((uint)(outFlow.WidthRoi), (uint)(outFlow.HeightRoi), 1);

            CudaResourceDesc descImg = new CudaResourceDesc(inFlow);
            CudaTextureDescriptor texDescImg = new CudaTextureDescriptor(CUAddressMode.Clamp, CUFilterMode.Linear, CUTexRefSetFlags.NormalizedCoordinates);
            CudaTexObject texImg = new CudaTexObject(descImg, texDescImg);


            float t =  this.Run(outFlow.DevicePointerRoi, texImg.TexObject, tileSize, tileCountX, tileCountY, outFlow.WidthRoi, outFlow.HeightRoi, outFlow.Pitch, baseShift, baseRotation);
            texImg.Dispose();
            return t;
        }
    }
    public class ComputeDerivativesKernel : CudaKernel
    {
        const string kernelName = "ComputeDerivativesKernel";
        public ComputeDerivativesKernel(CudaContext ctx, CUmodule module)
            : base(kernelName, module, ctx)
        {
        }

        //int width, int height, int stride,
        //float* Ix, float* Iy, float* Iz,
        //cudaTextureObject_t texSource,
	    //cudaTextureObject_t texTarget
        public float RunSafe(NPPImage_32fC1 imgSource, NPPImage_32fC1 imgTarget, NPPImage_32fC1 Ix, NPPImage_32fC1 Iy, NPPImage_32fC1 Iz)
        {
            this.BlockDimensions = new dim3(32, 6, 1);
            this.SetComputeSize((uint)(imgSource.WidthRoi), (uint)(imgSource.HeightRoi), 1);

            CudaResourceDesc descImgSource = new CudaResourceDesc(imgSource);
            CudaTextureDescriptor texDescImgSource = new CudaTextureDescriptor(CUAddressMode.Mirror, CUFilterMode.Linear, CUTexRefSetFlags.NormalizedCoordinates);
            CudaTexObject texImgSource = new CudaTexObject(descImgSource, texDescImgSource);

            CudaResourceDesc descImgTarget = new CudaResourceDesc(imgTarget);
            CudaTextureDescriptor texDescImgTarget = new CudaTextureDescriptor(CUAddressMode.Mirror, CUFilterMode.Linear, CUTexRefSetFlags.NormalizedCoordinates);
            CudaTexObject texImgTarget = new CudaTexObject(descImgTarget, texDescImgTarget);


            float t =  this.Run(Ix.WidthRoi, Ix.HeightRoi, Ix.Pitch, Ix.DevicePointerRoi, Iy.DevicePointerRoi, Iz.DevicePointerRoi, texImgSource.TexObject, texImgTarget.TexObject);

            texImgTarget.Dispose();
            texImgSource.Dispose();

            return t;
        }
    }
    public class ComputeDerivatives2Kernel : CudaKernel
    {
        const string kernelName = "ComputeDerivatives2Kernel";
        public ComputeDerivatives2Kernel(CudaContext ctx, CUmodule module)
            : base(kernelName, module, ctx)
        {
        }

        public float RunSafe(NPPImage_32fC1 imgSource, NPPImage_32fC1 Ix, NPPImage_32fC1 Iy)
        {
            this.BlockDimensions = new dim3(32, 6, 1);
            this.SetComputeSize((uint)(imgSource.WidthRoi), (uint)(imgSource.HeightRoi), 1);

            CudaResourceDesc descImgSource = new CudaResourceDesc(imgSource);
            CudaTextureDescriptor texDescImgSource = new CudaTextureDescriptor(CUAddressMode.Mirror, CUFilterMode.Linear, CUTexRefSetFlags.NormalizedCoordinates);
            CudaTexObject texImgSource = new CudaTexObject(descImgSource, texDescImgSource);


            float t = this.Run(Ix.WidthRoi, Ix.HeightRoi, Ix.Pitch, Ix.DevicePointerRoi, Iy.DevicePointerRoi, texImgSource.TexObject);

            texImgSource.Dispose();

            return t;
        }
    }

    public class LukasKanadeKernel : CudaKernel
    {
        /*const float4 * __restrict__ temp,
	float2 * __restrict__ shifts,
	const float* __restrict__ imFx,
	const float* __restrict__ imFy,
	const float* __restrict__ imFt,
	int pitchShift,
	int pitchImg,
	int pitchTemp,
	int width,
	int height,
	int halfWindowSize
         */
        const string kernelName = "lucasKanadeOptim";
        public LukasKanadeKernel(CudaContext ctx, CUmodule module)
            : base(kernelName, module, ctx)
        {
        }
        public float RunSafe(NPPImage_32fC2 shifts, NPPImage_32fC1 imFx, NPPImage_32fC1 imFy, NPPImage_32fC1 imFt, float minDet, int windowSize)
        {
            this.BlockDimensions = new dim3(32, 16, 1);
            this.SetComputeSize((uint)(shifts.WidthRoi), (uint)(shifts.HeightRoi), 1);
            //this.DynamicSharedMemory = (uint)(5 * windowSize * windowSize) * BlockDimensions.x * BlockDimensions.y * sizeof(float);

            int windowSizeHalf = windowSize / 2;
            return this.Run(shifts.DevicePointerRoi, imFx.DevicePointerRoi, imFy.DevicePointerRoi, imFt.DevicePointerRoi, shifts.Pitch, imFx.Pitch, shifts.WidthRoi, shifts.HeightRoi, windowSizeHalf, minDet);

        }
    }
}
