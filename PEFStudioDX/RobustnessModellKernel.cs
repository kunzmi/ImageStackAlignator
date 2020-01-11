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
    public class RobustnessModellKernel : CudaKernel
    {
        const string kernelName = "ComputeRobustnessMask";
        public RobustnessModellKernel(CudaContext ctx, CUmodule module)
            : base(kernelName, module, ctx)
        {
        }
        /*
         
	const float3* __restrict__ rawImgRef,
	const float3* __restrict__ rawImgMoved,
	float3* __restrict__ robustnessMask,
	cudaTextureObject_t texUV,
	int imgWidth,
	int imgHeight,
	int imgPitch,
	float alpha,
	float beta)
             */
        public float RunSafe(NPPImage_32fC3 rawImgRef, NPPImage_32fC3 rawImgMoved, NPPImage_32fC4 robustnessMask, NPPImage_32fC2 shift, float alpha, float beta, float thresholdM)
        {
            this.BlockDimensions = new dim3(8, 8, 1);
            this.SetComputeSize((uint)(rawImgRef.WidthRoi), (uint)(rawImgRef.HeightRoi), 1);
            this.DynamicSharedMemory = BlockDimensions.x * BlockDimensions.y * float3.SizeOf * 3 * 3;

            CudaResourceDesc descShift = new CudaResourceDesc(shift);
            CudaTextureDescriptor texDescShift = new CudaTextureDescriptor(CUAddressMode.Mirror, CUFilterMode.Linear, CUTexRefSetFlags.NormalizedCoordinates);
            CudaTexObject texShift = new CudaTexObject(descShift, texDescShift);

            
            float t = this.Run(rawImgRef.DevicePointerRoi, rawImgMoved.DevicePointerRoi, robustnessMask.DevicePointerRoi, texShift.TexObject, rawImgRef.WidthRoi, 
                rawImgRef.HeightRoi, rawImgRef.Pitch, robustnessMask.Pitch, alpha, beta, thresholdM);

            texShift.Dispose();
            
            return t;
        }
    }
}
