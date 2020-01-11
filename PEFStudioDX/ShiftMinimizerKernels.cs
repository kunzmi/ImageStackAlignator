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
    public class concatenateShiftsKernel : CudaKernel
    {
        const string kernelName = "concatenateShifts";
        public concatenateShiftsKernel(CudaContext ctx, CUmodule module)
            : base(kernelName, module, ctx)
        {
        }
        
        /*shiftIn,
	float2* __restrict__ shiftOut,
	int shiftCount,
	int tileCount
         */

        public float RunSafe(CudaDeviceVariable<CUdeviceptr> measuredShifts, CudaDeviceVariable<int> shiftPitch, CudaDeviceVariable<float2> shiftOut, int shiftCount, int tileCountX, int tileCountY)
        {
            this.BlockDimensions = new dim3(8, 8, 4);
            this.SetComputeSize((uint)(shiftCount), (uint)(tileCountX), (uint)(tileCountY));

            return this.Run(measuredShifts.DevicePointer, shiftPitch.DevicePointer, shiftOut.DevicePointer, shiftCount, tileCountX, tileCountY);
        }
    }
    public class separateShiftsKernel : CudaKernel
    {
        const string kernelName = "separateShifts";
        public separateShiftsKernel(CudaContext ctx, CUmodule module)
            : base(kernelName, module, ctx)
        {
        }

        public float RunSafe(CudaDeviceVariable<float2> shiftIn, CudaDeviceVariable<CUdeviceptr> measuredShifts, CudaDeviceVariable<int> shiftPitch, int shiftCount, int tileCountX, int tileCountY)
        {
            this.BlockDimensions = new dim3(8, 8, 4);
            this.SetComputeSize((uint)(shiftCount), (uint)(tileCountX), (uint)(tileCountY));

            return this.Run(shiftIn.DevicePointer, measuredShifts.DevicePointer, shiftPitch.DevicePointer, shiftCount, tileCountX, tileCountY);
        }
    }
    public class getOptimalShiftsKernel : CudaKernel
    {
        /*
         getOptimalShifts(
	float2 * __restrict__ optimalShifts,
	const float2 * __restrict__ bestShifts,
	int imageCount,
	int tileCountX,
	int tileCountY,
	int optimalShiftsPitch,
	int referenceImage,
	int imageToTrack
             */

        const string kernelName = "getOptimalShifts";
        public getOptimalShiftsKernel(CudaContext ctx, CUmodule module)
            : base(kernelName, module, ctx)
        {
        }

        public float RunSafe(NPPImage_32fC2 optimalShifts, CudaDeviceVariable<float2> bestShifts, int imageCount, int referenceImage, int imageToTrack)
        {
            this.BlockDimensions = new dim3(32, 8, 1);
            this.SetComputeSize((uint)(optimalShifts.WidthRoi), (uint)(optimalShifts.HeightRoi), 1);

            return this.Run(optimalShifts.DevicePointer, bestShifts.DevicePointer, imageCount, optimalShifts.WidthRoi, optimalShifts.HeightRoi, optimalShifts.Pitch, referenceImage, imageToTrack);
        }
    }

    public class copyShiftMatrixKernel : CudaKernel
    {
        const string kernelName = "copyShiftMatrix";
        public copyShiftMatrixKernel(CudaContext ctx, CUmodule module)
            : base(kernelName, module, ctx)
        {
        }

        //copyShiftMatrix(float* __restrict__ matrices, int tileCount, int imageCount, int shiftCount)
        public float RunSafe(CudaDeviceVariable<float> matrices, int tileCount, int imageCount, int shiftCount)
        {
            this.BlockDimensions = new dim3(128, 1, 1);
            this.SetComputeSize((uint)tileCount, 1, 1);

            return this.Run(matrices.DevicePointer, tileCount, imageCount, shiftCount);
        }
    }

    public class setPointersKernel : CudaKernel
    {
        const string kernelName = "setPointers";
        public setPointersKernel(CudaContext ctx, CUmodule module)
            : base(kernelName, module, ctx)
        {
        }

        /*
         * setPointers(float** __restrict__ shiftMatrixArray, float** __restrict__ matrixSquareArray,
	float** __restrict__ matrixInvertedArray, float** __restrict__ solvedMatrixArray,
	float2**__restrict__ shiftOneToOneArray, float2** __restrict__ shiftMeasuredArray, float2** __restrict__ shiftOptimArray,
	float* shiftMatrices, float* matricesSquared, float* matricesInverted, float* solvedMatrices,
	float2* shiftsOneToOne, float2* shiftsMeasured, float2* shiftsOptim, int tileCount, int imageCount)
         */
        public float RunSafe(CudaDeviceVariable<CUdeviceptr> shiftMatrixArray, CudaDeviceVariable<CUdeviceptr> shiftMatrixSafeArray, CudaDeviceVariable<CUdeviceptr> matrixSquareArray,
    CudaDeviceVariable<CUdeviceptr> matrixInvertedArray, CudaDeviceVariable<CUdeviceptr> solvedMatrixArray,
    CudaDeviceVariable<CUdeviceptr> shiftOneToOneArray, CudaDeviceVariable<CUdeviceptr> shiftMeasuredArray, CudaDeviceVariable<CUdeviceptr> shiftOptimArray,
    CudaDeviceVariable<float> shiftMatrices, CudaDeviceVariable<float> shiftSafeMatrices, CudaDeviceVariable<float> matricesSquared, CudaDeviceVariable<float> matricesInverted, CudaDeviceVariable<float> solvedMatrices,
    CudaDeviceVariable<float2> shiftsOneToOne, CudaDeviceVariable<float2> shiftsMeasured, CudaDeviceVariable<float2> shiftsOptim, int tileCount, int imageCount, int shiftCount)
        {
            this.BlockDimensions = new dim3(128, 1, 1);
            this.SetComputeSize((uint)tileCount, 1, 1);

            return this.Run(shiftMatrixArray.DevicePointer, shiftMatrixSafeArray.DevicePointer, matrixSquareArray.DevicePointer,
                            matrixInvertedArray.DevicePointer, solvedMatrixArray.DevicePointer,
                            shiftOneToOneArray.DevicePointer, shiftMeasuredArray.DevicePointer, shiftOptimArray.DevicePointer,
                            shiftMatrices.DevicePointer, shiftSafeMatrices.DevicePointer, matricesSquared.DevicePointer, matricesInverted.DevicePointer, solvedMatrices.DevicePointer,
                            shiftsOneToOne.DevicePointer, shiftsMeasured.DevicePointer, shiftsOptim.DevicePointer, tileCount, imageCount, shiftCount);
        }
    }


    public class checkForOutliersKernel : CudaKernel
    {
        const string kernelName = "checkForOutliers";
        public checkForOutliersKernel(CudaContext ctx, CUmodule module)
            : base(kernelName, module, ctx)
        {
        }

        /*
         * checkForOutliers(
	float2* __restrict__ measuredShifts,
	const float2* __restrict__ optimShifts,
	const float* __restrict__ shiftsOneToOneT,
	float2* __restrict__ shiftsOneToOne,
	float* __restrict__ shiftMatrix,
	int* __restrict__ status,
	int* __restrict__ inversionInfo,
	int tileCount, int imageCount)
         */
        public float RunSafe(CudaDeviceVariable<float2> measuredShifts, CudaDeviceVariable<float2> optimShifts,
                            CudaDeviceVariable<float> shiftMatrix, CudaDeviceVariable<int> status,
                            CudaDeviceVariable<int> inversionInfo, int tileCount, int imageCount, int shiftCount)
        {
            this.BlockDimensions = new dim3(128, 1, 1);
            this.SetComputeSize((uint)tileCount, 1, 1);

            return this.Run(measuredShifts.DevicePointer, optimShifts.DevicePointer,
                            shiftMatrix.DevicePointer, status.DevicePointer,
                            inversionInfo.DevicePointer, tileCount, imageCount, shiftCount);
        }
    }


    public class transposeShiftsKernel : CudaKernel
    {
        const string kernelName = "transposeShifts";
        public transposeShiftsKernel(CudaContext ctx, CUmodule module)
            : base(kernelName, module, ctx)
        {
        }

        /*
         * transposeShifts(
	float2 * __restrict__ measuredShifts,
	const float* __restrict__ measuredShiftsT,
	int tileCount, int imageCount)
         */
        public float RunSafe(CudaDeviceVariable<float2> measuredShifts, CudaDeviceVariable<float2> measuredShiftsT,
                            CudaDeviceVariable<float2> shiftsOneToOneT, CudaDeviceVariable<float2> shiftsOneToOne,
                            int tileCount, int imageCount, int shiftCount)
        {
            uint m = (uint)((imageCount - 1) * ((float)imageCount / 2.0f));
            this.BlockDimensions = new dim3(32, 8, 1);
            this.SetComputeSize((uint)tileCount, m, 1);

            return this.Run(measuredShifts.DevicePointer, measuredShiftsT.DevicePointer,
                            shiftsOneToOneT.DevicePointer, shiftsOneToOne.DevicePointer,
                            tileCount, imageCount, shiftCount);
        }
    }

}
