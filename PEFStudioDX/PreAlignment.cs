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
using System.IO;
using System.Text;
using System.Threading.Tasks;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using ManagedCuda.NPP;
using ManagedCuda.NPP.NPPsExtensions;
using ManagedCuda.CudaFFT;

namespace PEFStudioDX
{

    public class PreAlignment
    {
        NPPImage_32fC1 imgToTrackRotated;
        CudaDeviceVariable<float2> imgToTrackCplx;
        CudaDeviceVariable<float2> imgRefCplx;
        CudaDeviceVariable<byte> buffer;
        CudaDeviceVariable<int> x;
        CudaDeviceVariable<int> y;
        CudaDeviceVariable<float> val;

        CudaFFTPlanMany forward;
        CudaFFTPlanMany backward;
        CudaDeviceVariable<byte> bufferFFT;
        conjugateComplexMulKernel conjKernel;
        fourierFilterKernel fourierFilterKernel;
        fftshiftKernel fftshiftKernel;
        squaredSumKernel squaredSumKernel;
        boxFilterWithBorderXKernel boxFilterXKernel;
        boxFilterWithBorderYKernel boxFilterYKernel;
        normalizedCCKernel normalizedCCKernel;
        findMinimumKernel findMinimumKernel;

        int width;
        int height;
        SizeT FFTBufferSize;
        float highPass;
        bool memoryAllocated = false;

        public void AllocateDeviceMemory()
        {
            int fftWidth = width / 2 + 1;
            imgToTrackCplx = new CudaDeviceVariable<float2>(fftWidth * height);
            imgRefCplx = new CudaDeviceVariable<float2>(fftWidth * height);
            x = new CudaDeviceVariable<int>(1);
            y = new CudaDeviceVariable<int>(1);
            val = new CudaDeviceVariable<float>(1);

            bufferFFT = new CudaDeviceVariable<byte>(FFTBufferSize);

            int maxBufferSize = imgToTrackRotated.MaxIndexGetBufferHostSize();
            maxBufferSize = Math.Max(maxBufferSize, imgToTrackRotated.MinMaxGetBufferHostSize());
            buffer = new CudaDeviceVariable<byte>(maxBufferSize);

            forward.SetWorkArea(bufferFFT.DevicePointer);
            backward.SetWorkArea(bufferFFT.DevicePointer);
            memoryAllocated = true;
        }

        public PreAlignment(NPPImage_32fC1 img, CudaContext ctx)
        {
            width = img.WidthRoi;
            height = img.HeightRoi;
            imgToTrackRotated = new NPPImage_32fC1(width, height);

            CUmodule mod = ctx.LoadModule("kernel.ptx");

            int fftWidth = width / 2 + 1;
            conjKernel = new conjugateComplexMulKernel(ctx, mod);
            fourierFilterKernel = new fourierFilterKernel(ctx, mod);
            fftshiftKernel = new fftshiftKernel(ctx, mod);

            squaredSumKernel = new squaredSumKernel(ctx, mod);
            boxFilterXKernel = new boxFilterWithBorderXKernel(ctx, mod);
            boxFilterYKernel = new boxFilterWithBorderYKernel(ctx, mod);
            normalizedCCKernel = new normalizedCCKernel(ctx, mod);
            findMinimumKernel = new findMinimumKernel(ctx, mod);



            int n = 2;
            int[] dims = new int[] { height, width };
            int batches = 1;
            int[] inembed = new int[] { 1, imgToTrackRotated.Pitch / 4 };
            int[] onembed = new int[] { 1, fftWidth };
            int idist = height * imgToTrackRotated.Pitch / 4;
            int odist = height * fftWidth;
            int istride = 1;
            int ostride = 1;

            cufftHandle handleForward = cufftHandle.Create();
            cufftHandle handleBackward = cufftHandle.Create();

            SizeT sizeForward = new SizeT();
            SizeT sizeBackward = new SizeT();
            forward = new CudaFFTPlanMany(handleForward, n, dims, batches, cufftType.R2C, inembed, istride, idist, onembed, ostride, odist, ref sizeForward, false);
            backward = new CudaFFTPlanMany(handleBackward, n, dims, batches, cufftType.C2R, onembed, ostride, odist, inembed, istride, idist, ref sizeBackward, false);

            FFTBufferSize = sizeForward > sizeBackward ? sizeForward : sizeBackward;

        }

        public void FreeDeviceMemory()
        {
            if (bufferFFT == null)
                return;

            bufferFFT.Dispose();

            val.Dispose();
            y.Dispose();
            x.Dispose();
            buffer.Dispose();
            imgRefCplx.Dispose();
            imgToTrackCplx.Dispose();
            imgToTrackRotated.Dispose();
            memoryAllocated = false;
        }

        public void FreeResources()
        {
            if (bufferFFT == null)
                return;
            bufferFFT.Dispose();
            backward.Dispose();
            forward.Dispose();

            val.Dispose();
            y.Dispose();
            x.Dispose();
            buffer.Dispose();
            imgRefCplx.Dispose();
            imgToTrackCplx.Dispose();
            imgToTrackRotated.Dispose();
            memoryAllocated = false;
        }

        public void SetReferenceImage(NPPImage_32fC1 reference)
        {
            NppiRect saveRoi = new NppiRect(reference.PointRoi, reference.SizeRoi);
            NppiRect roi = new NppiRect();
            roi.x = 0;// (reference.WidthRoi - imgToTrackRotated.WidthRoi) / 2;
            roi.y = 0;// (reference.HeightRoi - imgToTrackRotated.HeightRoi) / 2;
            roi.width = imgToTrackRotated.WidthRoi;
            roi.height = imgToTrackRotated.HeightRoi;

            reference.SetRoi(roi);
            reference.Copy(imgToTrackRotated);
            
            forward.Exec(imgToTrackRotated.DevicePointerRoi, imgRefCplx.DevicePointer);
            reference.SetRoi(saveRoi);
        }

        public double4 ScanAngles(NPPImage_32fC1 img, double incr, double range, double zero)
        {
            NppiRect saveRoi = new NppiRect(img.PointRoi, img.SizeRoi);
            NppiRect roi = new NppiRect();
            roi.x = 0;
            roi.y = 0;
            roi.width = imgToTrackRotated.WidthRoi;
            roi.height = imgToTrackRotated.HeightRoi;
            img.SetRoi(roi);

            double maxVal = -double.MaxValue;
            double maxAng = 0;
            double maxX = 0;
            double maxY = 0;
                       
            //first perform a coarse search
            for (double ang = zero - range; ang <= zero + range; ang += 5 * incr)
            {
                Matrix3x3 mat = Matrix3x3.RotAroundCenter(ang, imgToTrackRotated.Width, imgToTrackRotated.Height);
                imgToTrackRotated.Set(0);
                img.WarpAffine(imgToTrackRotated, mat.ToAffine(), InterpolationMode.Cubic);

                forward.Exec(imgToTrackRotated.DevicePointerRoi, imgToTrackCplx.DevicePointer);

                conjKernel.RunSafe(imgRefCplx, imgToTrackCplx);
                backward.Exec(imgToTrackCplx.DevicePointer, imgToTrackRotated.DevicePointerRoi);
                imgToTrackRotated.Div(imgToTrackRotated.WidthRoi * imgToTrackRotated.HeightRoi);
                
                imgToTrackRotated.MaxIndex(val, x, y);
                float v = val;
                int hx = x;
                int hy = y;
                //Console.WriteLine("Found Max at " + ang.ToString("0.000") + " deg (" + hx + ", " + hy + ") = " + v);
                if (v > maxVal)
                {
                    maxVal = v;
                    maxAng = ang;
                    maxX = x;
                    maxY = y;
                    //Console.WriteLine("Max set!");
                }
            }

            zero = maxAng;
            range = 10 * incr;
            //now perform a fine search but only around the previously found peak
            for (double ang = zero - range; ang <= zero + range; ang += incr)
            {
                Matrix3x3 mat = Matrix3x3.RotAroundCenter(ang, imgToTrackRotated.Width, imgToTrackRotated.Height);
                imgToTrackRotated.Set(0);
                img.WarpAffine(imgToTrackRotated, mat.ToAffine(), InterpolationMode.Cubic);

                int fftWidth = width / 2 + 1;
                forward.Exec(imgToTrackRotated.DevicePointerRoi, imgToTrackCplx.DevicePointer);
                conjKernel.RunSafe(imgRefCplx, imgToTrackCplx);
                backward.Exec(imgToTrackCplx.DevicePointer, imgToTrackRotated.DevicePointerRoi);
                imgToTrackRotated.Div(imgToTrackRotated.WidthRoi * imgToTrackRotated.HeightRoi);

                imgToTrackRotated.MaxIndex(val, x, y);

                float v = val;
                int hx = x;
                int hy = y;
                if (v > maxVal)
                {
                    maxVal = v;
                    maxAng = ang;
                    maxX = x;
                    maxY = y;
                    //Console.WriteLine("Found Max at " + ang.ToString("0.000") + " deg (" + hx + ", " + hy + ") = " + v);
                    //Console.WriteLine("Max set!");
                }
            }

            if (maxX > imgToTrackRotated.WidthRoi / 2)
            {
                maxX -= imgToTrackRotated.WidthRoi;
            }
            if (maxY > imgToTrackRotated.HeightRoi / 2)
            {
                maxY -= imgToTrackRotated.HeightRoi;
            }

            img.SetRoi(saveRoi);
            return new double4(-maxX, -maxY, maxAng, maxVal);
        }

        public bool DimensionsFit(int aWidth, int aHeight)
        {
            return memoryAllocated && aWidth == width && aHeight == height;
        }

        public void FourierFilter(NPPImage_32fC1 img, int clearAxis, float aHighPass, float aHighPassSigma)
        {
            forward.Exec(img.DevicePointerRoi, imgToTrackCplx.DevicePointer);
            fourierFilterKernel.RunSafe(imgToTrackCplx, width, height, clearAxis, 1, aHighPass, 1, aHighPassSigma);
            backward.Exec(imgToTrackCplx.DevicePointer, img.DevicePointerRoi);
            img.Div(img.WidthRoi * img.HeightRoi);
            CudaDeviceVariable<float> minVal = new CudaDeviceVariable<float>(x.DevicePointer, 4);
            CudaDeviceVariable<float> maxVal = new CudaDeviceVariable<float>(y.DevicePointer, 4);

            img.MinMax(minVal, maxVal, buffer);
            float min = minVal;
            float max = maxVal;
            img.ThresholdLTGT(0, 0, 1, 1);
        }

        public (int, int) TestCC(NPPImage_32fC1 img, int shiftX, int shiftY)
        {
            Matrix3x3 mat = Matrix3x3.ShiftAffine(shiftX, shiftY);

            img.WarpAffine(imgToTrackRotated, mat.ToAffine(), InterpolationMode.NearestNeighbor);

            forward.Exec(imgToTrackRotated.DevicePointerRoi, imgToTrackCplx.DevicePointer);
            conjKernel.RunSafe(imgRefCplx, imgToTrackCplx);
            fftshiftKernel.RunSafe(imgToTrackCplx, width, height);
            backward.Exec(imgToTrackCplx.DevicePointer, img.DevicePointerRoi);
            img.Div(img.WidthRoi * img.HeightRoi);

            img.MaxIndex(val, x, y);

            float v = val;
            int hx = x;
            int hy = y;

            hx -= imgToTrackRotated.WidthRoi / 2;
            hy -= imgToTrackRotated.HeightRoi / 2;
            

            CudaDeviceVariable<float> minVal = new CudaDeviceVariable<float>(x.DevicePointer, 4);
            CudaDeviceVariable<float> maxVal = new CudaDeviceVariable<float>(y.DevicePointer, 4);

            img.MinMax(minVal, maxVal, buffer);
            float min = minVal;
            float max = maxVal;
            img.Sub(min);
            img.Div(max - min);


            img.ThresholdLTGT(0, 0, 1, 1);

            return (hx, hy);
        }
    }
}
