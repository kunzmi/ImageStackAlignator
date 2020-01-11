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

namespace PEFStudioDX
{
    public class OpticalFlow
    {
        NPPImage_32fC1 d_tmp;
        NPPImage_32fC1 d_Ix;
        NPPImage_32fC1 d_Iy;
        NPPImage_32fC1 d_Iz;

        NPPImage_32fC2 d_flow;
        CudaDeviceVariable<byte> buffer;
        CudaDeviceVariable<double> mean;
        CudaDeviceVariable<double> std;
        CudaDeviceVariable<float> d_filterX;
        CudaDeviceVariable<float> d_filterY;
        CudaDeviceVariable<float> d_filterT;

        CreateFlowFieldFromTiles createFlowFieldFromTiles;
        WarpingKernel warpingKernel;
        ComputeDerivativesKernel computeDerivativesKernel;
        LukasKanadeKernel lukasKanade;

        public void FreeDeviceMemory()
        {
            d_tmp.Dispose();
            d_Ix.Dispose();
            d_Iy.Dispose();
            d_Iz.Dispose();
            //d_imageHalf.Dispose();

            d_flow.Dispose();
            buffer.Dispose();
            mean.Dispose();
            std.Dispose();
            d_filterX.Dispose();
            d_filterY.Dispose();
            d_filterT.Dispose();
        }

        private void DumpFlowField(NPPImage_32fC2 flow, string filename)
        {
            float2[] f = new float2[flow.Width * flow.Height];

            flow.CopyToHost(f);

            FileStream fs = File.OpenWrite(filename);
            BinaryWriter bw = new BinaryWriter(fs);

            bw.Write(flow.Width);
            bw.Write(flow.Height);

            for (int i = 0; i < f.Length; i++)
            {
                bw.Write(f[i].x);
                bw.Write(f[i].y);
            }

            bw.Close();
            fs.Close();
            bw.Dispose();
            fs.Dispose();
        }

        private void DumpImage(NPPImage_32fC1 img, string filename)
        {
            float[] f = new float[img.Width * img.Height];

            img.CopyToHost(f);

            FileStream fs = File.OpenWrite(filename);
            BinaryWriter bw = new BinaryWriter(fs);

            bw.Write(img.Width);
            bw.Write(img.Height);

            for (int i = 0; i < f.Length; i++)
            {
                bw.Write(f[i]);
            }

            bw.Close();
            fs.Close();
            bw.Dispose();
            fs.Dispose();
        }

        public OpticalFlow(int width, int height, CudaContext ctx)
        {
            CUmodule mod = ctx.LoadModulePTX("opticalFlow.ptx");

            warpingKernel = new WarpingKernel(ctx, mod);
            createFlowFieldFromTiles = new CreateFlowFieldFromTiles(ctx, mod);
            computeDerivativesKernel = new ComputeDerivativesKernel(ctx, mod);
            lukasKanade = new LukasKanadeKernel(ctx, mod);

            d_tmp = new NPPImage_32fC1(width, height);
            d_Ix = new NPPImage_32fC1(width, height);
            d_Iy = new NPPImage_32fC1(width, height);
            d_Iz = new NPPImage_32fC1(width, height);
            d_flow = new NPPImage_32fC2(width, height);
            
            buffer = new CudaDeviceVariable<byte>(d_tmp.MeanStdDevGetBufferHostSize() * 3);
            mean = new CudaDeviceVariable<double>(1);
            std = new CudaDeviceVariable<double>(1);


            d_filterX = new float[] { -0.25f, 0.25f, -0.25f, 0.25f };
            d_filterY = new float[] { -0.25f, -0.25f, 0.25f, 0.25f };
            d_filterT = new float[] { 0.25f, 0.25f, 0.25f, 0.25f };
        }

        private void Swap(ref NPPImage_32fC2 a, ref NPPImage_32fC2 b)
        {
            NPPImage_32fC2 temp = a;
            a = b;
            b = temp;
        }

        public void LucasKanade(NPPImage_32fC1 sourceImg, NPPImage_32fC1 targetImg, NPPImage_32fC2 tiledFlow, int tileSize, int tileCountX, int tileCountY, int iterations, float2 baseShift, float baseRotation, float minDet, int windowSize)
        {
            createFlowFieldFromTiles.RunSafe(tiledFlow, d_flow, baseShift, baseRotation, tileSize, tileCountX, tileCountY);

            for (int iter = 0; iter < iterations; iter++)
            {
                warpingKernel.RunSafe(sourceImg, d_tmp, d_flow);
                NppiPoint p = new NppiPoint(0,0);
                d_Ix.Set(0);
                d_Iy.Set(0);
                d_Iz.Set(0);

                computeDerivativesKernel.RunSafe(d_tmp, targetImg, d_Ix, d_Iy, d_Iz);
                lukasKanade.RunSafe(d_flow, d_Ix, d_Iy, d_Iz, minDet, windowSize);
            }
            warpingKernel.RunSafe(sourceImg, d_tmp, d_flow);
            d_tmp.Copy(sourceImg);
        }

        public NPPImage_32fC2 LastFlow
        {
            get { return d_flow; }
        }

    }
}
