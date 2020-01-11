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
    public class PatchTracker
    {
        int maxWidth;
        int maxHeight;

        int maxPixelsImage;
        int maxPixelsFFT;
        int maxPixelsShiftImage;
        int maxBlockCountX;
        int maxBlockCountY;
        SizeT FTTBufferSize;

        int currentWidth;
        int currentHeight;
        int currentTileSize;
        int currentMaxShift;


        CudaDeviceVariable<float2> imgToTrackCplx;
        CudaDeviceVariable<float2> imgRefCplx;
        CudaDeviceVariable<float> imgToTrackSortedTiles;
        CudaDeviceVariable<float> imgRefSortedTiles;
        CudaDeviceVariable<float> imgCrossCorrelation;
        CudaDeviceVariable<float> squaredSumsOfTiles;
        CudaDeviceVariable<float> shiftImages;
        NPPImage_32fC2 patchShift;
        CudaDeviceVariable<byte> FFTBuffer;

        CudaFFTPlanMany[] forward;
        CudaFFTPlanMany[] backward;

        conjugateComplexMulKernel conjKernel;
        convertToTilesOverlapKernel convertToTiles;
        convertToTilesOverlapBorderKernel convertToTilesBorder;
        squaredSumKernel squaredSumKernel;
        boxFilterWithBorderXKernel boxFilterXKernel;
        boxFilterWithBorderYKernel boxFilterYKernel;
        normalizedCCKernel normalizedCCKernel;
        findMinimumKernel findMinimumKernel;

        int debugCallCounter = 0;

        public int CurrentBlockSize
        {
            get { return currentTileSize + 2 * currentMaxShift; }
        }

        public int CurrentBlockCountX
        {
            get { return (currentWidth - currentMaxShift * 2) / currentTileSize; }
        }
        public int MaxBlockCountX
        {
            get { return maxBlockCountX; /*(maxWidth - maxShift * 2) / maxTileSize;*/ }
        }
        public int CurrentBlockCountY
        {
            get { return (currentHeight - currentMaxShift * 2) / currentTileSize; }
        }
        public int MaxBlockCountY
        {
            get { return maxBlockCountY; /*(maxHeight - maxShift * 2) / maxTileSize;*/ }
        }

        public void AllocateDeviceMemory()
        {
            //Allocate FFT Buffer
            FFTBuffer = new CudaDeviceVariable<byte>(FTTBufferSize);
            for (int i = 0; i < forward.Length; i++)
            {
                forward[i].SetWorkArea(FFTBuffer.DevicePointer);
                backward[i].SetWorkArea(FFTBuffer.DevicePointer);
            }


            int tilePixels = maxPixelsImage;
            imgToTrackSortedTiles = new CudaDeviceVariable<float>(tilePixels);
            imgRefSortedTiles = new CudaDeviceVariable<float>(tilePixels);
            imgCrossCorrelation = new CudaDeviceVariable<float>(tilePixels);
            int tilePixelsFFT = maxPixelsFFT;
            imgToTrackCplx = new CudaDeviceVariable<float2>(tilePixelsFFT);
            imgRefCplx = new CudaDeviceVariable<float2>(tilePixelsFFT);
            squaredSumsOfTiles = new CudaDeviceVariable<float>(MaxBlockCountX * MaxBlockCountY);
            shiftImages = new CudaDeviceVariable<float>(maxPixelsShiftImage);
            patchShift = new NPPImage_32fC2(MaxBlockCountX, MaxBlockCountY);
        }

        public PatchTracker(int aMaxWidth, int aMaxHeight, List<int> aTileSizes, List<int> aMaxShifts, List<int> aLevels, CudaContext ctx)
        {
            forward = new CudaFFTPlanMany[aLevels.Count];
            backward = new CudaFFTPlanMany[aLevels.Count];


            //Allocate FFT plans
            SizeT oldFFTSize = 0;
            for (int i = 0; i < aTileSizes.Count; i++)
            {
                SizeT memFFT = InitFFT(i, aMaxWidth / aLevels[i], aMaxHeight / aLevels[i], aTileSizes[i], aMaxShifts[i]);
                if (memFFT > oldFFTSize)
                {
                    oldFFTSize = memFFT;
                }
            }
            FTTBufferSize = oldFFTSize;

            //find maximum for allocations:
            for (int i = 0; i < aTileSizes.Count; i++)
            {
                currentWidth = aMaxWidth / aLevels[i];
                currentHeight = aMaxHeight / aLevels[i];
                currentTileSize = aTileSizes[i];
                currentMaxShift = aMaxShifts[i];

                int currentMaxPixelsShiftImage = (2 * currentMaxShift + 1) * (2 * currentMaxShift + 1) * CurrentBlockCountX * CurrentBlockCountY;
                maxPixelsShiftImage = Math.Max(currentMaxPixelsShiftImage, maxPixelsShiftImage);

                int tilePixels = CurrentBlockSize * CurrentBlockSize * CurrentBlockCountX * CurrentBlockCountY;
                maxPixelsImage = Math.Max(tilePixels, maxPixelsImage);
                int fftWidth = CurrentBlockSize / 2 + 1;
                int fftPixels = fftWidth * CurrentBlockSize * CurrentBlockCountX * CurrentBlockCountY;
                maxPixelsFFT = Math.Max(fftPixels, maxPixelsFFT);

                maxWidth = Math.Max(aMaxWidth / aLevels[i], maxWidth);
                maxHeight = Math.Max(aMaxHeight / aLevels[i], maxHeight);

                maxBlockCountX = Math.Max(maxBlockCountX, CurrentBlockCountX);
                maxBlockCountY = Math.Max(maxBlockCountY, CurrentBlockCountY);
            }

            CUmodule mod = ctx.LoadModule("kernel.ptx");

            conjKernel = new conjugateComplexMulKernel(ctx, mod);
            convertToTiles = new convertToTilesOverlapKernel(ctx, mod);
            convertToTilesBorder = new convertToTilesOverlapBorderKernel(ctx, mod);
            squaredSumKernel = new squaredSumKernel(ctx, mod);
            boxFilterXKernel = new boxFilterWithBorderXKernel(ctx, mod);
            boxFilterYKernel = new boxFilterWithBorderYKernel(ctx, mod);
            normalizedCCKernel = new normalizedCCKernel(ctx, mod);
            findMinimumKernel = new findMinimumKernel(ctx, mod);

        }

        public void FreeResources()
        {
            foreach (var item in backward)
            {
                item.Dispose();
            }
            foreach (var item in forward)
            {
                item.Dispose();
            }
            FFTBuffer.Dispose();
            patchShift.Dispose();
            shiftImages.Dispose();
            squaredSumsOfTiles.Dispose();
            imgCrossCorrelation.Dispose();
            imgRefSortedTiles.Dispose();
            imgToTrackSortedTiles.Dispose();
            imgRefCplx.Dispose();
            imgToTrackCplx.Dispose();
        }

        public void FreeDeviceMemory()
        {
            if (FFTBuffer == null)
                return;

            FFTBuffer.Dispose();
            patchShift.Dispose();
            shiftImages.Dispose();
            squaredSumsOfTiles.Dispose();
            imgCrossCorrelation.Dispose();
            imgRefSortedTiles.Dispose();
            imgToTrackSortedTiles.Dispose();
            imgRefCplx.Dispose();
            imgToTrackCplx.Dispose();
        }

        public void InitForSize(int aWidth, int aHeight, int aTileSize, int aMaxShift)
        {
            if (aWidth > maxWidth || aHeight > maxHeight)
            {
                throw new ArgumentOutOfRangeException();
            }

            currentWidth = aWidth;
            currentHeight = aHeight;
            currentTileSize = aTileSize;
            currentMaxShift = aMaxShift;

        }

        private SizeT InitFFT(int i, int width, int height, int tileSize, int maxShift)
        {
            int blockSize = tileSize + 2 * maxShift;
            int blockCountX = (width - maxShift * 2) / tileSize;
            int blockCountY = (height - maxShift * 2) / tileSize;

            int fftWidth = blockSize / 2 + 1;
            int n = 2;
            int[] dims = new int[] { blockSize, blockSize };
            int batches = blockCountX * blockCountY;
            int[] inembed = new int[] { 1, blockSize };
            int[] onembed = new int[] { 1, fftWidth };
            int idist = blockSize * blockSize;
            int odist = blockSize * fftWidth;
            int istride = 1;
            int ostride = 1;

            cufftHandle handleForward = cufftHandle.Create();
            cufftHandle handleBackward = cufftHandle.Create();

            SizeT sizeForward = new SizeT();
            SizeT sizeBackward = new SizeT();
            forward[i] = new CudaFFTPlanMany(handleForward, n, dims, batches, cufftType.R2C, inembed, istride, idist, onembed, ostride, odist, ref sizeForward, false);
            backward[i] = new CudaFFTPlanMany(handleBackward, n, dims, batches, cufftType.C2R, onembed, ostride, odist, inembed, istride, idist, ref sizeBackward, false);

            Console.WriteLine("Size FFT forward: " + sizeForward.ToString() + " backward: " + sizeBackward.ToString());

            return sizeForward > sizeBackward ? sizeForward : sizeBackward;
        }

        private void DumpFloat(float[] data, int width, int height, int tileCount, int tileIdx, string fileName)
        {
            FileStream fs = File.OpenWrite(fileName);
            BinaryWriter bw = new BinaryWriter(fs);

            int offset = tileIdx * width * height;

            bw.Write(width);
            bw.Write(height);
            bw.Write(1);
            for (int i = 0; i < width * height; i++)
            {
                bw.Write(data[i + offset]);
            }

            bw.Close();
            fs.Close();
            bw.Dispose();
            fs.Dispose();
        }
        public void DumpFlowField(NPPImage_32fC2 flow, string filename)
        {
            float2[] f = new float2[flow.WidthRoi * flow.HeightRoi];

            flow.CopyToHostRoi(f, new NppiRect(0, 0, flow.WidthRoi, flow.HeightRoi));

            FileStream fs = File.OpenWrite(filename);
            BinaryWriter bw = new BinaryWriter(fs);

            bw.Write(flow.WidthRoi);
            bw.Write(flow.HeightRoi);

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

        private void DumpToBMP(float[] data, int width, int height, int tileCount, int tileIdx, string fileName)
        {
            System.Drawing.Bitmap bmp = new System.Drawing.Bitmap(width, height);

            int offset = tileIdx * width * height;
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int c = (int)(data[offset + y * width + x] * 255);
                    bmp.SetPixel(x, y, System.Drawing.Color.FromArgb(c, c, c));
                }
            }

            bmp.Save(fileName);
            bmp.Dispose();
        }

        public void Track(NPPImage_32fC1 imgTrack, NPPImage_32fC1 imgRef, NPPImage_32fC2 preShift, int i, float2 baseShiftRef, float baseRotationRef, float2 baseShifttoTrack, float baseRotationtoTrack, float threshold)
        {
            if (imgTrack.WidthRoi != imgRef.WidthRoi || imgTrack.HeightRoi != imgRef.HeightRoi ||
                imgTrack.WidthRoi != currentWidth || imgTrack.HeightRoi != currentHeight)
            {
                throw new ArgumentOutOfRangeException();
            }

            int level = imgTrack.Width / imgTrack.WidthRoi;
            
            convertToTilesBorder.RunSafe(imgRef, imgRefSortedTiles, currentTileSize, currentMaxShift, CurrentBlockCountX, CurrentBlockCountY, baseShiftRef, baseRotationRef); //template
            forward[i].Exec(imgRefSortedTiles.DevicePointer, imgRefCplx.DevicePointer);

            convertToTiles.RunSafe(imgTrack, imgToTrackSortedTiles, preShift, currentTileSize, currentMaxShift, CurrentBlockCountX, CurrentBlockCountY, baseShifttoTrack, baseRotationtoTrack); //image in paper

            //DumpFloat(imgToTrackSortedTiles, currentTileSize + 2* currentMaxShift, currentTileSize + 2 * currentMaxShift, CurrentBlockCountX * CurrentBlockCountY, tileIdx, "tilesTrack_" + level + "_" + debugCallCounter + ".bin");
            //DumpFloat(imgRefSortedTiles, currentTileSize + 2 * currentMaxShift, currentTileSize + 2 * currentMaxShift, CurrentBlockCountX * CurrentBlockCountY, tileIdx, "tilesRef_" + level + "_" + debugCallCounter + ".bin");

            forward[i].Exec(imgToTrackSortedTiles.DevicePointer, imgToTrackCplx.DevicePointer);

            conjKernel.RunSafe(imgRefCplx, imgToTrackCplx);

            backward[i].Exec(imgToTrackCplx.DevicePointer, imgCrossCorrelation.DevicePointer);
            imgCrossCorrelation.DivC(CurrentBlockSize * CurrentBlockSize);

            squaredSumKernel.RunSafe(imgRefSortedTiles, squaredSumsOfTiles, currentMaxShift, currentTileSize, CurrentBlockCountX * CurrentBlockCountY); 
            //DumpFloat(squaredSumsOfTiles, 1, 1, CurrentBlockCountX * CurrentBlockCountY, tileIdx, "squaredSums_" + level + "_" + debugCallCounter + ".bin");

            boxFilterXKernel.RunSafe(imgToTrackSortedTiles, imgRefSortedTiles, currentMaxShift, currentTileSize, CurrentBlockCountX * CurrentBlockCountY);
            boxFilterYKernel.RunSafe(imgRefSortedTiles, imgToTrackSortedTiles, currentMaxShift, currentTileSize, CurrentBlockCountX * CurrentBlockCountY);
            //DumpFloat(imgToTrackSortedTiles, currentTileSize + 2 * currentMaxShift, currentTileSize + 2 * currentMaxShift, CurrentBlockCountX * CurrentBlockCountY, tileIdx, "boxFilter_" + level + "_" + debugCallCounter + ".bin");
            normalizedCCKernel.RunSafe(imgCrossCorrelation, squaredSumsOfTiles, imgToTrackSortedTiles, shiftImages, currentMaxShift, currentTileSize, CurrentBlockCountX * CurrentBlockCountY);
            
            //DumpFloat(shiftImages, (2 * currentMaxShift + 1), (2 * currentMaxShift + 1), CurrentBlockCountX * CurrentBlockCountY, tileIdx, "tilesShift_" + level + "_" + debugCallCounter + ".bin");

            patchShift.SetRoi(0, 0, CurrentBlockCountX, CurrentBlockCountY);
            findMinimumKernel.RunSafe(shiftImages, patchShift, currentMaxShift, CurrentBlockCountX, CurrentBlockCountY, threshold);
            
            NPPImage_32fC1 preShiftFloat = new NPPImage_32fC1(preShift.DevicePointer, 2 * CurrentBlockCountX, CurrentBlockCountY, preShift.Pitch);
            NPPImage_32fC1 patchShiftFloat = new NPPImage_32fC1(patchShift.DevicePointer, 2 * CurrentBlockCountX, CurrentBlockCountY, patchShift.Pitch);
            
            preShiftFloat.Add(patchShiftFloat);
            debugCallCounter++;
        }
    }
}
