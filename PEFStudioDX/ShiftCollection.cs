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
using System.ComponentModel;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using ManagedCuda.NPP;
using ManagedCuda.NPP.NPPsExtensions;
using ManagedCuda.CudaBlas;

namespace PEFStudioDX
{
    public class ShiftCollection
    {
        [TypeConverter(typeof(EnumDescriptionTypeConverter))]
        public enum TrackingStrategy
        {
            [Description("Full")]
            Full,
            [Description("Only on reference")]
            OnlyOnReference,
            [Description("On reference block")]
            OnReferenceBlock,
            [Description("Blocks")]
            Blocks
        }

        public struct ShiftPair
        {
            public int reference;
            public int toTrack;

            public ShiftPair(int aReference, int aToTrack)
            {
                reference = aReference;
                toTrack = aToTrack;
            }
        }

        int frameCount;
        int referenceIndex;
        TrackingStrategy strategy;
        List<ShiftPair> shiftPairs;
        int blockSize;


        CudaBlas blas;

        List<NPPImage_32fC2> shifts;
        CudaDeviceVariable<int> shiftPitches;
        concatenateShiftsKernel concatenateShifts;
        separateShiftsKernel separateShifts;
        getOptimalShiftsKernel getOptimalShifts;
        CudaDeviceVariable<CUdeviceptr> shifts_d;
        CudaDeviceVariable<float2> AllShifts_d;
        CudaDeviceVariable<float2> shiftsOneToOne_d;

        copyShiftMatrixKernel copyShiftMatrixKernel;
        setPointersKernel setPointers;
        checkForOutliersKernel checkForOutliers;
        transposeShiftsKernel transposeShifts;


        CudaDeviceVariable<int> status;
        CudaDeviceVariable<int> infoInverse;
        CudaDeviceVariable<CUdeviceptr> shiftMatrixArray;
        CudaDeviceVariable<CUdeviceptr> shiftMatrixSafeArray;
        CudaDeviceVariable<CUdeviceptr> matrixSquareArray;
        CudaDeviceVariable<CUdeviceptr> matrixInvertedArray;
        CudaDeviceVariable<CUdeviceptr> solvedMatrixArray;
        CudaDeviceVariable<CUdeviceptr> shiftOneToOneArray;
        CudaDeviceVariable<CUdeviceptr> shiftMeasuredArray;
        CudaDeviceVariable<CUdeviceptr> shiftOptimArray;
        CudaDeviceVariable<float> shiftMatrices;
        CudaDeviceVariable<float> shiftSafeMatrices;
        CudaDeviceVariable<float> matricesSquared;
        CudaDeviceVariable<float> matricesInverted;
        CudaDeviceVariable<float> solvedMatrices;
        CudaDeviceVariable<float2> shiftsOneToOne;
        CudaDeviceVariable<float2> shiftsMeasured;
        CudaDeviceVariable<float2> shiftsOptim;
        CudaDeviceVariable<float> one;
        CudaDeviceVariable<float> zero;
        CudaDeviceVariable<byte> buffer;
        CudaDeviceVariable<int> statusSum;
        CudaDeviceVariable<int> pivotArray;



        int[,] indices;

        private int GetShiftCount()
        {
            switch (strategy)
            {
                case TrackingStrategy.Full:
                    return (int)((frameCount - 1) * ((float)frameCount / 2.0f));
                case TrackingStrategy.OnlyOnReference:
                    return frameCount - 1;
                case TrackingStrategy.OnReferenceBlock:
                    { 
                        int shiftsInBlock = (int)((blockSize - 1) * ((float)blockSize / 2.0f));
                        int framesOutsideBlock = frameCount - blockSize;
                        return shiftsInBlock + blockSize * framesOutsideBlock;
                    }
                case TrackingStrategy.Blocks:
                    return (int)(-0.5 * blockSize * (blockSize - 2 * frameCount + 1));
                default:
                    return 0;
            }
        }

        private float[] CreateShiftMatrix()
        {
            float[] matrix = new float[(frameCount - 1) * GetShiftCount()];
            switch (strategy)
            {
                case TrackingStrategy.Full:
                    {
                        int m = GetShiftCount();

                        int counter = 0;
                        for (int shifts = 0; shifts < (frameCount - 1); shifts++)
                        {
                            int count = (frameCount - 1) - (shifts);

                            for (int line = 0; line < count; line++)
                            {
                                for (int l = line; l <= line + (frameCount - 1) - count; l++)
                                {
                                    matrix[l * m + counter] = 1;
                                }
                                counter++;
                            }
                        }
                    }
                    break;
                case TrackingStrategy.OnlyOnReference:
                    {
                        int m = GetShiftCount();
                        for (int img = 0; img < frameCount - 1; img++)
                        {
                            if (img < referenceIndex)
                            {
                                int dist = referenceIndex - img;

                                for (int i = referenceIndex - dist; i < referenceIndex; i++)
                                {
                                    matrix[i * m + img] = -1;
                                }
                            }
                            if (img >= referenceIndex)
                            {
                                int dist = img - referenceIndex;

                                for (int i = referenceIndex; i <= referenceIndex + dist; i++)
                                {
                                    matrix[i * m + img] = 1;
                                }
                            }
                        }
                    }
                    break;
                case TrackingStrategy.OnReferenceBlock:
                    {
                        int m = GetShiftCount();
                        int firstIndex; //in block
                        int lastIndex;
                        if (referenceIndex < frameCount / 2)
                        {
                            firstIndex = referenceIndex - blockSize / 2;
                            if (firstIndex < 0)
                            {
                                firstIndex = 0;
                            }
                            lastIndex = firstIndex + blockSize - 1;
                        }
                        else
                        {
                            lastIndex = referenceIndex + blockSize / 2;
                            if (lastIndex >= frameCount)
                            {
                                lastIndex = frameCount - 1;
                            }
                            firstIndex = lastIndex - blockSize + 1;
                        }

                        //full shifts in block
                        int counter = 0;
                        for (int shifts = 0; shifts < blockSize-1; shifts++)
                        {
                            int count = blockSize - 1 - (shifts);
                            
                            for (int line = 0; line < count; line++)
                            {
                                for (int l = line; l <= line + blockSize - 1 - count; l++)
                                {
                                    matrix[(l + firstIndex) * m + (counter)] = 1;
                                }
                                counter++;
                            }                            
                        }

                        //track all frames before block to all frames in block
                        for (int i = 0; i < firstIndex; i++)
                        {
                            for (int toBlock = firstIndex; toBlock <= lastIndex; toBlock++)
                            {
                                for (int j = i; j < toBlock; j++)
                                {
                                    matrix[j * m + counter] = -1;
                                }
                                counter = counter + 1;
                            }
                        }

                        //track all frames after block to all frames in block
                        for (int i = lastIndex + 1; i < frameCount; i++)
                        {
                            for (int toBlock = firstIndex; toBlock <= lastIndex; toBlock++)
                            {
                                for (int j = toBlock; j < i; j++)
                                {
                                    matrix[j * m + counter] = 1;
                                }
                                counter = counter + 1;
                            }
                        }
                    }
                    break;
                case TrackingStrategy.Blocks:
                    {
                        int m = GetShiftCount();

                        int counter = 0;
                        for (int shifts = 0; shifts < blockSize; shifts++)
                        {
                            int count = (frameCount - 1) - (shifts);
                            
                            for (int line = 0; line < count; line++)
                            {
                                for (int l = line; l <= line + (frameCount - 1) - count; l++)
                                {
                                    matrix[l * m + counter] = 1;
                                }
                                counter++;
                            }
                            
                        }
                    }
                    break;
                default:
                    break;
            }
            return matrix;
        }

        private void FillShiftPairs()
        {
            shiftPairs.Clear();
            switch (strategy)
            {
                case TrackingStrategy.Full:
                    for (int reference = 0; reference < frameCount; reference++)
                    {
                        for (int toTrack = reference + 1; toTrack < frameCount; toTrack++)
                        {
                            shiftPairs.Add(new ShiftPair(reference, toTrack));
                        }
                    }
                    break;
                case TrackingStrategy.OnlyOnReference:
                    for (int toTrack = 0; toTrack < frameCount; toTrack++)
                    {
                        if (toTrack != referenceIndex)
                        {
                            shiftPairs.Add(new ShiftPair(referenceIndex, toTrack));
                        }
                    }
                    break;
                case TrackingStrategy.OnReferenceBlock:
                    {
                        int m = GetShiftCount();
                        int firstIndex; //in block
                        int lastIndex;
                        if (referenceIndex < frameCount / 2)
                        {
                            firstIndex = referenceIndex - blockSize / 2;
                            if (firstIndex < 0)
                            {
                                firstIndex = 0;
                            }
                            lastIndex = firstIndex + blockSize - 1;
                        }
                        else
                        {
                            lastIndex = referenceIndex + blockSize / 2;
                            if (lastIndex >= frameCount)
                            {
                                lastIndex = frameCount - 1;
                            }
                            firstIndex = lastIndex - blockSize + 1;
                        }

                        //full shifts in block
                        for (int shifts = 0; shifts < blockSize - 1; shifts++)
                        {
                            int count = blockSize - 1 - (shifts);
                            for (int line = 0; line < count; line++)
                            {
                                shiftPairs.Add(new ShiftPair(line + firstIndex, line + firstIndex + blockSize - count));
                            }
                        }

                        //track all frames before block to all frames in block
                        for (int i = 0; i < firstIndex; i++)
                        {
                            for (int toBlock = firstIndex; toBlock <= lastIndex; toBlock++)
                            {
                                shiftPairs.Add(new ShiftPair(toBlock, i));
                            }
                        }

                        //track all frames after block to all frames in block
                        for (int i = lastIndex + 1; i < frameCount; i++)
                        {
                            for (int toBlock = firstIndex; toBlock <= lastIndex; toBlock++)
                            {
                                shiftPairs.Add(new ShiftPair(toBlock, i));
                            }
                        }
                    }
                    break;
                case TrackingStrategy.Blocks:
                    {
                        int m = GetShiftCount();

                        for (int shifts = 0; shifts < blockSize; shifts++)
                        {
                            int count = (frameCount - 1) - (shifts);
                            for (int line = 0; line < count; line++)
                            {
                                shiftPairs.Add(new ShiftPair(line, line + frameCount - count));
                                
                            }

                        }
                        //for (int reference = 0; reference < frameCount; reference++)
                        //{
                        //    for (int toTrack = reference + 1; toTrack - reference < blockSize; toTrack++)
                        //    {
                        //        shiftPairs.Add(new ShiftPair(reference, toTrack));
                        //    }
                        //}
                    }
                    break;
                default:
                    break;
            }
        }

        private void FillIndexTable()
        {
            indices = new int[frameCount, frameCount];
            for (int i = 0; i < frameCount; i++)
            {
                for (int j = 0; j < frameCount; j++)
                {
                    indices[i, j] = -1;
                }
            }

            
            switch (strategy)
            {
                case TrackingStrategy.Full:
                    {
                        int counter = 0;
                        for (int distance = 1; distance < frameCount; distance++)
                        {
                            for (int frame = 0; frame + distance < frameCount; frame++)
                            {
                                indices[frame, frame + distance] = counter;
                                counter++;
                            }
                        }
                    }
                    break;
                case TrackingStrategy.OnlyOnReference:
                    {
                        int counter = 0;
                        for (int toTrack = 0; toTrack < frameCount; toTrack++)
                        {
                            if (toTrack != referenceIndex)
                            {
                                indices[referenceIndex, toTrack] = counter;
                                counter++;
                            }
                        }
                    }
                    break;
                case TrackingStrategy.OnReferenceBlock:
                    {
                        int m = GetShiftCount();
                        int firstIndex; //in block
                        int lastIndex;
                        if (referenceIndex < frameCount / 2)
                        {
                            firstIndex = referenceIndex - blockSize / 2;
                            if (firstIndex < 0)
                            {
                                firstIndex = 0;
                            }
                            lastIndex = firstIndex + blockSize - 1;
                        }
                        else
                        {
                            lastIndex = referenceIndex + blockSize / 2;
                            if (lastIndex >= frameCount)
                            {
                                lastIndex = frameCount - 1;
                            }
                            firstIndex = lastIndex - blockSize + 1;
                        }

                        int counter = 0;
                        //full shifts in block
                        for (int shifts = 0; shifts < blockSize - 1; shifts++)
                        {
                            int count = blockSize - 1 - (shifts);
                            for (int line = 0; line < count; line++)
                            {
                                indices[line + firstIndex, line + firstIndex + blockSize - count] = counter;
                                counter++;
                            }
                        }

                        //track all frames before block to all frames in block
                        for (int i = 0; i < firstIndex; i++)
                        {
                            for (int toBlock = firstIndex; toBlock <= lastIndex; toBlock++)
                            {
                                indices[toBlock, i] = counter;
                                counter++;
                            }
                        }

                        //track all frames after block to all frames in block
                        for (int i = lastIndex + 1; i < frameCount; i++)
                        {
                            for (int toBlock = firstIndex; toBlock <= lastIndex; toBlock++)
                            {
                                indices[toBlock, i] = counter;
                                counter++;
                            }
                        }
                    }
                    break;
                case TrackingStrategy.Blocks:
                    {
                        int counter = 0;
                        for (int distance = 1; distance <= blockSize; distance++)
                        {
                            for (int frame = 0; frame + distance < frameCount; frame++)
                            {
                                indices[frame, frame + distance] = counter;
                                counter++;
                            }
                        }
                    }
                    break;
                default:
                    break;
            }
        }

        public ShiftCollection(int aFrameCount, int aMaxTileCountX, int aMaxTileCountY, int aReferenceIndex, TrackingStrategy aStrategy, int aBlockSize, CudaContext ctx)
        {
            strategy = aStrategy;
            referenceIndex = aReferenceIndex;
            frameCount = aFrameCount;
            if (aBlockSize >= aFrameCount)
            {
                blockSize = aFrameCount - 1;
            }
            else
            {
                blockSize = aBlockSize;
            }

            blas = new CudaBlas(PointerMode.Device, AtomicsMode.Allowed);
            one = 1.0f;
            zero = 0.0f;

            shiftPairs = new List<ShiftPair>();
            int shiftCount = GetShiftCount();
            FillShiftPairs();
            FillIndexTable();


            if (shiftPairs.Count != shiftCount)
            {
                throw new Exception("Ooups, something went wrong with my math...");
            }

            shifts = new List<NPPImage_32fC2>(shiftCount);

            int[] shiftPitches_h = new int[shiftCount];
            CUdeviceptr[] ptrList = new CUdeviceptr[shiftCount];
            for (int i = 0; i < shiftCount; i++)
            {
                NPPImage_32fC2 devVar = new NPPImage_32fC2(aMaxTileCountX, aMaxTileCountY);
                shifts.Add(devVar);
                shiftPitches_h[i] = devVar.Pitch;
                ptrList[i] = devVar.DevicePointer;
            }
            shiftPitches = shiftPitches_h;
            AllShifts_d = new CudaDeviceVariable<float2>(aMaxTileCountX * aMaxTileCountY * shiftCount);
            shiftsOneToOne_d = new CudaDeviceVariable<float2>(aMaxTileCountX * aMaxTileCountY * (frameCount - 1));
            shifts_d = ptrList;



            status = new CudaDeviceVariable<int>(aMaxTileCountX * aMaxTileCountY);
            infoInverse = new CudaDeviceVariable<int>(aMaxTileCountX * aMaxTileCountY);
            shiftMatrixArray = new CudaDeviceVariable<CUdeviceptr>(aMaxTileCountX * aMaxTileCountY);
            shiftMatrixSafeArray = new CudaDeviceVariable<CUdeviceptr>(aMaxTileCountX * aMaxTileCountY);
            matrixSquareArray = new CudaDeviceVariable<CUdeviceptr>(aMaxTileCountX * aMaxTileCountY);
            matrixInvertedArray = new CudaDeviceVariable<CUdeviceptr>(aMaxTileCountX * aMaxTileCountY);
            solvedMatrixArray = new CudaDeviceVariable<CUdeviceptr>(aMaxTileCountX * aMaxTileCountY);
            shiftOneToOneArray = new CudaDeviceVariable<CUdeviceptr>(aMaxTileCountX * aMaxTileCountY);
            shiftMeasuredArray = new CudaDeviceVariable<CUdeviceptr>(aMaxTileCountX * aMaxTileCountY);
            shiftOptimArray = new CudaDeviceVariable<CUdeviceptr>(aMaxTileCountX * aMaxTileCountY);
            shiftMatrices = new CudaDeviceVariable<float>(aMaxTileCountX * aMaxTileCountY * shiftCount * (frameCount - 1));
            shiftSafeMatrices = new CudaDeviceVariable<float>(aMaxTileCountX * aMaxTileCountY * shiftCount * (frameCount - 1));
            matricesSquared = new CudaDeviceVariable<float>(aMaxTileCountX * aMaxTileCountY * (frameCount - 1) * (frameCount - 1));
            matricesInverted = new CudaDeviceVariable<float>(aMaxTileCountX * aMaxTileCountY * (frameCount - 1) * (frameCount - 1));
            solvedMatrices = new CudaDeviceVariable<float>(aMaxTileCountX * aMaxTileCountY * shiftCount * (frameCount - 1));
            shiftsOneToOne = new CudaDeviceVariable<float2>(aMaxTileCountX * aMaxTileCountY * (frameCount - 1));
            pivotArray = new CudaDeviceVariable<int>(aMaxTileCountX * aMaxTileCountY * (frameCount - 1));
            shiftsMeasured = new CudaDeviceVariable<float2>(aMaxTileCountX * aMaxTileCountY * shiftCount);
            shiftsOptim = new CudaDeviceVariable<float2>(aMaxTileCountX * aMaxTileCountY * shiftCount);
            buffer = new CudaDeviceVariable<byte>(status.SumGetBufferSize());
            statusSum = new CudaDeviceVariable<int>(1);



            CUmodule mod = ctx.LoadModulePTX("ShiftMinimizerKernels.ptx");

            concatenateShifts = new concatenateShiftsKernel(ctx, mod);
            separateShifts = new separateShiftsKernel(ctx, mod);
            getOptimalShifts = new getOptimalShiftsKernel(ctx, mod);
            copyShiftMatrixKernel = new copyShiftMatrixKernel(ctx, mod);
            setPointers = new setPointersKernel(ctx, mod);
            checkForOutliers = new checkForOutliersKernel(ctx, mod);
            transposeShifts = new transposeShiftsKernel(ctx, mod);

            setPointers.RunSafe(shiftMatrixArray, shiftMatrixSafeArray, matrixSquareArray, matrixInvertedArray, solvedMatrixArray,
                shiftOneToOneArray, shiftMeasuredArray, shiftOptimArray, shiftMatrices, shiftSafeMatrices, matricesSquared,
                matricesInverted, solvedMatrices, shiftsOneToOne, shiftsMeasured, shiftsOptim, aMaxTileCountX * aMaxTileCountY, frameCount, shiftCount);

            

            Reset();
        }

        public void FreeResources()
        {
            shiftsOneToOne_d.Dispose();
            AllShifts_d.Dispose();
            shifts_d.Dispose();
            shiftPitches.Dispose();
            foreach (var item in shifts)
            {
                item.Dispose();
            }

            status.Dispose();
            infoInverse.Dispose();
            shiftMatrixArray.Dispose();
            shiftMatrixSafeArray.Dispose();
            matrixSquareArray.Dispose();
            matrixInvertedArray.Dispose();
            solvedMatrixArray.Dispose();
            shiftOneToOneArray.Dispose();
            shiftMeasuredArray.Dispose();
            shiftOptimArray.Dispose();
            shiftMatrices.Dispose();
            shiftSafeMatrices.Dispose();
            matricesSquared.Dispose();
            matricesInverted.Dispose();
            solvedMatrices.Dispose();
            shiftsOneToOne.Dispose();
            shiftsMeasured.Dispose();
            shiftsOptim.Dispose();
            buffer.Dispose();
            statusSum.Dispose();
            pivotArray.Dispose();
        }
        public void FreeUneededResources()
        {
            //shiftsOneToOne_d.Dispose();
            AllShifts_d.Dispose();
            shifts_d.Dispose();
            shiftPitches.Dispose();

            for (int i = 1; i < shifts.Count; i++)
            {
                shifts[i].Dispose();
            }

            status.Dispose();
            infoInverse.Dispose();
            shiftMatrixArray.Dispose();
            shiftMatrixSafeArray.Dispose();
            matrixSquareArray.Dispose();
            matrixInvertedArray.Dispose();
            solvedMatrixArray.Dispose();
            shiftOneToOneArray.Dispose();
            shiftMeasuredArray.Dispose();
            shiftOptimArray.Dispose();
            shiftMatrices.Dispose();
            shiftSafeMatrices.Dispose();
            matricesSquared.Dispose();
            matricesInverted.Dispose();
            solvedMatrices.Dispose();
            shiftsOneToOne.Dispose();
            shiftsMeasured.Dispose();
            shiftsOptim.Dispose();
            buffer.Dispose();
            statusSum.Dispose();
            pivotArray.Dispose();
        }

        private int GetIndex(int from, int to)
        {
            if (from >= frameCount || to >= frameCount)
                throw new ArgumentOutOfRangeException();

            int idx = indices[from, to];
            if (idx < 0)
                throw new ArgumentOutOfRangeException();

            return idx;
        }

        public void Reset()
        {
            for (int i = 0; i < shifts.Count; i++)
            {
                shifts[i].Set(new float[] { 0, 0 });
            }
        }

        public NPPImage_32fC2 getOptimalShift(int imageToTrack)
        {
            //assume measured shifts are not used anymore:
            NPPImage_32fC2 ret = shifts[0];
            ret.ResetRoi();

            //float2[] test2 = shiftsOneToOne_d;
            getOptimalShifts.RunSafe(ret, shiftsOneToOne_d, frameCount, referenceIndex, imageToTrack);
            //float2[] test = ret.ToCudaPitchedDeviceVariable();
            return ret;
        }

        public void dumpOptimisedShiftsToFile(string filename)
        {
            float2[] dump = shiftsOneToOne_d;

            FileStream fs = new FileStream(filename, FileMode.Create, FileAccess.Write);
            BinaryWriter bw = new BinaryWriter(fs);

            for (int i = 0; i < dump.Length; i++)
            {
                bw.Write(dump[i].x);
                bw.Write(dump[i].y);
            }

            bw.Close();
            bw.Dispose();
            fs.Close();
            fs.Dispose();
        }

        public void restoreOptimisedShifts(string filename)
        {
            float2[] dump = shiftsOneToOne_d;

            FileStream fs = new FileStream(filename, FileMode.Open, FileAccess.Read);
            BinaryReader br = new BinaryReader(fs);

            for (int i = 0; i < dump.Length; i++)
            {
                dump[i].x = br.ReadSingle();
                dump[i].y = br.ReadSingle();
            }
            shiftsOneToOne_d.CopyToDevice(dump);

            br.Close();
            br.Dispose();
            fs.Close();
            fs.Dispose();

        }

        public NPPImage_32fC2 this[int from, int to]
        {
            get { return shifts[GetIndex(from, to)]; }
        }

        public List<ShiftPair> GetShiftPairs()
        {
            return shiftPairs;
        }

        public void MinimizeCUBLAS(int tileCountX, int tileCountY)
        {
            int shiftCount;// = shifts.Count;
            shiftCount = GetShiftCount();

            concatenateShifts.RunSafe(shifts_d, shiftPitches, AllShifts_d, shiftCount, tileCountX, tileCountY);


            shiftsMeasured.CopyToDevice(AllShifts_d);

            CudaStopWatch sw = new CudaStopWatch();
            sw.Start();


            int imageCount = frameCount;
            int tileCount = tileCountX * tileCountY;
            int n1 = imageCount - 1;
            int m = shiftCount; 

            status.Memset(0);
            shiftMatrices.Memset(0);
            float[] shiftMatrix = CreateShiftMatrix();
            shiftMatrices.CopyToDevice(shiftMatrix, 0, 0, shiftMatrix.Length * sizeof(float));

            copyShiftMatrixKernel.RunSafe(shiftMatrices, tileCount, imageCount, shiftCount);
            shiftSafeMatrices.CopyToDevice(shiftMatrices);


            for (int i = 0; i < 10; i++)
            {
                blas.GemmBatched(Operation.Transpose, Operation.NonTranspose, n1, n1, m, one, shiftMatrixArray, m, shiftMatrixArray, m, zero, matrixSquareArray, n1, tileCount);
                //float[] mSqr = matricesSquared;

                if (n1 <= 32)
                {
                    //MatinvBatchedS can only invert up to 32x32 matrices
                    blas.MatinvBatchedS(n1, matrixSquareArray, n1, matrixInvertedArray, n1, infoInverse, tileCount);
                }
                else
                {
                    blas.GetrfBatchedS(n1, matrixSquareArray, n1, pivotArray, infoInverse, tileCount);
                    blas.GetriBatchedS(n1, matrixSquareArray, n1, pivotArray, matrixInvertedArray, n1, infoInverse, tileCount);
                }


                //int[] info = infoInverse;
                //mSqr = matricesInverted;
                blas.GemmBatched(Operation.NonTranspose, Operation.Transpose, n1, m, n1, one, matrixInvertedArray, n1, shiftMatrixArray, m, zero, solvedMatrixArray, n1, tileCount);
                blas.GemmBatched(Operation.NonTranspose, Operation.Transpose, n1, 2, m, one, solvedMatrixArray, n1, shiftMeasuredArray, 2, zero, shiftOneToOneArray, n1, tileCount);
                blas.GemmBatched(Operation.NonTranspose, Operation.NonTranspose, m, 2, n1, one, shiftMatrixArray, m, shiftOneToOneArray, n1, zero, shiftOptimArray, m, tileCount);

                checkForOutliers.RunSafe(shiftsMeasured, shiftsOptim, shiftMatrices, status, infoInverse, tileCount, imageCount, shiftCount);

                status.Sum(statusSum, buffer, 0);
                int[] stats = status;

                for (int j = 0; j < tileCount; j++)
                {
                    if (stats[j] >= 0)
                        Console.Write(j + ": " + stats[j] + "; ");
                }
                Console.WriteLine();

                int stat = statusSum;
                if (stat == -tileCount)
                {
                    break;
                }

                //float2[] AllShifts_h = shiftsMeasured;
            }

            blas.GemmBatched(Operation.NonTranspose, Operation.NonTranspose, m, 2, n1, one, shiftMatrixSafeArray, m, shiftOneToOneArray, n1, zero, shiftMeasuredArray, m, tileCount);

            AllShifts_d.Memset(0);
            transposeShifts.RunSafe(AllShifts_d, shiftsMeasured, shiftsOneToOne, shiftsOneToOne_d, tileCount, imageCount, shiftCount);
            //shiftsMeasured.CopyToDevice(AllShifts_d);

            //float2[] AllShiftsFinal_h = shiftsMeasured;

            sw.Stop();
            Console.WriteLine("Time for optimisation: " + sw.GetElapsedTime() + " msec.");

            separateShifts.RunSafe(AllShifts_d, shifts_d, shiftPitches, shiftCount, tileCountX, tileCountY);
        }


    }
}
