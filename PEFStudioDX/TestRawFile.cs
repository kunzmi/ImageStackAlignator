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
    public class TestRawFile : PentaxPefFile.RawFile
    {
        public TestRawFile(string dummyFile, int size, float preShiftX, float preShiftY, float preRotDeg, float shiftX, float shiftY)
            : base(dummyFile)
        {
            //close the file
            Close();
            Random rand = new Random(0);

            mRawImage = new ushort[size, size];

            float[] temp = new float[size * size];
            NPPImage_32fC1 img1 = new NPPImage_32fC1(size, size);
            NPPImage_32fC1 img2 = new NPPImage_32fC1(size, size);
            NPPImage_16uC1 img16u = new NPPImage_16uC1(size, size);

            for (int i = 0; i < temp.Length; i++)
            {
                temp[i] = (float)rand.NextDouble();
            }

            img1.CopyToDevice(temp);
            img1.FilterGaussBorder(img2, MaskSize.Size_5_X_5, NppiBorderType.Replicate);
            img1.Set(0);
            img2.WarpAffine(img1, Matrix3x3.ShiftAffine(-preShiftX, -preShiftY).ToAffine(), InterpolationMode.Cubic);
            img2.Set(0);
            img1.WarpAffine(img2, Matrix3x3.RotAroundCenter(-preRotDeg, size, size).ToAffine(), InterpolationMode.Cubic);
            img1.Set(0);
            img2.WarpAffine(img1, Matrix3x3.ShiftAffine(-shiftX, -shiftY).ToAffine(), InterpolationMode.Cubic);
            img1.Mul(65535);
            img1.Convert(img16u, NppRoundMode.Near);

            img16u.CopyToHost(mRawImage, size * sizeof(ushort));

            double[] colorMatrix = new double[] { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
            mColorSpec = new PentaxPefFile.DNGColorSpec(colorMatrix, colorMatrix, PentaxPefFile.IFDDNGCalibrationIlluminant.Illuminant.D50, 
                PentaxPefFile.IFDDNGCalibrationIlluminant.Illuminant.D65, new float[] {1f, 1,1f });
            mOrientation =  new PentaxPefFile.DNGOrientation(PentaxPefFile.DNGOrientation.Orientation.Normal);

            mWidth = size;
            mHeight = size;
            mCropLeft = 0;
            mCropTop = 0;
            mCroppedWidth = size;
            mCroppedHeight = size;
            mBitDepth = 16;
            mISO = 100;
            mBayerPattern = new BayerColor[] { BayerColor.Red, BayerColor.Cyan, BayerColor.Blue };
            mWhiteLevel = new float[] { 65535, 65535, 65535 };
            mBlackLevel = new float[] { 0, 0, 0 };
            mWhiteBalance = new float[] { 1, 1, 1 }; ;
            mRollAngle = 0;
            mRollAnglePresent = false;
            mNoiseModelAlpha = float.Epsilon;
            mNoiseModelBeta = 0;
            mExposureTime = new PentaxPefFile.Rational(1, 1);
            mRecordingDate = DateTime.Now;
            mMake = "None";
            mUniqueModelName = "Test file";

            img1.Dispose();
            img2.Dispose();
            img16u.Dispose();
        }
    }
}
