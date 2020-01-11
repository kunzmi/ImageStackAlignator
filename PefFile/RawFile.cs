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
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PentaxPefFile
{
    public abstract class RawFile : FileReader
    {
        public enum BayerColor : int
        {
            Red = 0,
            Green = 1,
            Blue = 2,
            Cyan = 3,
            Magenta = 4,
            Yellow = 5,
            White = 6
        }

        public RawFile(string aFileName)
            : base(aFileName)
        {
            //set some default values:

            //color channels are assumed in order RGB if no tag present
            mColorTwist = new float[3,4];
            mColorTwist[0, 0] = 1.0f;
            mColorTwist[1, 1] = 1.0f;
            mColorTwist[2, 2] = 1.0f;
            mColorTwistIsIdentity = true;

            //it seems as if only Pentax is capable of giving us that info
            mRollAnglePresent = false;
            mRecordingDate = new DateTime();

            //read all IFDs/Tags:
            byte a = mFileReader.ReadByte();
            byte b = mFileReader.ReadByte();

            bool fileIsLittleEndian;
            if (a == b && b == 'I')
                fileIsLittleEndian = true;
            else
                if (a == b && b == 'M')
                fileIsLittleEndian = false;
            else
                throw new FileLoadException("Could not determine file endianess. Is this a proper TIFF/PEF/DNG file?", aFileName);

            mEndianSwap = fileIsLittleEndian != BitConverter.IsLittleEndian;

            ushort magicNumber = ReadUI2();

            if (magicNumber != 42)
                throw new FileLoadException("This is not a valid TIFF/PEF/DNG file: Magic number is not 42.", aFileName);

            uint offsetToFirstIFD = ReadUI4();

            mFile.Seek(offsetToFirstIFD, SeekOrigin.Begin);
            mIfds = new List<ImageFileDirectory>();
            while (true)
            {
                ImageFileDirectory ifd = new ImageFileDirectory(this);
                mIfds.Add(ifd);
                uint offsetToNext = ReadUI4();
                if (offsetToNext == 0)
                    break;
                Seek(offsetToNext, System.IO.SeekOrigin.Begin);
            }

            //until here PEF and DNG are the same. They diverge on how to read the tags
        }

        protected List<ImageFileDirectory> mIfds;
        protected DNGColorSpec mColorSpec;
        protected DNGOrientation mOrientation;
        protected ushort[,] mRawImage;
        protected int mWidth;
        protected int mHeight;
        protected int mCropLeft;
        protected int mCropTop;
        protected int mCroppedWidth;
        protected int mCroppedHeight;
        protected int mBitDepth;
        protected int mISO;
        protected BayerColor[] mBayerPattern;
        protected float[,] mColorTwist;
        protected bool mColorTwistIsIdentity;
        protected float[] mWhiteLevel;
        protected float[] mBlackLevel;
        protected float[] mWhiteBalance;
        protected float mRollAngle;
        protected bool mRollAnglePresent;
        protected float mNoiseModelAlpha;
        protected float mNoiseModelBeta;
        protected Rational mExposureTime;
        protected DateTime mRecordingDate;
        protected string mMake;
        protected string mUniqueModelName;




        public ushort[,] RawImage
        {
            get { return mRawImage; }
        }
        public int RawWidth
        {
            get { return mWidth; }
        }
        public int RawHeight
        {
            get { return mHeight; }
        }
        public int CropLeft
        {
            get { return mCropLeft; }
        }
        public int CropTop
        {
            get { return mCropTop; }
        }
        public int CroppedWidth
        {
            get { return mCroppedWidth; }
        }
        public int CroppedHeight
        {
            get { return mCroppedHeight; }
        }
        public DNGOrientation Orientation
        {
            get { return mOrientation; }
        }
        public float[,] ColorTwist
        {
            get { return mColorTwist; }
        }
        public bool ColorTwistIsIdentity
        {
            get { return mColorTwistIsIdentity; }
        }
        public BayerColor[] BayerPattern
        {
            get { return mBayerPattern; }
        }
        public float RollAngle
        {
            get { return mRollAngle; }
        }
        public bool RollAnglePresent
        {
            get { return mRollAnglePresent; }
        }
        public Rational ExposureTime
        {
            get { return mExposureTime; }
        }
        public DateTime RecordingDate
        {
            get { return mRecordingDate; }
        }

        /// <summary>
        /// In R G B order
        /// </summary>
        public float[] WhiteLevel 
        {
            get { return mWhiteLevel; } 
        }
        /// <summary>
        /// In R G B order
        /// </summary>
        public float[] BlackLevel 
        { 
            get { return mBlackLevel; } 
        }
        /// <summary>
        /// In R G B order
        /// </summary>
        public float[] WhiteBalance 
        {
            get { return mWhiteBalance; } 
        }
        public int ISO 
        { 
            get { return mISO; }
        }

        public float NoiseModelAlpha
        {
            get { return mNoiseModelAlpha; }
        }

        public float NoiseModelBeta
        {
            get { return mNoiseModelBeta; }
        }

        public string Make
        {
            get { return mMake; }
        }

        public string UniqueModelName
        {
            get { return mUniqueModelName; }
        }

        public DNGColorSpec ColorSpec
        {
            get { return mColorSpec; }
        }

        public bool LoadExtraCameraProfile(string xmlFileName)
        {
            PentaxPefFile.ExtraCameraProfiles extraCameraProfiles = PentaxPefFile.ExtraCameraProfiles.Load(xmlFileName);
            PentaxPefFile.ExtraCameraProfile p = extraCameraProfiles.GetProfile(mMake, mUniqueModelName);

            bool setAnything = false;
            if (p == null)
            {
                return false;                
            }

            if (p.CropInfo != null)
            {
                int cropLeft = p.CropInfo.Left;
                int cropTop = p.CropInfo.Top;

                int croppedWidth = p.CropInfo.Width;
                int croppedHeight = p.CropInfo.Height;            

                //we always crop at least two pixels because of our algos...
                mCropLeft = Math.Max(2, cropLeft);
                mCropTop = Math.Max(2, cropTop);

                mCroppedWidth = croppedWidth - Math.Max(0, (cropLeft + croppedWidth) - (mWidth - 2));
                mCroppedHeight = croppedHeight - Math.Max(0, (cropTop + croppedHeight) - (mHeight - 2));
                setAnything = true;
            }

            if (p.ColorMatrix1 != null)
            {
                mColorSpec = new DNGColorSpec(p.ColorMatrix1, p.ColorMatrix2, p.Illuminant1, p.Illuminant2, mWhiteBalance);
                setAnything = true;
            }

            if (p.NoiseModel != null)
            {
                double a, b;
                (a, b) = p.NoiseModel.GetValue(mISO);
                mNoiseModelAlpha = (float)a;
                mNoiseModelBeta = (float)b;
                setAnything = true;
            }
            return setAnything;
        }
    }
}
