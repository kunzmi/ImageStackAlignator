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
using System.Drawing;
using System.Drawing.Imaging;
using System.Threading.Tasks;

namespace PentaxPefFile
{
	public class PEFFile : RawFile
	{
        private MNHuffmanTable mHhuffmanTable;

        #region "Inspired" by dcraw
        //these functions are a bit different than in PNG!
        uint bitbuf = 0;
        int vbits = 0, reset = 0;
        private unsafe uint getbithuff(int nbits, ushort* huff)
        {
            uint c;

            if (nbits > 25) return 0;
            if (nbits < 0)
            {
                reset = 0;
                vbits = 0;
                bitbuf = 0;
                return 0;
            }

            if (nbits == 0 || vbits < 0) return 0;

            while (!(reset != 0) && vbits < nbits /*&& (c = ReadUI1() fgetc(ifp)) != EOF && !(reset = 0 && c == 0xff && fgetc(ifp))*/)
            {
                c = ReadUI1();
                bitbuf = (bitbuf << 8) + (byte)c;
                vbits += 8;
            }
            c = bitbuf << (32 - vbits) >> (32 - nbits);
            if (huff != null)
            {
                vbits -= huff[c] >> 8;
                c = (byte)huff[c];
            }
            else
                vbits -= nbits;
            return c;
        }

        private unsafe int ljpeg_diff(ushort* huff)
        {
            int len, diff;
            len = (int)getbithuff(*huff, huff + 1);
            diff = (int)getbithuff(len, null);
            if ((diff & (1 << (len - 1))) == 0)
                diff -= (1 << len) - 1;
            return diff;
        }
        #endregion

        public PEFFile(string aFileName)
            : base(aFileName)
        {
            ImageFileDirectory raw = SetMembers();

            mRawImage = new ushort[mHeight, mWidth];
            if (mHhuffmanTable != null && raw.GetEntry<IFDStripOffsets>().Value.Length == 1)
            {
			    uint offset = raw.GetEntry<IFDStripOffsets>().Value[0];

			    Seek(offset, SeekOrigin.Begin);
			    int[,] vpred = new int[2,2];
			    int[]  hpred = new int[2];
                unsafe
                {
                    fixed (ushort* huff = mHhuffmanTable.Value)
                    {
                        getbithuff(-1, null);

                        for (int row = 0; row < mHeight; row++)
                            for (int col = 0; col < mWidth; col++)
                            {
                                int diff = ljpeg_diff(huff);
                                if (col < 2)
                                    hpred[col] = vpred[row & 1, col] += diff;
                                else
                                    hpred[col & 1] += diff;
                                mRawImage[row, col] = (ushort)hpred[col & 1];
                            }
                    }
                }
            }
            else if (mHhuffmanTable != null)
            {
                throw new Exception("TODO: implement compressed multiple stripes");
            }
            else
            {
                //uncompressed

                uint[] sizes = raw.GetEntry<IFDStripByteCounts>().Value;
                uint[] offsets = raw.GetEntry<IFDStripOffsets>().Value;
                uint rowsPerStrip = raw.GetEntry<IFDRowsPerStrip>().Value;

                
                for (int strip = 0; strip < offsets.Length; strip++)
                {
                    byte[] data;
                    Seek(offsets[strip], SeekOrigin.Begin);
                    data = mFileReader.ReadBytes((int)sizes[strip]);

                    uint pos = 0;
                    int bitOffset = 0;

                    for (int line = 0; line < rowsPerStrip; line++)
                    {
                        for (int pixel = 0; pixel < mWidth; pixel++)
                        {
                            int row = strip * (int)rowsPerStrip + line;
                            int col = pixel;

                            uint val = data[pos];

                            while (bitOffset < mBitDepth)
                            {
                                val = data[pos];
                                bitbuf = (bitbuf << 8) + val;
                                bitOffset += 8;
                                pos++;
                            }

                            val = bitbuf << (32 - bitOffset) >> (32 - mBitDepth);
                            bitOffset -= mBitDepth;

                            mRawImage[row, col] = (ushort)(val);
                        }
                    }
                    
                }


            }

            //close the file
            Close();
        }

        public PEFFile(string aFileName, bool HeaderOnly)
            : base(aFileName)
        {

            SetMembers();
            //close the file
            Close();
        }

        private ImageFileDirectory SetMembers()
        {
            //Raw Data:
            ImageFileDirectory raw = mIfds[0];
            IFDExif exif = raw.GetEntry<IFDExif>();
            mISO = exif.GetEntry<ExifISOSpeedRatings>().Value;
            mExposureTime = exif.GetEntry<ExifExposureTime>().Value;
            mRecordingDate = exif.GetEntry<ExifDateTimeDigitized>().Value;
            mBitDepth = raw.GetEntry<IFDBitsPerSample>().Value[0];
            ExifMakerNote makernote = exif.GetEntry<ExifMakerNote>();
            //needed to decompress image data:
            mHhuffmanTable = makernote.Value.GetEntry<MNHuffmanTable>();
            ExifCFAPattern.BayerColor[] bayer = exif.GetEntry<ExifCFAPattern>().Value;
            mBayerPattern = new BayerColor[bayer.Length];
            for (int i = 0; i < bayer.Length; i++)
            {
                mBayerPattern[i] = (BayerColor)(int)bayer[i];
            }
            
            MNWhiteLevel whiteLevel = makernote.Value.GetEntry<MNWhiteLevel>();
            MNWhitePoint whitePoint = makernote.Value.GetEntry<MNWhitePoint>();
            MNBlackPoint blackPoint = makernote.Value.GetEntry<MNBlackPoint>();
            MNDataScaling scaling = makernote.Value.GetEntry<MNDataScaling>();

            float whiteLevelAll = (float)Math.Pow(2, mBitDepth);
            if (whiteLevel != null)
                whiteLevelAll = whiteLevel.Value;

            mBlackLevel = new float[3];
            if (blackPoint != null)
            {
                //only one value for all colors
                if (blackPoint.Value.Length == 1)
                {
                    mBlackLevel[0] = (float)blackPoint.Value[0];
                    mBlackLevel[1] = (float)blackPoint.Value[0];
                    mBlackLevel[2] = (float)blackPoint.Value[0];
                }

                //values per color channel
                if (blackPoint.Value.Length == 3)
                {
                    mBlackLevel[0] = (float)blackPoint.Value[0];
                    mBlackLevel[1] = (float)blackPoint.Value[1];
                    mBlackLevel[2] = (float)blackPoint.Value[2];
                }

                //values per color bayer pattern
                if (blackPoint.Value.Length == 4)
                {
                    //red
                    int indexR = -1;
                    for (int i = 0; i < mBayerPattern.Length; i++)
                    {
                        if (mBayerPattern[i] == BayerColor.Red)
                        {
                            indexR = i;
                            break;
                        }
                    }
                    mBlackLevel[0] = (float)blackPoint.Value[indexR];

                    //blue
                    int indexB = -1;
                    for (int i = 0; i < mBayerPattern.Length; i++)
                    {
                        if (mBayerPattern[i] == BayerColor.Blue)
                        {
                            indexB = i;
                            break;
                        }
                    }
                    mBlackLevel[2] = (float)blackPoint.Value[indexB];

                    //green, the two remaining indices
                    int indexG1 = -1, indexG2 = -1;
                    for (int i = 0; i < mBayerPattern.Length; i++)
                    {
                        if (mBayerPattern[i] == BayerColor.Green && indexG1 == -1)
                        {
                            indexG1 = i;
                        }
                        if (mBayerPattern[i] == BayerColor.Green && indexG1 != -1)
                        {
                            indexG2 = i;
                        }
                    }
                    float g1 = (float)blackPoint.Value[indexG1];
                    float g2 = (float)blackPoint.Value[indexG2];

                    mBlackLevel[1] = Math.Max(g1, g2); //well, one could distinguish the two greens, but what for?
                }
            }

            mWhiteLevel = new float[] { whiteLevelAll, whiteLevelAll, whiteLevelAll };
            mWhiteLevel[0] -= mBlackLevel[0];
            mWhiteLevel[1] -= mBlackLevel[1];
            mWhiteLevel[2] -= mBlackLevel[2];
            float scale = scaling.Value;
            mWhiteBalance = new float[] { whitePoint.Value[0] / scale, whitePoint.Value[1] / scale, whitePoint.Value[3] / scale };

            if (makernote.Value.GetEntry<MNLevelInfo>() != null)
            {
                mRollAngle = makernote.Value.GetEntry<MNLevelInfo>().Value.RollAngle;
                mRollAnglePresent = true;
            }

            mWidth = (int)raw.GetEntry<IFDImageWidth>().Value;
            mHeight = (int)raw.GetEntry<IFDImageLength>().Value;


            //look for orientation tag.
            if (raw.GetEntry<IFDOrientation>() != null)
            {
                mOrientation = new DNGOrientation(raw.GetEntry<IFDOrientation>().Value);
            }
            else
            {
                //no tag found, use default
                mOrientation = new DNGOrientation(DNGOrientation.Orientation.Normal);
            }

            //we always crop at least two pixels because of our algos...
            mCropLeft = 2;
            mCropTop = 2;

            mCroppedWidth = mWidth - 4;
            mCroppedHeight = mHeight - 4;


            mMake = raw.GetEntry<IFDMake>().Value;
            mUniqueModelName = raw.GetEntry<IFDModel>().Value;

            //missing data, like noise model, crop area, etc..., must be loaded afterwards!
            double[] colorMatrix = new double[] { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
            mColorSpec = new DNGColorSpec(colorMatrix, colorMatrix, IFDDNGCalibrationIlluminant.Illuminant.D50, IFDDNGCalibrationIlluminant.Illuminant.D50, mWhiteBalance);
            return raw;
        }
    }
}
