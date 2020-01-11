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
using System.Runtime.InteropServices;

namespace PentaxPefFile
{
	public abstract class ExifEntry : SaveTiffTag
    {
		protected FileReader mPefFile;
		protected ushort mTagID;
		protected TIFFValueType mFieldType;
		protected uint mValueCount;
		protected uint mOffset;

		public ExifEntry(FileReader aPefFile)
		{
			mPefFile = aPefFile;
			mTagID = mPefFile.ReadUI2();
			mFieldType = new TIFFValueType((TIFFValueTypes)mPefFile.ReadUI2());
			mValueCount = mPefFile.ReadUI4();
			mOffset = mPefFile.ReadUI4();
		}

		protected ExifEntry(FileReader aPefFile, ushort aTagID)
		{
			mPefFile = aPefFile;
			mTagID = aTagID;
			mFieldType = new TIFFValueType((TIFFValueTypes)mPefFile.ReadUI2());
			mValueCount = mPefFile.ReadUI4();
			mOffset = mPefFile.ReadUI4();
        }

        protected ExifEntry(ushort aTagID, TIFFValueType aFieldType, uint aValueCount)
        {
            mPefFile = null;
            mTagID = aTagID;
            mFieldType = aFieldType;
            mValueCount = aValueCount;
            mOffset = 0;
        }

        public override void SavePass1(Stream stream)
        {
            base.SavePass1(stream);
        }
        public override void SavePass2(Stream stream)
        {
            base.SavePass2(stream);
        }

        protected virtual uint WriteEntryHeader(uint offsetOrValue, Stream stream, int valueCount = -1)
        {
            BinaryWriter br = new BinaryWriter(stream, Encoding.ASCII, true);
            br.Write(mTagID);
            br.Write(mFieldType.GetValue());
            if (valueCount == -1)
            {
                br.Write(mValueCount);
            }
            else
            {
                br.Write(valueCount);
            }
            uint streamPosition = (uint)stream.Position;
            br.Write(offsetOrValue);
            br.Dispose();
            return streamPosition;
        }

        protected virtual void WritePass2(byte[] data, Stream stream)
        {
            stream.Seek(0, SeekOrigin.End);
            uint offset = (uint)stream.Position;
            stream.Write(data, 0, data.Length);
            stream.Seek(mOffsetInStream, SeekOrigin.Begin);
            byte[] bytes = BitConverter.GetBytes(offset);
            stream.Write(bytes, 0, 4);
            stream.Seek(0, SeekOrigin.End);
        }

        public static ExifEntry CreateExifEntry(FileReader aPefFile)
		{
			ushort tagID = aPefFile.ReadUI2();

			switch (tagID)	
			{
				case ExifExposureTime.TagID:
					return new ExifExposureTime(aPefFile, tagID);
				case ExifFNumber.TagID:
					return new ExifFNumber(aPefFile, tagID);
				case ExifExposureProgram.TagID:
					return new ExifExposureProgram(aPefFile, tagID);
				case ExifSpectralSensitivity.TagID:
					return new ExifSpectralSensitivity(aPefFile, tagID);
				case ExifISOSpeedRatings.TagID:
					return new ExifISOSpeedRatings(aPefFile, tagID);
				case ExifOECF.TagID:
					return new ExifOECF(aPefFile, tagID);
				case ExifExifVersion.TagID:
					return new ExifExifVersion(aPefFile, tagID);
				case ExifDateTimeOriginal.TagID:
					return new ExifDateTimeOriginal(aPefFile, tagID);
				case ExifDateTimeDigitized.TagID:
					return new ExifDateTimeDigitized(aPefFile, tagID);
				case ExifComponentsConfiguration.TagID:
					return new ExifComponentsConfiguration(aPefFile, tagID);
				case ExifCompressedBitsPerPixel.TagID:
					return new ExifCompressedBitsPerPixel(aPefFile, tagID);
				case ExifShutterSpeedValue.TagID:
					return new ExifShutterSpeedValue(aPefFile, tagID);
				case ExifApertureValue.TagID:
					return new ExifApertureValue(aPefFile, tagID);
				case ExifBrightnessValue.TagID:
					return new ExifBrightnessValue(aPefFile, tagID);
				case ExifExposureBiasValue.TagID:
					return new ExifExposureBiasValue(aPefFile, tagID);
				case ExifMaxApertureValue.TagID:
					return new ExifMaxApertureValue(aPefFile, tagID);
				case ExifSubjectDistance.TagID:
					return new ExifSubjectDistance(aPefFile, tagID);
				case ExifMeteringMode.TagID:
					return new ExifMeteringMode(aPefFile, tagID);
				case ExifLightSource.TagID:
					return new ExifLightSource(aPefFile, tagID);
				case ExifFlash.TagID:
					return new ExifFlash(aPefFile, tagID);
				case ExifFocalLength.TagID:
					return new ExifFocalLength(aPefFile, tagID);
				case ExifSubjectArea.TagID:
					return new ExifSubjectArea(aPefFile, tagID);
				case ExifMakerNote.TagID:
					return new ExifMakerNote(aPefFile, tagID);
				case ExifUserComment.TagID:
					return new ExifUserComment(aPefFile, tagID);
				case ExifSubsecTime.TagID:
					return new ExifSubsecTime(aPefFile, tagID);
				case ExifSubsecTimeOriginal.TagID:
					return new ExifSubsecTimeOriginal(aPefFile, tagID);
				case ExifSubsecTimeDigitized.TagID:
					return new ExifSubsecTimeDigitized(aPefFile, tagID);
				case ExifFlashpixVersion.TagID:
					return new ExifFlashpixVersion(aPefFile, tagID);
				case ExifColorSpace.TagID:
					return new ExifColorSpace(aPefFile, tagID);
				case ExifPixelXDimension.TagID:
					return new ExifPixelXDimension(aPefFile, tagID);
				case ExifPixelYDimension.TagID:
					return new ExifPixelYDimension(aPefFile, tagID);
				case ExifRelatedSoundFile.TagID:
					return new ExifRelatedSoundFile(aPefFile, tagID);
				case ExifFlashEnergy.TagID:
					return new ExifFlashEnergy(aPefFile, tagID);
				case ExifSpatialFrequencyResponse.TagID:
					return new ExifSpatialFrequencyResponse(aPefFile, tagID);
				case ExifFocalPlaneXResolution.TagID:
					return new ExifFocalPlaneXResolution(aPefFile, tagID);
				case ExifFocalPlaneYResolution.TagID:
					return new ExifFocalPlaneYResolution(aPefFile, tagID);
				case ExifFocalPlaneResolutionUnit.TagID:
					return new ExifFocalPlaneResolutionUnit(aPefFile, tagID);
				case ExifSubjectLocation.TagID:
					return new ExifSubjectLocation(aPefFile, tagID);
				case ExifExposureIndex.TagID:
					return new ExifExposureIndex(aPefFile, tagID);
				case ExifSensingMethod.TagID:
					return new ExifSensingMethod(aPefFile, tagID);
				case ExifFileSource.TagID:
					return new ExifFileSource(aPefFile, tagID);
				case ExifSceneType.TagID:
					return new ExifSceneType(aPefFile, tagID);
				case ExifCFAPattern.TagID:
					return new ExifCFAPattern(aPefFile, tagID);
				case ExifCustomRendered.TagID:
					return new ExifCustomRendered(aPefFile, tagID);
				case ExifExposureMode.TagID:
					return new ExifExposureMode(aPefFile, tagID);
				case ExifWhiteBalance.TagID:
					return new ExifWhiteBalance(aPefFile, tagID);
				case ExifDigitalZoomRatio.TagID:
					return new ExifDigitalZoomRatio(aPefFile, tagID);
				case ExifFocalLengthIn35mmFilm.TagID:
					return new ExifFocalLengthIn35mmFilm(aPefFile, tagID);
				case ExifSceneCaptureType.TagID:
					return new ExifSceneCaptureType(aPefFile, tagID);
				case ExifGainControl.TagID:
					return new ExifGainControl(aPefFile, tagID);
				case ExifContrast.TagID:
					return new ExifContrast(aPefFile, tagID);
				case ExifSaturation.TagID:
					return new ExifSaturation(aPefFile, tagID);
				case ExifSharpness.TagID:
					return new ExifSharpness(aPefFile, tagID);
				case ExifDeviceSettingDescription.TagID:
					return new ExifDeviceSettingDescription(aPefFile, tagID);
				case ExifSubjectDistanceRange.TagID:
					return new ExifSubjectDistanceRange(aPefFile, tagID);
				case ExifImageUniqueID.TagID:
					return new ExifImageUniqueID(aPefFile, tagID);
				case ExifSensitivityType.TagID:
					return new ExifSensitivityType(aPefFile, tagID);
				case ExifStandardOutputSensitivity.TagID:
					return new ExifStandardOutputSensitivity(aPefFile, tagID);
				case ExifBodySerialNumber.TagID:
					return new ExifBodySerialNumber(aPefFile, tagID);
				case ExifLensSpecification.TagID:
					return new ExifLensSpecification(aPefFile, tagID);
				case ExifLensMake.TagID:
					return new ExifLensMake(aPefFile, tagID);
				case ExifLensModel.TagID:
					return new ExifLensModel(aPefFile, tagID);
                case ExifLensSerialNumber.TagID:
                    return new ExifLensSerialNumber(aPefFile, tagID);
                default:
					return new ExifUnknownTag(aPefFile, tagID);
			}
		}
    }

    #region Typed base classes
	public class StringExifEntry : ExifEntry
	{
		protected string mValue;

        public StringExifEntry(string aValue, ushort aTagID)
            : base(aTagID, new TIFFValueType(TIFFValueTypes.Ascii), 0)
        {
            mValue = aValue;
        }

        public StringExifEntry(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			if (mFieldType.SizeInBytes * mValueCount <= 4)
			{
				unsafe
				{
					fixed (uint* ptr = &mOffset)
					{
						byte* bptr = (byte*)ptr;
						byte[] text = new byte[4];

						for (int i = 0; i < mValueCount; i++)
						{
							if (aPefFile.EndianSwap)
								text[i] = bptr[4 / mFieldType.SizeInBytes - i - 1];
							else
								text[i] = bptr[i];
						}

						mValue = ASCIIEncoding.ASCII.GetString(text).Replace("\0", string.Empty);
					}
				}
			}
			else
			{
				uint currentOffset = mPefFile.Position();
				mPefFile.Seek(mOffset, System.IO.SeekOrigin.Begin);

				mValue = mPefFile.ReadStr((int)mFieldType.SizeInBytes * (int)mValueCount).Replace("\0", string.Empty);

				mPefFile.Seek(currentOffset, System.IO.SeekOrigin.Begin);
			}
			mValue = mValue.TrimEnd(' ');
        }

        public override void SavePass1(Stream stream)
        {
            byte[] temp = ASCIIEncoding.ASCII.GetBytes(mValue);
            int tempLength = Math.Max(4, temp.Length + 1);
            byte[] valueToWrite = new byte[tempLength];
            for (int i = 0; i < temp.Length; i++)
            {
                valueToWrite[i] = temp[i];
            }
            valueToWrite[temp.Length] = 0;

            if (valueToWrite.Length <= 4)
            {
                unsafe
                {
                    fixed (byte* ptr = valueToWrite)
                    {
                        uint* uiptr = (uint*)ptr;
                        uint val = *uiptr;

                        WriteEntryHeader(val, stream, temp.Length + 1);
                        mOffsetInStream = 0;
                    }
                }
            }
            else
            {
                mOffsetInStream = WriteEntryHeader(0, stream, temp.Length + 1);
            }
        }
        public override void SavePass2(Stream stream)
        {
            if (mOffsetInStream != 0)
            {
                byte[] temp = ASCIIEncoding.ASCII.GetBytes(mValue);
                byte[] valueToWrite = new byte[temp.Length + 1];
                for (int i = 0; i < temp.Length; i++)
                {
                    valueToWrite[i] = temp[i];
                }
                valueToWrite[temp.Length] = 0;

                WritePass2(valueToWrite, stream);
            }
        }
    }

	public class ByteExifEntry : ExifEntry
	{
		protected byte[] mValue;

        public ByteExifEntry(byte[] aValue, ushort aTagID)
            : base(aTagID, new TIFFValueType(TIFFValueTypes.Byte), (uint)aValue.Length)
        {
            mValue = aValue;
        }

        public ByteExifEntry(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			if (mFieldType.SizeInBytes * mValueCount <= 4)
			{
				unsafe
				{
					fixed (uint* ptr = &mOffset)
					{
						byte* ptrUS = (byte*)ptr;
						mValue = new byte[mValueCount];
						for (int i = 0; i < mValueCount; i++)
						{
							if (aPefFile.EndianSwap)
								mValue[i] = ptrUS[4 / mFieldType.SizeInBytes - i - 1];
							else
								mValue[i] = ptrUS[i];
						}
					}
				}
			}
			else
			{
				uint currentOffset = mPefFile.Position();
				mPefFile.Seek(mOffset, System.IO.SeekOrigin.Begin);

				mValue = new byte[mFieldType.SizeInBytes * mValueCount];
				for (int i = 0; i < mFieldType.SizeInBytes * mValueCount; i++)
				{
					mValue[i] = mPefFile.ReadUI1();
				}
				mPefFile.Seek(currentOffset, System.IO.SeekOrigin.Begin);
			}
        }

        public override void SavePass1(Stream stream)
        {
            if (mFieldType.SizeInBytes * mValue.Length <= 4)
            {
                byte[] valueToWrite;
                if (mValue.Length == 4)
                {
                    valueToWrite = mValue;
                }
                else
                {
                    valueToWrite = new byte[4];
                    for (int i = 0; i < mValue.Length; i++)
                    {
                        valueToWrite[i] = mValue[i];
                    }
                }

                unsafe
                {
                    fixed (byte* ptr = valueToWrite)
                    {
                        uint* uiptr = (uint*)ptr;
                        uint val = *uiptr;

                        WriteEntryHeader(val, stream, mValue.Length / mFieldType.SizeInBytes);
                        mOffsetInStream = 0;
                    }
                }
            }
            else
            {
                mOffsetInStream = WriteEntryHeader(0, stream, mValue.Length / mFieldType.SizeInBytes);
            }
        }
        public override void SavePass2(Stream stream)
        {
            if (mOffsetInStream != 0)
            {
                WritePass2(mValue, stream);
            }
        }
    }

	public class SByteExifEntry : ExifEntry
	{
		protected sbyte[] mValue;
        public SByteExifEntry(sbyte[] aValue, ushort aTagID)
            : base(aTagID, new TIFFValueType(TIFFValueTypes.SignedByte), (uint)aValue.Length)
        {
            mValue = aValue;
        }

        public SByteExifEntry(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			if (mFieldType.SizeInBytes * mValueCount <= 4)
			{
				unsafe
				{
					fixed (uint* ptr = &mOffset)
					{
						sbyte* ptrUS = (sbyte*)ptr;
						mValue = new sbyte[mValueCount];
						for (int i = 0; i < mValueCount; i++)
						{
							if (aPefFile.EndianSwap)
								mValue[i] = ptrUS[4 / mFieldType.SizeInBytes - i - 1];
							else
								mValue[i] = ptrUS[i];
						}
					}
				}
			}
			else
			{
				uint currentOffset = mPefFile.Position();
				mPefFile.Seek(mOffset, System.IO.SeekOrigin.Begin);

				mValue = new sbyte[mValueCount];
				for (int i = 0; i < mValueCount; i++)
				{
					mValue[i] = mPefFile.ReadI1();
				}
				mPefFile.Seek(currentOffset, System.IO.SeekOrigin.Begin);
			}
        }

        public override void SavePass1(Stream stream)
        {
            if (mValue.Length <= 4)
            {
                sbyte[] valueToWrite = mValue;
                if (mValue.Length < 4)
                {
                    valueToWrite = new sbyte[4];
                    for (int i = 0; i < mValue.Length; i++)
                    {
                        valueToWrite[i] = mValue[i];
                    }
                }
                unsafe
                {
                    fixed (sbyte* ptr = valueToWrite)
                    {
                        uint* uiptr = (uint*)ptr;
                        uint val = *uiptr;

                        WriteEntryHeader(val, stream, mValue.Length);
                        mOffsetInStream = 0;
                    }
                }
            }
            else
            {
                mOffsetInStream = WriteEntryHeader(0, stream, mValue.Length);
            }
        }
        public override void SavePass2(Stream stream)
        {
            if (mOffsetInStream != 0)
            {
                byte[] valueToWrite = new byte[mValue.Length];
                unsafe
                {
                    fixed (sbyte* ptr = mValue)
                    {
                        Marshal.Copy((IntPtr)ptr, valueToWrite, 0, mValue.Length);
                    }
                }
                WritePass2(valueToWrite, stream);
            }
        }
    }

	public class UShortExifEntry : ExifEntry
	{
		protected ushort[] mValue;
        public UShortExifEntry(ushort[] aValue, ushort aTagID)
            : base(aTagID, new TIFFValueType(TIFFValueTypes.UnsignedShort), (uint)aValue.Length)
        {
            mValue = aValue;
        }


        public UShortExifEntry(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			if (mFieldType.SizeInBytes * mValueCount <= 4)
			{
				unsafe
				{
					fixed (uint* ptr = &mOffset)
					{
						ushort* ptrUS = (ushort*)ptr;
						mValue = new ushort[mValueCount];
						for (int i = 0; i < mValueCount; i++)
						{
							if (aPefFile.EndianSwap)
								mValue[i] = ptrUS[4 / mFieldType.SizeInBytes - i - 1];
							else
								mValue[i] = ptrUS[i];
						}
					}
				}
			}
			else
			{
				uint currentOffset = mPefFile.Position();
				mPefFile.Seek(mOffset, System.IO.SeekOrigin.Begin);

				mValue = new ushort[mValueCount];
				for (int i = 0; i < mValueCount; i++)
				{
					mValue[i] = mPefFile.ReadUI2();
				}
				mPefFile.Seek(currentOffset, System.IO.SeekOrigin.Begin);
			}
        }

        public override void SavePass1(Stream stream)
        {
            if (mValue.Length <= 2)
            {
                ushort[] valueToWrite = mValue;
                if (mValue.Length < 2)
                {
                    valueToWrite = new ushort[2];
                    for (int i = 0; i < mValue.Length; i++)
                    {
                        valueToWrite[i] = mValue[i];
                    }
                }
                unsafe
                {
                    fixed (ushort* ptr = valueToWrite)
                    {
                        uint* uiptr = (uint*)ptr;
                        uint val = *uiptr;

                        WriteEntryHeader(val, stream, mValue.Length);
                        mOffsetInStream = 0;
                    }
                }
            }
            else
            {
                mOffsetInStream = WriteEntryHeader(0, stream, mValue.Length);
            }
        }
        public override void SavePass2(Stream stream)
        {
            if (mOffsetInStream != 0)
            {
                byte[] valueToWrite = new byte[2 * mValue.Length];
                unsafe
                {
                    fixed (ushort* ptr = mValue)
                    {
                        Marshal.Copy((IntPtr)ptr, valueToWrite, 0, valueToWrite.Length);
                    }
                }
                WritePass2(valueToWrite, stream);
            }
        }
    }

	public class ShortExifEntry : ExifEntry
	{
		protected short[] mValue;
        public ShortExifEntry(short[] aValue, ushort aTagID)
            : base(aTagID, new TIFFValueType(TIFFValueTypes.SignedShort), (uint)aValue.Length)
        {
            mValue = aValue;
        }

        public ShortExifEntry(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			if (mFieldType.SizeInBytes * mValueCount <= 4)
			{
				unsafe
				{
					fixed (uint* ptr = &mOffset)
					{
						short* ptrUS = (short*)ptr;
						mValue = new short[mValueCount];
						for (int i = 0; i < mValueCount; i++)
						{
							if (aPefFile.EndianSwap)
								mValue[i] = ptrUS[4 / mFieldType.SizeInBytes - i - 1];
							else
								mValue[i] = ptrUS[i];
						}
					}
				}
			}
			else
			{
				uint currentOffset = mPefFile.Position();
				mPefFile.Seek(mOffset, System.IO.SeekOrigin.Begin);

				mValue = new short[mValueCount];
				for (int i = 0; i < mValueCount; i++)
				{
					mValue[i] = mPefFile.ReadI2();
				}
				mPefFile.Seek(currentOffset, System.IO.SeekOrigin.Begin);
			}
        }

        public override void SavePass1(Stream stream)
        {
            if (mValue.Length <= 2)
            {
                short[] valueToWrite = mValue;
                if (mValue.Length < 2)
                {
                    valueToWrite = new short[2];
                    for (int i = 0; i < mValue.Length; i++)
                    {
                        valueToWrite[i] = mValue[i];
                    }
                }
                unsafe
                {
                    fixed (short* ptr = valueToWrite)
                    {
                        uint* uiptr = (uint*)ptr;
                        uint val = *uiptr;

                        WriteEntryHeader(val, stream, mValue.Length);
                        mOffsetInStream = 0;
                    }
                }
            }
            else
            {
                mOffsetInStream = WriteEntryHeader(0, stream, mValue.Length);
            }
        }
        public override void SavePass2(Stream stream)
        {
            if (mOffsetInStream != 0)
            {
                byte[] valueToWrite = new byte[2 * mValue.Length];
                unsafe
                {
                    fixed (short* ptr = mValue)
                    {
                        Marshal.Copy((IntPtr)ptr, valueToWrite, 0, valueToWrite.Length);
                    }
                }
                WritePass2(valueToWrite, stream);
            }
        }
    }

	public class IntExifEntry : ExifEntry
	{
		protected int[] mValue;
        public IntExifEntry(int[] aValue, ushort aTagID)
            : base(aTagID, new TIFFValueType(TIFFValueTypes.SignedLong), (uint)aValue.Length)
        {
            mValue = aValue;
        }

        public IntExifEntry(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			if (mFieldType.SizeInBytes * mValueCount <= 4)
			{
				unsafe
				{
					fixed (uint* ptr = &mOffset)
					{
						int* ptrUS = (int*)ptr;
						mValue = new int[mValueCount];
						for (int i = 0; i < mValueCount; i++)
						{
							if (aPefFile.EndianSwap)
								mValue[i] = ptrUS[4 / mFieldType.SizeInBytes - i - 1];
							else
								mValue[i] = ptrUS[i];
						}
					}
				}
			}
			else
			{
				uint currentOffset = mPefFile.Position();
				mPefFile.Seek(mOffset, System.IO.SeekOrigin.Begin);

				mValue = new int[mValueCount];
				for (int i = 0; i < mValueCount; i++)
				{
					mValue[i] = mPefFile.ReadI4();
				}
				mPefFile.Seek(currentOffset, System.IO.SeekOrigin.Begin);
			}
        }

        public override void SavePass1(Stream stream)
        {
            if (mValue.Length <= 1)
            {
                unsafe
                {
                    fixed (int* ptr = mValue)
                    {
                        uint* uiptr = (uint*)ptr;
                        uint val = *uiptr;

                        WriteEntryHeader(val, stream, mValue.Length);
                        mOffsetInStream = 0;
                    }
                }
            }
            else
            {
                mOffsetInStream = WriteEntryHeader(0, stream, mValue.Length);
            }
        }
        public override void SavePass2(Stream stream)
        {
            if (mOffsetInStream != 0)
            {
                byte[] valueToWrite = new byte[4 * mValue.Length];
                unsafe
                {
                    fixed (int* ptr = mValue)
                    {
                        Marshal.Copy((IntPtr)ptr, valueToWrite, 0, valueToWrite.Length);
                    }
                }
                WritePass2(valueToWrite, stream);
            }
        }
    }

	public class UIntExifEntry : ExifEntry
	{
		protected uint[] mValue;
        public UIntExifEntry(uint[] aValue, ushort aTagID)
            : base(aTagID, new TIFFValueType(TIFFValueTypes.UnsignedLong), (uint)aValue.Length)
        {
            mValue = aValue;
        }


        public UIntExifEntry(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			if (mFieldType.SizeInBytes * mValueCount <= 4)
			{
				mValue = new uint[mValueCount];
				mValue[0] = mOffset;
			}
			else
			{
				uint currentOffset = mPefFile.Position();
				mPefFile.Seek(mOffset, System.IO.SeekOrigin.Begin);

				mValue = new uint[mValueCount];
				for (int i = 0; i < mValueCount; i++)
				{
					mValue[i] = mPefFile.ReadUI4();
				}
				mPefFile.Seek(currentOffset, System.IO.SeekOrigin.Begin);
			}
        }

        public override void SavePass1(Stream stream)
        {
            if (mValue.Length <= 1)
            {
                unsafe
                {
                    fixed (uint* ptr = mValue)
                    {
                        uint* uiptr = (uint*)ptr;
                        uint val = *uiptr;

                        WriteEntryHeader(val, stream, mValue.Length);
                        mOffsetInStream = 0;
                    }
                }
            }
            else
            {
                mOffsetInStream = WriteEntryHeader(0, stream, mValue.Length);
            }
        }
        public override void SavePass2(Stream stream)
        {
            if (mOffsetInStream != 0)
            {
                byte[] valueToWrite = new byte[4 * mValue.Length];
                unsafe
                {
                    fixed (uint* ptr = mValue)
                    {
                        Marshal.Copy((IntPtr)ptr, valueToWrite, 0, valueToWrite.Length);
                    }
                }
                WritePass2(valueToWrite, stream);
            }
        }
    }

	public class RationalExifEntry : ExifEntry
	{
		protected Rational[] mValue;
        public RationalExifEntry(Rational[] aValue, ushort aTagID)
            : base(aTagID, new TIFFValueType(TIFFValueTypes.Rational), (uint)aValue.Length)
        {
            mValue = aValue;
        }

        public RationalExifEntry(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			uint currentOffset = mPefFile.Position();
			mPefFile.Seek(mOffset, System.IO.SeekOrigin.Begin);

			mValue = new Rational[mValueCount];
			for (int i = 0; i < mValueCount; i++)
			{
				uint tempNom = mPefFile.ReadUI4();
				uint tempDenom = mPefFile.ReadUI4();
				mValue[i] = new Rational(tempNom, tempDenom);
			}
			mPefFile.Seek(currentOffset, System.IO.SeekOrigin.Begin);
        }

        public override void SavePass1(Stream stream)
        {
            mOffsetInStream = WriteEntryHeader(0, stream, mValue.Length);
        }
        public override void SavePass2(Stream stream)
        {
            if (mOffsetInStream != 0)
            {
                byte[] valueToWrite = new byte[8 * mValue.Length];

                for (int i = 0; i < mValue.Length; i++)
                {
                    int index = i * 8;
                    byte[] nom = BitConverter.GetBytes(mValue[i].numerator);
                    for (int j = 0; j < 4; j++)
                    {
                        valueToWrite[index + j] = nom[j];
                    }
                    byte[] den = BitConverter.GetBytes(mValue[i].denominator);
                    for (int j = 0; j < 4; j++)
                    {
                        valueToWrite[index + j+4] = den[j];
                    }
                }
                WritePass2(valueToWrite, stream);
            }
        }
    }

	public class SRationalExifEntry : ExifEntry
	{
		protected SRational[] mValue;
        public SRationalExifEntry(SRational[] aValue, ushort aTagID)
            : base(aTagID, new TIFFValueType(TIFFValueTypes.SignedRational), (uint)aValue.Length)
        {
            mValue = aValue;
        }

        public SRationalExifEntry(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			uint currentOffset = mPefFile.Position();
			mPefFile.Seek(mOffset, System.IO.SeekOrigin.Begin);

			mValue = new SRational[mValueCount];
			for (int i = 0; i < mValueCount; i++)
			{
				int tempNom = mPefFile.ReadI4();
				int tempDenom = mPefFile.ReadI4();
				mValue[i] = new SRational(tempNom, tempDenom);
			}
			mPefFile.Seek(currentOffset, System.IO.SeekOrigin.Begin);
        }

        public override void SavePass1(Stream stream)
        {
            mOffsetInStream = WriteEntryHeader(0, stream, mValue.Length);
        }
        public override void SavePass2(Stream stream)
        {
            if (mOffsetInStream != 0)
            {
                byte[] valueToWrite = new byte[8 * mValue.Length];

                for (int i = 0; i < mValue.Length; i++)
                {
                    int index = i * 8;
                    byte[] nom = BitConverter.GetBytes(mValue[i].numerator);
                    for (int j = 0; j < 4; j++)
                    {
                        valueToWrite[index + j] = nom[j];
                    }
                    byte[] den = BitConverter.GetBytes(mValue[i].denominator);
                    for (int j = 0; j < 4; j++)
                    {
                        valueToWrite[index + j+4] = den[j];
                    }
                }
                WritePass2(valueToWrite, stream);
            }
        }
    }

	public class FloatExifEntry : ExifEntry
	{
		protected float[] mValue;
        public FloatExifEntry(float[] aValue, ushort aTagID)
            : base(aTagID, new TIFFValueType(TIFFValueTypes.Float), (uint)aValue.Length)
        {
            mValue = aValue;
        }


        public FloatExifEntry(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			if (mFieldType.SizeInBytes * mValueCount <= 4)
			{
				unsafe
				{
					fixed (uint* ptr = &mOffset)
					{
						float* ptrUS = (float*)ptr;
						mValue = new float[mValueCount];
						for (int i = 0; i < mValueCount; i++)
						{
							if (aPefFile.EndianSwap)
								mValue[i] = ptrUS[4 / mFieldType.SizeInBytes - i - 1];
							else
								mValue[i] = ptrUS[i];
						}
					}
				}
			}
			else
			{
				uint currentOffset = mPefFile.Position();
				mPefFile.Seek(mOffset, System.IO.SeekOrigin.Begin);

				mValue = new float[mValueCount];
				for (int i = 0; i < mValueCount; i++)
				{
					mValue[i] = mPefFile.ReadF4();
				}
				mPefFile.Seek(currentOffset, System.IO.SeekOrigin.Begin);
			}
        }

        public override void SavePass1(Stream stream)
        {
            if (mValue.Length <= 1)
            {
                unsafe
                {
                    fixed (float* ptr = mValue)
                    {
                        uint* uiptr = (uint*)ptr;
                        uint val = *uiptr;

                        WriteEntryHeader(val, stream, mValue.Length);
                        mOffsetInStream = 0;
                    }
                }
            }
            else
            {
                mOffsetInStream = WriteEntryHeader(0, stream, mValue.Length);
            }
        }
        public override void SavePass2(Stream stream)
        {
            if (mOffsetInStream != 0)
            {
                byte[] valueToWrite = new byte[4 * mValue.Length];
                unsafe
                {
                    fixed (float* ptr = mValue)
                    {
                        Marshal.Copy((IntPtr)ptr, valueToWrite, 0, valueToWrite.Length);
                    }
                }
                WritePass2(valueToWrite, stream);
            }
        }
    }

	public class DoubleExifEntry : ExifEntry
	{
		protected double[] mValue;
        public DoubleExifEntry(double[] aValue, ushort aTagID)
            : base(aTagID, new TIFFValueType(TIFFValueTypes.Double), (uint)aValue.Length)
        {
            mValue = aValue;
        }

        public DoubleExifEntry(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			uint currentOffset = mPefFile.Position();
			mPefFile.Seek(mOffset, System.IO.SeekOrigin.Begin);

			mValue = new double[mValueCount];
			for (int i = 0; i < mValueCount; i++)
			{
				mValue[i] = mPefFile.ReadF8();
			}
			mPefFile.Seek(currentOffset, System.IO.SeekOrigin.Begin);
        }

        public override void SavePass1(Stream stream)
        {
            mOffsetInStream = WriteEntryHeader(0, stream, mValue.Length);
        }
        public override void SavePass2(Stream stream)
        {
            if (mOffsetInStream != 0)
            {
                byte[] valueToWrite = new byte[8 * mValue.Length];
                unsafe
                {
                    fixed (double* ptr = mValue)
                    {
                        Marshal.Copy((IntPtr)ptr, valueToWrite, 0, valueToWrite.Length);
                    }
                }
                WritePass2(valueToWrite, stream);
            }
        }
    }
    #endregion

    #region Exif Tags
    public class ExifUnknownTag : ByteExifEntry
    {
        public ExifUnknownTag(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        {

        }

        public ExifUnknownTag(byte[] aValue, ushort aTagID, TIFFValueType aValueType)
            : base(aValue, aTagID)
        {
            mFieldType = aValueType;
        }

        public override string ToString()
        {
            return "Unknown EXIF entry. ID: " + mTagID;
        }
    }

    public class ExifExposureTime : RationalExifEntry
	{
		public ExifExposureTime(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }
        public ExifExposureTime(Rational aValue)
            : base(new Rational[] { aValue }, TagID)
        {

        }

        public const ushort TagID = 33434;

		public const string TagName = "Exposure time";

		public Rational Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString() + " sec.";
		}
	}

	public class ExifFNumber : RationalExifEntry
	{
		public ExifFNumber(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }
        public ExifFNumber(Rational aValue)
            : base(new Rational[] { aValue }, TagID)
        {

        }

        public const ushort TagID = 33437;

		public const string TagName = "F number";

		public Rational Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class ExifExposureProgram : UShortExifEntry
	{
		public enum ExposureProgram
		{ 
			NotDefined = 0,
			Manual = 1,
			NormalProgram = 2,
			AperturePriority = 3,
			ShutterPriority = 4,
			CreativeProgram = 5,
			ActionProgram = 6,
			PortraitMode = 7,
			LandscapeMode = 8
		}

		public ExifExposureProgram(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }
        public ExifExposureProgram(ExposureProgram aValue)
            : base(new ushort[] { (ushort)aValue }, TagID)
        {

        }

        public const ushort TagID = 34850;

		public const string TagName = "Exposure program";

		public ExposureProgram Value
		{
			get { return (ExposureProgram)mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class ExifSpectralSensitivity : StringExifEntry
	{
		public ExifSpectralSensitivity(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }
        public ExifSpectralSensitivity(string aValue)
            : base(aValue, TagID)
        {

        }

        public const ushort TagID = 34852;

		public const string TagName = "Spectral sensitivity";

		public string Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value;
		}
	}

	public class ExifISOSpeedRatings : UShortExifEntry
	{
		public ExifISOSpeedRatings(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }
        public ExifISOSpeedRatings(ushort aValue)
            : base(new ushort[] { aValue }, TagID)
        {

        }

        public const ushort TagID = 34855;

		public const string TagName = "ISO speed ratings";

		public ushort Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class ExifOECF : ByteExifEntry
	{
		public ExifOECF(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }

        public ExifOECF(byte[] aValue)
            : base(aValue, TagID)
        {

        }

        public const ushort TagID = 34856;

		public const string TagName = "Opto-Electric Conversion Function (OECF)";

		public byte[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value[0].ToString();
		}
	}

	public class ExifExifVersion : StringExifEntry
	{
		public ExifExifVersion(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }
        public ExifExifVersion(string aValue)
            : base(aValue, TagID)
        {

        }

        public const ushort TagID = 36864;

		public const string TagName = "Exif version";

		public string Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value;
		}
	}

	public class ExifDateTimeOriginal : StringExifEntry
	{
		DateTime dt_value;

		public ExifDateTimeOriginal(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			int year = int.Parse(mValue.Substring(0, 4));
			int month = int.Parse(mValue.Substring(5, 2));
			int day = int.Parse(mValue.Substring(8, 2));
			int hour = int.Parse(mValue.Substring(11, 2));
			int min = int.Parse(mValue.Substring(14, 2));
			int sec = int.Parse(mValue.Substring(17, 2));
			dt_value = new DateTime(year, month, day, hour, min, sec);
        }
        public ExifDateTimeOriginal(DateTime aValue)
            : base(string.Empty, TagID)
        {
            mValue = "";
            mValue += aValue.Year.ToString("0000");
            mValue += ":";
            mValue += aValue.Month.ToString("00");
            mValue += ":";
            mValue += aValue.Day.ToString("00");
            mValue += " ";
            mValue += aValue.Hour.ToString("00");
            mValue += ":";
            mValue += aValue.Minute.ToString("00");
            mValue += ":";
            mValue += aValue.Second.ToString("00");
            mValueCount = 20; //+trailing zero
            dt_value = aValue;
        }

        public const ushort TagID = 36867;

		public const string TagName = "Date/Time original";

		public DateTime Value
		{
			get { return dt_value; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class ExifDateTimeDigitized : StringExifEntry
	{
		DateTime dt_value;

		public ExifDateTimeDigitized(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			int year = int.Parse(mValue.Substring(0, 4));
			int month = int.Parse(mValue.Substring(5, 2));
			int day = int.Parse(mValue.Substring(8, 2));
			int hour = int.Parse(mValue.Substring(11, 2));
			int min = int.Parse(mValue.Substring(14, 2));
			int sec = int.Parse(mValue.Substring(17, 2));
			dt_value = new DateTime(year, month, day, hour, min, sec);
        }
        public ExifDateTimeDigitized(DateTime aValue)
            : base(string.Empty, TagID)
        {
            mValue = "";
            mValue += aValue.Year.ToString("0000");
            mValue += ":";
            mValue += aValue.Month.ToString("00");
            mValue += ":";
            mValue += aValue.Day.ToString("00");
            mValue += " ";
            mValue += aValue.Hour.ToString("00");
            mValue += ":";
            mValue += aValue.Minute.ToString("00");
            mValue += ":";
            mValue += aValue.Second.ToString("00");
            mValueCount = 20; //+trailing zero
            dt_value = aValue;
        }

        public const ushort TagID = 36868;

		public const string TagName = "Date/Time digitized";

		public DateTime Value
		{
			get { return dt_value; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class ExifComponentsConfiguration : ByteExifEntry
	{
		public ExifComponentsConfiguration(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }

        public ExifComponentsConfiguration(byte[] aValue)
            : base(aValue, TagID)
        {

        }

        public const ushort TagID = 37121;

		public const string TagName = "Components configuration";

		public byte[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class ExifCompressedBitsPerPixel : RationalExifEntry
	{
		public ExifCompressedBitsPerPixel(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }
        public ExifCompressedBitsPerPixel(Rational aValue)
            : base(new Rational[] { aValue }, TagID)
        {

        }

        public const ushort TagID = 37122;

		public const string TagName = "Compressed bits per pixel";

		public Rational Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class ExifShutterSpeedValue : SRationalExifEntry
	{
		public ExifShutterSpeedValue(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }
        public ExifShutterSpeedValue(SRational aValue)
            : base(new SRational[] { aValue }, TagID)
        {

        }

        public const ushort TagID = 37377;

		public const string TagName = "Shutter speed value";

		public SRational Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class ExifApertureValue : RationalExifEntry
	{
		public ExifApertureValue(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }
        public ExifApertureValue(Rational aValue)
            : base(new Rational[] { aValue }, TagID)
        {

        }

        public const ushort TagID = 37378;

		public const string TagName = "Aperture value";

		public Rational Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class ExifBrightnessValue : SRationalExifEntry
	{
		public ExifBrightnessValue(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }
        public ExifBrightnessValue(SRational aValue)
            : base(new SRational[] { aValue }, TagID)
        {

        }

        public const ushort TagID = 37379;

		public const string TagName = "Brightness value";

		public SRational Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class ExifExposureBiasValue : SRationalExifEntry
	{
		public ExifExposureBiasValue(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }
        public ExifExposureBiasValue(SRational aValue)
            : base(new SRational[] { aValue }, TagID)
        {

        }

        public const ushort TagID = 37380;

		public const string TagName = "Exposure bias value";

		public SRational Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class ExifMaxApertureValue : RationalExifEntry
	{
		public ExifMaxApertureValue(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }
        public ExifMaxApertureValue(Rational aValue)
            : base(new Rational[] { aValue }, TagID)
        {

        }

        public const ushort TagID = 37381;

		public const string TagName = "Max. aperture value";

		public Rational Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class ExifSubjectDistance : RationalExifEntry
	{
		public ExifSubjectDistance(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }
        public ExifSubjectDistance(Rational aValue)
            : base(new Rational[] { aValue }, TagID)
        {

        }

        public const ushort TagID = 37382;

		public const string TagName = "Subject distance";

		public Rational Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class ExifMeteringMode : UShortExifEntry
	{
		public enum MeteringMode : ushort
		{ 
			Unknown = 0,
			Average = 1,
			CenterWEightedAverage = 2,
			Spot = 3,
			MultiSpot = 4,
			Pattern = 5,
			Partial = 6,
			Other = 255
		}

		public ExifMeteringMode(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }
        public ExifMeteringMode(MeteringMode aValue)
            : base(new ushort[] { (ushort)aValue }, TagID)
        {

        }

        public const ushort TagID = 37383;

		public const string TagName = "Metering mode";

		public MeteringMode Value
		{
			get { return (MeteringMode)mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class ExifLightSource : UShortExifEntry
	{
		public enum LightSource : ushort
		{ 
			Unknown = 0,
			Daylight = 1,
			Fluorescent = 2,
			Tungsten = 3,
			Flash = 4,
			FineWeather = 9,
			CloudyWeather = 10,
			Shade = 11,
			DaylightFluorescent = 12,
			DayWhiteFluorescent = 13,
			CoolWhiteFluorescent = 14,
			WhiteFluorescent = 15,
			StandardLightA = 17,
			StandardLightB = 18,
			StandardLightC = 19,
			D55 = 20,
			D65 = 21,
			D75 = 22,
			D50 = 23,
			ISOStudioTungsten = 24,
			Other = 255
		}

		public ExifLightSource(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }
        public ExifLightSource(LightSource aValue)
            : base(new ushort[] { (ushort)aValue }, TagID)
        {

        }

        public const ushort TagID = 37384;

		public const string TagName = "Light source";

		public LightSource Value
		{
			get { return (LightSource)mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class ExifFlash : UShortExifEntry
	{
		string str_value;

		public ExifFlash(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			switch (mValue[0])
			{
				case 0x0000:
					str_value = "Flash did not fire";
					break;
				case 0x0001:
					str_value = "Flash fired";
					break;
				case 0x0005:
					str_value = "Strobe return light not detected";
					break;
				case 0x0007:
					str_value = "Strobe return light detected";
					break;
				case 0x0009:
					str_value = "Flash fired, compulsory flash mode";
					break;
				case 0x000D:
					str_value = "Flash fired, compulsory flash mode, return light not detected";
					break;
				case 0x000F:
					str_value = "Flash fired, compulsory flash mode, return light detected";
					break;
				case 0x0010:
					str_value = "Flash did not fire, compulsory flash mode";
					break;
				case 0x0018:
					str_value = "Flash did not fire, auto mode";
					break;
				case 0x0019:
					str_value = "Flash fired, auto mode";
					break;
				case 0x001D:
					str_value = "Flash fired, auto mode, return light not detected";
					break;
				case 0x001F:
					str_value = "Flash fired, auto mode, return light detected";
					break;
				case 0x0020:
					str_value = "No flash function";
					break;
				case 0x0041:
					str_value = "Flash fired, red-eye reduction mode";
					break;
				case 0x0045:
					str_value = "Flash fired, red-eye reduction mode, return light not detected";
					break;
				case 0x0047:
					str_value = "Flash fired, red-eye reduction mode, return light detected";
					break;
				case 0x0049:
					str_value = "Flash fired, compulsory flash mode, red-eye reduction mode";
					break;
				case 0x004D:
					str_value = "Flash fired, compulsory flash mode, red-eye reduction mode, return light not detected";
					break;
				case 0x004F:
					str_value = "Flash fired, compulsory flash mode, red-eye reduction mode, return light detected";
					break;
				case 0x0059:
					str_value = "Flash fired, auto mode, red-eye reduction mode";
					break;
				case 0x005D:
					str_value = "Flash fired, auto mode, return light not detected, red-eye reduction mode";
					break;
				case 0x005F:
					str_value = "Flash fired, auto mode, return light detected, red-eye reduction mode";
					break; 
				default:
					str_value = "Flash unknown - value is: 0x" + mValue[0].ToString("X4");
					break;
			}
        }
        public ExifFlash(ushort aValue)
            : base(new ushort[] { aValue }, TagID)
        {

        }

        public const ushort TagID = 37385;

		public const string TagName = "Flash";

		public string Value
		{
			get { return str_value; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value;
		}
	}

	public class ExifFocalLength : RationalExifEntry
	{
		public ExifFocalLength(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }
        public ExifFocalLength(Rational aValue)
            : base(new Rational[] { aValue }, TagID)
        {

        }

        public const ushort TagID = 37386;

		public const string TagName = "Focal length";

		public Rational Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class ExifSubjectArea : UShortExifEntry
	{
		public ExifSubjectArea(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }
        public ExifSubjectArea(ushort[] aValue)
            : base(aValue, TagID)
        {

        }

        public const ushort TagID = 37396;

		public const string TagName = "Subject area";

		public ushort[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			switch (Value.Length)
			{
				case 2:
					return TagName + ": X: " + Value[0].ToString() + " Y: " + Value[1].ToString();
				case 3:
					return TagName + ": X: " + Value[0].ToString() + " Y: " + Value[1].ToString() + " Radius: " + Value[2].ToString();
				case 4:
					return TagName + ": X: " + Value[0].ToString() + " Y: " + Value[1].ToString() + " Width: " + Value[2].ToString() + " Height: " + Value[3].ToString();
				default:
					break;
			}

			return TagName + ": " + Value.Length.ToString() + " entries";
		}
	}

	public class ExifMakerNote : ByteExifEntry
	{
		PentaxMakerNotes mMakerNotes;

		public ExifMakerNote(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			mMakerNotes = new PentaxMakerNotes(mValue, mOffset);
		}

		public const ushort TagID = 37500;

		public const string TagName = "Maker note";

		public PentaxMakerNotes Value
		{
			get { return mMakerNotes; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class ExifUserComment : StringExifEntry
	{
		public ExifUserComment(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }
        public ExifUserComment(string aValue)
            : base(aValue, TagID)
        {

        }

        public const ushort TagID = 37510;

		public const string TagName = "User comment";

		public string Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value;
		}
	}

	public class ExifSubsecTime : StringExifEntry
	{
		public ExifSubsecTime(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }
        public ExifSubsecTime(string aValue)
            : base(aValue, TagID)
        {

        }

        public const ushort TagID = 37520;

		public const string TagName = "Subsec time";

		public string Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value;
		}
	}

	public class ExifSubsecTimeOriginal : StringExifEntry
	{
		public ExifSubsecTimeOriginal(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }
        public ExifSubsecTimeOriginal(string aValue)
            : base(aValue, TagID)
        {

        }

        public const ushort TagID = 37521;

		public const string TagName = "Subsec time original";

		public string Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value;
		}
	}

	public class ExifSubsecTimeDigitized : StringExifEntry
	{
		public ExifSubsecTimeDigitized(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }
        public ExifSubsecTimeDigitized(string aValue)
            : base(aValue, TagID)
        {

        }

        public const ushort TagID = 37522;

		public const string TagName = "Subsec time digitized";

		public string Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value;
		}
	}

	public class ExifFlashpixVersion : StringExifEntry
	{
		public ExifFlashpixVersion(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }
        public ExifFlashpixVersion(string aValue)
            : base(aValue, TagID)
        {

        }

        public const ushort TagID = 40960;

		public const string TagName = "Flashpix version";

		public string Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value;
		}
	}

	public class ExifColorSpace : UShortExifEntry
	{
		public enum ColorSpace : ushort
		{
			Unknown = 0,
			sRGB = 1,
			Uncalibrated = 65535
		}

		public ExifColorSpace(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }
        public ExifColorSpace(ColorSpace aValue)
            : base(new ushort[] { (ushort)aValue }, TagID)
        {

        }

        public const ushort TagID = 40961;

		public const string TagName = "Color space";

		public ColorSpace Value
		{
			get { return (ColorSpace)mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class ExifPixelXDimension : ExifEntry
	{
		uint mValue;

		public ExifPixelXDimension(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			if (mFieldType.SizeInBytes == 2 && aPefFile.EndianSwap)
			{
				unsafe
				{
					fixed (uint* ptr = &mOffset)
					{
						ushort* ptrUS = (ushort*)ptr;
						mValue = ptrUS[4 / mFieldType.SizeInBytes - 1];
					}
				}
			}
			else
				mValue = mOffset;
		}

		public const ushort TagID = 40962;

		public const string TagName = "Pixel X-dimension";

		public uint Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue.ToString();
		}
	}

	public class ExifPixelYDimension : ExifEntry
	{
		uint mValue;

		public ExifPixelYDimension(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			if (mFieldType.SizeInBytes == 2 && aPefFile.EndianSwap)
			{
				unsafe
				{
					fixed (uint* ptr = &mOffset)
					{
						ushort* ptrUS = (ushort*)ptr;
						mValue = ptrUS[4 / mFieldType.SizeInBytes - 1];
					}
				}
			}
			else
				mValue = mOffset;
		}

		public const ushort TagID = 40963;

		public const string TagName = "Pixel Y-dimension";

		public uint Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue.ToString();
		}
	}

	public class ExifRelatedSoundFile : StringExifEntry
	{
		public ExifRelatedSoundFile(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }
        public ExifRelatedSoundFile(string aValue)
            : base(aValue, TagID)
        {

        }

        public const ushort TagID = 40964;

		public const string TagName = "Related sound file";

		public string Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value;
		}
	}

	public class ExifFlashEnergy : RationalExifEntry
	{
		public ExifFlashEnergy(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }
        public ExifFlashEnergy(Rational aValue)
            : base(new Rational[] { aValue }, TagID)
        {

        }

        public const ushort TagID = 41483;

		public const string TagName = "Flash energy";

		public Rational Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class ExifSpatialFrequencyResponse : ByteExifEntry
	{
		public ExifSpatialFrequencyResponse(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }

        public ExifSpatialFrequencyResponse(byte[] aValue)
            : base(aValue, TagID)
        {

        }

        public const ushort TagID = 41484;

		public const string TagName = "Spatial frequency response";

		public Byte[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.Length.ToString() + " entries";
		}
	}

	public class ExifFocalPlaneXResolution : RationalExifEntry
	{
		public ExifFocalPlaneXResolution(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }
        public ExifFocalPlaneXResolution(Rational aValue)
            : base(new Rational[] { aValue }, TagID)
        {

        }

        public const ushort TagID = 41486;

		public const string TagName = "Focal plane X-resolution";

		public Rational Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class ExifFocalPlaneYResolution : RationalExifEntry
	{
		public ExifFocalPlaneYResolution(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }
        public ExifFocalPlaneYResolution(Rational aValue)
            : base(new Rational[] { aValue }, TagID)
        {

        }

        public const ushort TagID = 41487;

		public const string TagName = "Focal plane Y-resolution";

		public Rational Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class ExifFocalPlaneResolutionUnit : UShortExifEntry
	{
		public enum FocalPlaneResolutionUnit : ushort
		{
			NoAbsoluteUnitOfMeasurement = 1,
			Inch = 2,
			Centimeter = 3
		}

		public ExifFocalPlaneResolutionUnit(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }
        public ExifFocalPlaneResolutionUnit(FocalPlaneResolutionUnit aValue)
            : base(new ushort[] { (ushort)aValue }, TagID)
        {

        }

        public const ushort TagID = 41488;

		public const string TagName = "Focal plane resolution unit";

		public FocalPlaneResolutionUnit Value
		{
			get { return (FocalPlaneResolutionUnit)mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class ExifSubjectLocation : UShortExifEntry
	{
		public ExifSubjectLocation(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }
        public ExifSubjectLocation(ushort[] aValue)
            : base(aValue, TagID)
        {

        }

        public const ushort TagID = 41492;

		public const string TagName = "Subject location";

		public ushort[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": X: " + Value[0].ToString() + " Y: " + Value[1].ToString();
		}
	}

	public class ExifExposureIndex : RationalExifEntry
	{
		public ExifExposureIndex(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }
        public ExifExposureIndex(Rational aValue)
            : base(new Rational[] { aValue }, TagID)
        {

        }

        public const ushort TagID = 41493;

		public const string TagName = "Exposure index";

		public Rational Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class ExifSensingMethod : UShortExifEntry
	{
		public enum SensingMethod : ushort
		{
			NotDefined = 1,
			OneChipColorAreaSensor = 2,
			TwoChipColorAreaSensor = 3,
			ThreeChipColorAreaSensor = 4,
			ColorSequentialAreaSensor = 5,
			TrilinearSensor = 7,
			ColorSequentialLinearSensor = 8
		}

		public ExifSensingMethod(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }
        public ExifSensingMethod(SensingMethod aValue)
            : base(new ushort[] { (ushort)aValue }, TagID)
        {

        }

        public const ushort TagID = 41495;

		public const string TagName = "Sensing method";

		public SensingMethod Value
		{
			get { return (SensingMethod)mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class ExifFileSource : ByteExifEntry
	{
		public enum FileSource : byte
		{
			DigitalStillCamera = 3
		}

		public ExifFileSource(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }

        public ExifFileSource(FileSource aValue)
            : base(new byte[] { (byte)aValue }, TagID)
        {
        }

        public const ushort TagID = 41728;

		public const string TagName = "File source";

		public FileSource Value
		{
			get { return (FileSource)mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class ExifSceneType : ByteExifEntry
	{
		public enum SceneType : byte
		{
			DirectlyPhotographedImage = 1
		}

		public ExifSceneType(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }

        public ExifSceneType(SceneType aValue)
            : base(new byte[] { (byte)aValue }, TagID)
        {
        }

        public const ushort TagID = 41729;

		public const string TagName = "Scene type";

		public SceneType Value
		{
			get { return (SceneType)mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class ExifCFAPattern : ByteExifEntry
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

		public BayerColor[] CFAPattern;
		public int xCount;
		public int yCount;

		public ExifCFAPattern(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			xCount = mValue[1];
			yCount = mValue[3];
			CFAPattern = new BayerColor[xCount * yCount];
			for (int x = 0; x < xCount; x++)
			{
				for (int y = 0; y < yCount; y++)
				{
					CFAPattern[y * yCount + x] = (BayerColor)mValue[y * yCount + x + 4];
				}
			}
        }

        public ExifCFAPattern(BayerColor[] aValue, int aXCount, int aYCount)
            : base(new byte[aYCount * aXCount + 4], TagID)
        {
            xCount = mValue[1] = (byte)aXCount;
            yCount = mValue[3] = (byte)aYCount;
            CFAPattern = new BayerColor[xCount * yCount];
            for (int x = 0; x < xCount; x++)
            {
                for (int y = 0; y < yCount; y++)
                {
                    mValue[y * yCount + x + 4] = (byte)aValue[y * yCount + x];
                    CFAPattern[y * yCount + x] = aValue[y * yCount + x];
                }
            }
        }

        public const ushort TagID = 41730;

		public const string TagName = "CFA pattern";

		public BayerColor[] Value
		{
			get { return CFAPattern; }
		}

		public override string ToString()
		{
			int xCount = mValue[1];
			int yCount = mValue[3];

			StringBuilder sb = new StringBuilder();

			for (int x = 0; x < xCount; x++)
			{
				sb.Append("[");
				for (int y = 0; y < yCount; y++)
				{
					switch (mValue[y * yCount + x + 4])
					{
						case 0:
							sb.Append("Red"); break;
						case 1:
							sb.Append("Green"); break;
						case 2:
							sb.Append("Blue"); break;
						case 3:
							sb.Append("Cyan"); break;
						case 4:
							sb.Append("Magenta"); break;
						case 5:
							sb.Append("Yellow"); break;
						case 6:
							sb.Append("White"); break;
						default:
							sb.Append("Other"); break;
					}
					if (y < yCount - 1) sb.Append(", ");
				}
				if (x < xCount - 1)
					sb.Append("] ");
				else
					sb.Append("]");
			}

			return TagName + ": " + sb.ToString();
		}
	}

	public class ExifCustomRendered : UShortExifEntry
	{
		public ExifCustomRendered(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }
        public ExifCustomRendered(bool aValue)
            : base(new ushort[] { aValue ? (ushort)1 : (ushort)0 }, TagID)
        {

        }

        public const ushort TagID = 41985;

		public const string TagName = "Custom rendered";

		public bool Value
		{
			get { return mValue[0] == 1; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class ExifExposureMode : UShortExifEntry
	{
		public enum ExposureMode : ushort
		{ 
			AutoExposure = 0,
			ManualExposure = 1,
			AutoBracket = 2
		}

		public ExifExposureMode(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }
        public ExifExposureMode(ExposureMode aValue)
            : base(new ushort[] { (ushort)aValue }, TagID)
        {

        }

        public const ushort TagID = 41986;

		public const string TagName = "Exposure mode";

		public ExposureMode Value
		{
			get { return (ExposureMode)mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class ExifWhiteBalance : UShortExifEntry
	{
		public enum WhiteBalance : ushort
		{
			Auto = 0,
			Manual = 1
		}

		public ExifWhiteBalance(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }
        public ExifWhiteBalance(WhiteBalance aValue)
            : base(new ushort[] { (ushort)aValue }, TagID)
        {

        }

        public const ushort TagID = 41987;

		public const string TagName = "White balance";

		public WhiteBalance Value
		{
			get { return (WhiteBalance)mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class ExifDigitalZoomRatio : RationalExifEntry
	{
		public ExifDigitalZoomRatio(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }
        public ExifDigitalZoomRatio(Rational aValue)
            : base(new Rational[] { aValue }, TagID)
        {

        }

        public const ushort TagID = 41988;

		public const string TagName = "Digital zoom ratio";

		public Rational Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class ExifFocalLengthIn35mmFilm : UShortExifEntry
	{
		public ExifFocalLengthIn35mmFilm(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }
        public ExifFocalLengthIn35mmFilm(ushort aValue)
            : base(new ushort[] { aValue }, TagID)
        {

        }

        public const ushort TagID = 41989;

		public const string TagName = "Focal length in 35mm film";

		public ushort Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class ExifSceneCaptureType : UShortExifEntry
	{
		public enum SceneCaptureType : ushort
		{
			Standard = 0,
			Landscape = 1,
			Portrait = 2,
			NightScene = 3
		}

		public ExifSceneCaptureType(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }
        public ExifSceneCaptureType(SceneCaptureType aValue)
            : base(new ushort[] { (ushort)aValue }, TagID)
        {

        }

        public const ushort TagID = 41990;

		public const string TagName = "Scene capture type";

		public SceneCaptureType Value
		{
			get { return (SceneCaptureType)mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class ExifGainControl : UShortExifEntry
	{
		public enum GainControl : ushort
		{
			None = 0,
			LowGainUp = 1,
			HighGainUp = 2,
			LowGainDown = 3,
			HighGainDown = 4
		}

		public ExifGainControl(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }
        public ExifGainControl(GainControl aValue)
            : base(new ushort[] { (ushort)aValue }, TagID)
        {

        }

        public const ushort TagID = 41991;

		public const string TagName = "Gain control";

		public GainControl Value
		{
			get { return (GainControl)mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class ExifContrast : UShortExifEntry
	{
		public enum Contrast : ushort
		{
			Normal = 0,
			Soft = 1,
			Hard = 2
		}

		public ExifContrast(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }
        public ExifContrast(Contrast aValue)
            : base(new ushort[] { (ushort)aValue }, TagID)
        {

        }

        public const ushort TagID = 41992;

		public const string TagName = "Contrast";

		public Contrast Value
		{
			get { return (Contrast)mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class ExifSaturation : UShortExifEntry
	{
		public enum Saturation : ushort
		{
			Normal = 0,
			Low = 1,
			High = 2
		}

		public ExifSaturation(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }
        public ExifSaturation(Saturation aValue)
            : base(new ushort[] { (ushort)aValue }, TagID)
        {

        }

        public const ushort TagID = 41993;

		public const string TagName = "Saturation";

		public Saturation Value
		{
			get { return (Saturation)mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class ExifSharpness: UShortExifEntry
	{
		public enum Sharpness : ushort
		{
			Normal = 0,
			Soft = 1,
			Hard = 2
		}

		public ExifSharpness(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }
        public ExifSharpness(Sharpness aValue)
            : base(new ushort[] { (ushort)aValue }, TagID)
        {

        }

        public const ushort TagID = 41994;

		public const string TagName = "Sharpness";

		public Sharpness Value
		{
			get { return (Sharpness)mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class ExifDeviceSettingDescription: ByteExifEntry
	{
		public ExifDeviceSettingDescription(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }

        public ExifDeviceSettingDescription(byte[] aValue)
            : base(aValue, TagID)
        {

        }

        public const ushort TagID = 41995;

		public const string TagName = "Device setting description";

		public byte[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.Length.ToString() + " entries";
		}
	}

	public class ExifSubjectDistanceRange: UShortExifEntry
	{
		public enum SubjectDistanceRange : ushort
		{
			Unknown = 0,
			Macro = 1,
			CloseView = 2,
			DistantView = 3
		}

		public ExifSubjectDistanceRange(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }
        public ExifSubjectDistanceRange(SubjectDistanceRange aValue)
            : base(new ushort[] { (ushort)aValue }, TagID)
        {

        }

        public const ushort TagID = 41996;

		public const string TagName = "Subject distance range";

		public SubjectDistanceRange Value
		{
			get { return (SubjectDistanceRange)mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class ExifImageUniqueID: StringExifEntry
	{
		public ExifImageUniqueID(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }
        public ExifImageUniqueID(string aValue)
            : base(aValue, TagID)
        {

        }

        public const ushort TagID = 42016;

		public const string TagName = "Image unique ID";

		public string Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value;
		}
	}

	public class ExifSensitivityType : UShortExifEntry
	{
		public ExifSensitivityType(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }
        public ExifSensitivityType(ushort aValue)
            : base(new ushort[] { aValue }, TagID)
        {

        }

        public const ushort TagID = 34864;

		public const string TagName = "Sensitivity type";

		public ushort Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class ExifStandardOutputSensitivity : UShortExifEntry
	{
		public ExifStandardOutputSensitivity(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

        }
        public ExifStandardOutputSensitivity(ushort aValue)
            : base(new ushort[] { aValue }, TagID)
        {

        }

        public const ushort TagID = 34865;

		public const string TagName = "Standard output sensitivity";

		public ushort Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
    }

    public class ExifBodySerialNumber : StringExifEntry
    {
        public ExifBodySerialNumber(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        {

        }
        public ExifBodySerialNumber(string aValue)
            : base(aValue, TagID)
        {

        }

        public const ushort TagID = 42033;

        public const string TagName = "Body Serial Number";

        public string Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + Value;
        }
    }

    public class ExifLensSpecification : RationalExifEntry
    {
        public ExifLensSpecification(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        {

        }
        public ExifLensSpecification(Rational[] aValue)
            : base(aValue, TagID)
        {

        }

        public const ushort TagID = 42034;

        public const string TagName = "Lens Specification";

        public Rational[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + Value[0].ToString() + " + " + (Value.Length - 1) + " more entries.";
        }
    }

    public class ExifLensMake : StringExifEntry
    {
        public ExifLensMake(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        {

        }
        public ExifLensMake(string aValue)
            : base(aValue, TagID)
        {

        }

        public const ushort TagID = 42035;

        public const string TagName = "Lens Make";

        public string Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + Value;
        }
    }

    public class ExifLensModel : StringExifEntry
    {
        public ExifLensModel(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        {

        }
        public ExifLensModel(string aValue)
            : base(aValue, TagID)
        {

        }

        public const ushort TagID = 42036;

        public const string TagName = "Lens Model";

        public string Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + Value;
        }
    }

    public class ExifLensSerialNumber : StringExifEntry
    {
        public ExifLensSerialNumber(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        {

        }
        public ExifLensSerialNumber(string aValue)
            : base(aValue, TagID)
        {

        }

        public const ushort TagID = 42037;

        public const string TagName = "Lens Serial Number";

        public string Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + Value;
        }
    }
    #endregion
}
