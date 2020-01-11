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
using System.Runtime.InteropServices;

namespace PentaxPefFile
{
	public class ImageFileDirectoryEntry : SaveTiffTag
	{
		protected FileReader mPefFile;
		public ushort mTagID;
		protected TIFFValueType mFieldType;
		protected uint mValueCount;
		protected uint mOffset;

		public ImageFileDirectoryEntry(FileReader aPefFile)
		{
			mPefFile = aPefFile;
			mTagID = mPefFile.ReadUI2();
			mFieldType = new TIFFValueType((TIFFValueTypes)mPefFile.ReadUI2());
			mValueCount = mPefFile.ReadUI4();
			mOffset = mPefFile.ReadUI4();
		}

		protected ImageFileDirectoryEntry(FileReader aPefFile, ushort aTagID)
		{
			mPefFile = aPefFile;
			mTagID = aTagID;
			mFieldType = new TIFFValueType((TIFFValueTypes)mPefFile.ReadUI2());
			mValueCount = mPefFile.ReadUI4();
			mOffset = mPefFile.ReadUI4();
		}

        protected ImageFileDirectoryEntry(ushort aTagID, TIFFValueType aFieldType, uint aValueCount)
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

        public override string ToString()
        {
            return "Unknown IFD entry. ID: " + mTagID;
        }

        public static ImageFileDirectoryEntry CreateImageFileDirectoryEntry(FileReader aPefFile)
		{
			ushort tagID = aPefFile.ReadUI2();

			switch (tagID)	
			{
				case IFDArtist.TagID: 
					return new IFDArtist(aPefFile, tagID);
				case IFDBitsPerSample.TagID:
					return new IFDBitsPerSample(aPefFile, tagID);
				case IFDCellLength.TagID:
					return new IFDCellLength(aPefFile, tagID);
				case IFDCellWidth.TagID:
					return new IFDCellWidth(aPefFile, tagID);
				case IFDColorMap.TagID:
					return new IFDColorMap(aPefFile, tagID);
				case IFDCompression.TagID:
					return new IFDCompression(aPefFile, tagID);
				case IFDCopyright.TagID:
					return new IFDCopyright(aPefFile, tagID);
				case IFDDateTime.TagID:
					return new IFDDateTime(aPefFile, tagID);
				case IFDExtraSamples.TagID:
					return new IFDExtraSamples(aPefFile, tagID);
				case IFDFillOrder.TagID:
					return new IFDFillOrder(aPefFile, tagID);
				case IFDFreeByteCounts.TagID:
					return new IFDFreeByteCounts(aPefFile, tagID);
				case IFDFreeOffsets.TagID:
					return new IFDFreeOffsets(aPefFile, tagID);
				case IFDGrayResponseCurve.TagID:
					return new IFDGrayResponseCurve(aPefFile, tagID);
				case IFDGrayResponseUnit.TagID:
					return new IFDGrayResponseUnit(aPefFile, tagID);
				case IFDHostComputer.TagID:
					return new IFDHostComputer(aPefFile, tagID);
				case IFDImageDescription.TagID:
					return new IFDImageDescription(aPefFile, tagID);
				case IFDImageLength.TagID:
					return new IFDImageLength(aPefFile, tagID);
				case IFDImageWidth.TagID:
					return new IFDImageWidth(aPefFile, tagID);
				case IFDMake.TagID:
					return new IFDMake(aPefFile, tagID);
				case IFDMaxSampleValue.TagID:
					return new IFDMaxSampleValue(aPefFile, tagID);
				case IFDMinSampleValue.TagID:
					return new IFDMinSampleValue(aPefFile, tagID);
				case IFDModel.TagID:
					return new IFDModel(aPefFile, tagID);
				case IFDNewSubfileType.TagID:
					return new IFDNewSubfileType(aPefFile, tagID);
				case IFDOrientation.TagID:
					return new IFDOrientation(aPefFile, tagID);
				case IFDPhotometricInterpretation.TagID:
					return new IFDPhotometricInterpretation(aPefFile, tagID);
				case IFDPlanarConfiguration.TagID:
					return new IFDPlanarConfiguration(aPefFile, tagID);
				case IFDResolutionUnit.TagID:
					return new IFDResolutionUnit(aPefFile, tagID);
				case IFDRowsPerStrip.TagID:
					return new IFDRowsPerStrip(aPefFile, tagID);
				case IFDSamplesPerPixel.TagID:
					return new IFDSamplesPerPixel(aPefFile, tagID);
				case IFDSoftware.TagID:
					return new IFDSoftware(aPefFile, tagID);
				case IFDStripByteCounts.TagID:
					return new IFDStripByteCounts(aPefFile, tagID);
				case IFDStripOffsets.TagID:
					return new IFDStripOffsets(aPefFile, tagID);
				case IFDSubfileType.TagID:
					return new IFDSubfileType(aPefFile, tagID);
				case IFDThreshholding.TagID:
					return new IFDThreshholding(aPefFile, tagID);
				case IFDXResolution.TagID:
					return new IFDXResolution(aPefFile, tagID);
				case IFDYResolution.TagID:
					return new IFDYResolution(aPefFile, tagID);
				case IFDExif.TagID:
					return new IFDExif(aPefFile, tagID);
				case IFDGps.TagID:
					return new IFDGps(aPefFile, tagID);
				case IFDJPEGInterchangeFormat.TagID:
					return new IFDJPEGInterchangeFormat(aPefFile, tagID);
				case IFDJPEGInterchangeFormatLength.TagID:
					return new IFDJPEGInterchangeFormatLength(aPefFile, tagID);
                //DNG:
                case IFDDNGVersion.TagID:
                    return new IFDDNGVersion(aPefFile, tagID);
                case IFDDNGBackwardVersion.TagID:
                    return new IFDDNGBackwardVersion(aPefFile, tagID);
                case IFDDNGBlackLevelRepeatDim.TagID:
                    return new IFDDNGBlackLevelRepeatDim(aPefFile, tagID);
                case IFDDNGCFALayout.TagID:
                    return new IFDDNGCFALayout(aPefFile, tagID);
                case IFDDNGCFAPlaneColor.TagID:
                    return new IFDDNGCFAPlaneColor(aPefFile, tagID);
                case IFDDNGLinearizationTable.TagID:
                    return new IFDDNGLinearizationTable(aPefFile, tagID);
                case IFDDNGLocalizedCameraModel.TagID:
                    return new IFDDNGLocalizedCameraModel(aPefFile, tagID);
                case IFDDNGUniqueCameraModel.TagID:
                    return new IFDDNGUniqueCameraModel(aPefFile, tagID);
                case IFDSubIFDs.TagID:
                    return new IFDSubIFDs(aPefFile, tagID);

                case IFDTileWidth.TagID:
                    return new IFDTileWidth(aPefFile, tagID);
                case IFDTileLength.TagID:
                    return new IFDTileLength(aPefFile, tagID);
                case IFDTileOffsets.TagID:
                    return new IFDTileOffsets(aPefFile, tagID);
                case IFDTileByteCounts.TagID:
                    return new IFDTileByteCounts(aPefFile, tagID);

                case IFDCFARepeatPatternDim.TagID:
                    return new IFDCFARepeatPatternDim(aPefFile, tagID);
                case IFDCFAPattern.TagID:
                    return new IFDCFAPattern(aPefFile, tagID);

                case IFDSampleFormat.TagID:
                    return new IFDSampleFormat(aPefFile, tagID);
                case IFDDocumentName.TagID:
                    return new IFDDocumentName(aPefFile, tagID);

                case IFDFocalLength.TagID:
                    return new IFDFocalLength(aPefFile, tagID);
                case IFDDateTimeOriginal.TagID:
                    return new IFDDateTimeOriginal(aPefFile, tagID);
                case IFDISOSpeedRatings.TagID:
                    return new IFDISOSpeedRatings(aPefFile, tagID);
                case IFDFNumber.TagID:
                    return new IFDFNumber(aPefFile, tagID);
                case IFDExposureTime.TagID:
                    return new IFDExposureTime(aPefFile, tagID);

                case IFDDNGAnalogBalance.TagID:
                    return new IFDDNGAnalogBalance(aPefFile, tagID);
                case IFDDNGAsShotNeutral.TagID:
                    return new IFDDNGAsShotNeutral(aPefFile, tagID);
                case IFDDNGAsShotWhiteXY.TagID:
                    return new IFDDNGAsShotWhiteXY(aPefFile, tagID);
                case IFDDNGBaselineExposure.TagID:
                    return new IFDDNGBaselineExposure(aPefFile, tagID);
                case IFDDNGBaselineNoise.TagID:
                    return new IFDDNGBaselineNoise(aPefFile, tagID);
                case IFDDNGBaselineSharpness.TagID:
                    return new IFDDNGBaselineSharpness(aPefFile, tagID);
                case IFDDNGCalibrationIlluminant1.TagID:
                    return new IFDDNGCalibrationIlluminant1(aPefFile, tagID);
                case IFDDNGCalibrationIlluminant2.TagID:
                    return new IFDDNGCalibrationIlluminant2(aPefFile, tagID);
                case IFDDNGColorMatrix1.TagID:
                    return new IFDDNGColorMatrix1(aPefFile, tagID);
                case IFDDNGColorMatrix2.TagID:
                    return new IFDDNGColorMatrix2(aPefFile, tagID);
                case IFDDNGForwardMatrix1.TagID:
                    return new IFDDNGForwardMatrix1(aPefFile, tagID);
                case IFDDNGForwardMatrix2.TagID:
                    return new IFDDNGForwardMatrix2(aPefFile, tagID);
                case IFDDNGReductionMatrix1.TagID:
                    return new IFDDNGReductionMatrix1(aPefFile, tagID);
                case IFDDNGReductionMatrix2.TagID:
                    return new IFDDNGReductionMatrix2(aPefFile, tagID);
                case IFDDNGImageNumber.TagID:
                    return new IFDDNGImageNumber(aPefFile, tagID);
                case IFDDNGLensInfo.TagID:
                    return new IFDDNGLensInfo(aPefFile, tagID);
                case IFDDNGLinearResponseLimit.TagID:
                    return new IFDDNGLinearResponseLimit(aPefFile, tagID);
                case IFDDNGOriginalRawFileName.TagID:
                    return new IFDDNGOriginalRawFileName(aPefFile, tagID);
                case IFDDNGPreviewApplicationName.TagID:
                    return new IFDDNGPreviewApplicationName(aPefFile, tagID);
                case IFDDNGPreviewApplicationVersion.TagID:
                    return new IFDDNGPreviewApplicationVersion(aPefFile, tagID);
                case IFDDNGPreviewColorSpace.TagID:
                    return new IFDDNGPreviewColorSpace(aPefFile, tagID);
                case IFDDNGPreviewDateTime.TagID:
                    return new IFDDNGPreviewDateTime(aPefFile, tagID);
                case IFDDNGPreviewSettingsDigest.TagID:
                    return new IFDDNGPreviewSettingsDigest(aPefFile, tagID);
                case IFDDNGPrivateData.TagID:
                    return new IFDDNGPrivateData(aPefFile, tagID);
                case IFDDNGProfileCalibrationSignature.TagID:
                    return new IFDDNGProfileCalibrationSignature(aPefFile, tagID);
                case IFDDNGProfileCopyright.TagID:
                    return new IFDDNGProfileCopyright(aPefFile, tagID);
                case IFDDNGProfileEmbedPolicy.TagID:
                    return new IFDDNGProfileEmbedPolicy(aPefFile, tagID);
                case IFDDNGProfileLookTableData.TagID:
                    return new IFDDNGProfileLookTableData(aPefFile, tagID);
                case IFDDNGProfileLookTableDims.TagID:
                    return new IFDDNGProfileLookTableDims(aPefFile, tagID);
                case IFDDNGProfileName.TagID:
                    return new IFDDNGProfileName(aPefFile, tagID);
                case IFDDNGRawDataUniqueID.TagID:
                    return new IFDDNGRawDataUniqueID(aPefFile, tagID);
                case IFDDNGRawImageDigest.TagID:
                    return new IFDDNGRawImageDigest(aPefFile, tagID);
                case IFDDNGShadowScale.TagID:
                    return new IFDDNGShadowScale(aPefFile, tagID);
                case IFDDNGTimeZoneOffset.TagID:
                    return new IFDDNGTimeZoneOffset(aPefFile, tagID);
                case IFDDNGXMPMetaData.TagID:
                    return new IFDDNGXMPMetaData(aPefFile, tagID);
                case IFDDNGBlackLevel.TagID:
                    return new IFDDNGBlackLevel(aPefFile, tagID);
                case IFDDNGBlackLevelDeltaH.TagID:
                    return new IFDDNGBlackLevelDeltaH(aPefFile, tagID);
                case IFDDNGBlackLevelDeltaV.TagID:
                    return new IFDDNGBlackLevelDeltaV(aPefFile, tagID);
                case IFDDNGWhiteLevel.TagID:
                    return new IFDDNGWhiteLevel(aPefFile, tagID);
                case IFDDNGDefaultScale.TagID:
                    return new IFDDNGDefaultScale(aPefFile, tagID);
                case IFDDNGDefaultCropOrigin.TagID:
                    return new IFDDNGDefaultCropOrigin(aPefFile, tagID);
                case IFDDNGDefaultCropSize.TagID:
                    return new IFDDNGDefaultCropSize(aPefFile, tagID);
                case IFDDNGBayerGreenSplit.TagID:
                    return new IFDDNGBayerGreenSplit(aPefFile, tagID);
                case IFDDNGChromaBlurRadius.TagID:
                    return new IFDDNGChromaBlurRadius(aPefFile, tagID);
                case IFDDNGActiveArea.TagID:
                    return new IFDDNGActiveArea(aPefFile, tagID);
                case IFDDNGBestQualityScale.TagID:
                    return new IFDDNGBestQualityScale(aPefFile, tagID);
                case IFDDNGAntiAliasStrength.TagID:
                    return new IFDDNGAntiAliasStrength(aPefFile, tagID);
                case IFDDNGNoiseProfile.TagID:
                    return new IFDDNGNoiseProfile(aPefFile, tagID);
                case IFDDNGOpcodeList1.TagID:
                    return new IFDDNGOpcodeList1(aPefFile, tagID);
                case IFDDNGOpcodeList2.TagID:
                    return new IFDDNGOpcodeList2(aPefFile, tagID);
                case IFDDNGOpcodeList3.TagID:
                    return new IFDDNGOpcodeList3(aPefFile, tagID);
                case IFDTIFFEPStandardID.TagID:
                    return new IFDTIFFEPStandardID(aPefFile, tagID);
                case IFDDNGCameraCalibration1.TagID:
                    return new IFDDNGCameraCalibration1(aPefFile, tagID);
                case IFDDNGCameraCalibration2.TagID:
                    return new IFDDNGCameraCalibration2(aPefFile, tagID);

                case IFDDNGProfileLookTableEncoding.TagID:
                    return new IFDDNGProfileLookTableEncoding(aPefFile, tagID);
                case IFDDNGCameraSerialNumber.TagID:
                    return new IFDDNGCameraSerialNumber(aPefFile, tagID);
                case IFDDNGCameraCalibrationSignature.TagID:
                    return new IFDDNGCameraCalibrationSignature(aPefFile, tagID);
                case IFDDNGProfileHueSatMapDims.TagID:
                    return new IFDDNGProfileHueSatMapDims(aPefFile, tagID);
                case IFDDNGProfileHueSatMapData1.TagID:
                    return new IFDDNGProfileHueSatMapData1(aPefFile, tagID);
                case IFDDNGProfileHueSatMapData2.TagID:
                    return new IFDDNGProfileHueSatMapData2(aPefFile, tagID);
                case IFDDNGOriginalRawFileData.TagID:
                    return new IFDDNGOriginalRawFileData(aPefFile, tagID);
                case IFDDNGMaskedAreas.TagID:
                    return new IFDDNGMaskedAreas(aPefFile, tagID);
                case IFDDNGAsShotICCProfile.TagID:
                    return new IFDDNGAsShotICCProfile(aPefFile, tagID);
                case IFDDNGAsShotPreProfileMatrix.TagID:
                    return new IFDDNGAsShotPreProfileMatrix(aPefFile, tagID);
                case IFDDNGCurrentICCProfile.TagID:
                    return new IFDDNGCurrentICCProfile(aPefFile, tagID);
                case IFDDNGCurrentPreProfileMatrix.TagID:
                    return new IFDDNGCurrentPreProfileMatrix(aPefFile, tagID);

                case IFDDNGColorimetricReference.TagID:
                    return new IFDDNGColorimetricReference(aPefFile, tagID);
                case IFDDNGExtraCameraProfiles.TagID:
                    return new IFDDNGExtraCameraProfiles(aPefFile, tagID);
                case IFDDNGAsShotProfileName.TagID:
                    return new IFDDNGAsShotProfileName(aPefFile, tagID);
                case IFDDNGNoiseReductionApplied.TagID:
                    return new IFDDNGNoiseReductionApplied(aPefFile, tagID);
                case IFDDNGProfileToneCurve.TagID:
                    return new IFDDNGProfileToneCurve(aPefFile, tagID);
                case IFDDNGPreviewSettingsName.TagID:
                    return new IFDDNGPreviewSettingsName(aPefFile, tagID);
                case IFDDNGOriginalRawFileDigest.TagID:
                    return new IFDDNGOriginalRawFileDigest(aPefFile, tagID);

                case IFDDNGSubTileBlockSize.TagID:
                    return new IFDDNGSubTileBlockSize(aPefFile, tagID);
                case IFDDNGRowInterleaveFactor.TagID:
                    return new IFDDNGRowInterleaveFactor(aPefFile, tagID);

                case IFDDNGDefaultUserCrop.TagID:
                    return new IFDDNGDefaultUserCrop(aPefFile, tagID);
                case IFDDNGDefaultBlackRender.TagID:
                    return new IFDDNGDefaultBlackRender(aPefFile, tagID);
                case IFDDNGBaselineExposureOffset.TagID:
                    return new IFDDNGBaselineExposureOffset(aPefFile, tagID);
                case IFDDNGProfileHueSatMapEncoding.TagID:
                    return new IFDDNGProfileHueSatMapEncoding(aPefFile, tagID);
                case IFDDNGOriginalDefaultFinalSize.TagID:
                    return new IFDDNGOriginalDefaultFinalSize(aPefFile, tagID);
                case IFDDNGOriginalBestQualityFinalSize.TagID:
                    return new IFDDNGOriginalBestQualityFinalSize(aPefFile, tagID);
                case IFDDNGOriginalDefaultCropSize.TagID:
                    return new IFDDNGOriginalDefaultCropSize(aPefFile, tagID);
                case IFDDNGNewRawImageDigest.TagID:
                    return new IFDDNGNewRawImageDigest(aPefFile, tagID);
                case IFDDNGRawToPreviewGain.TagID:
                    return new IFDDNGRawToPreviewGain(aPefFile, tagID);

                default:
					return new IFDUnknownTag(aPefFile, tagID);
			}
		}

        public static string ConvertIdToName(ushort tagID)
        {
            switch (tagID)
            {
                case IFDArtist.TagID:
                    return IFDArtist.TagName;
                case IFDBitsPerSample.TagID:
                    return IFDBitsPerSample.TagName;
                case IFDCellLength.TagID:
                    return IFDCellLength.TagName;
                case IFDCellWidth.TagID:
                    return IFDCellWidth.TagName;
                case IFDColorMap.TagID:
                    return IFDColorMap.TagName;
                case IFDCompression.TagID:
                    return IFDCompression.TagName;
                case IFDCopyright.TagID:
                    return IFDCopyright.TagName;
                case IFDDateTime.TagID:
                    return IFDDateTime.TagName;
                case IFDExtraSamples.TagID:
                    return IFDExtraSamples.TagName;
                case IFDFillOrder.TagID:
                    return IFDFillOrder.TagName;
                case IFDFreeByteCounts.TagID:
                    return IFDFreeByteCounts.TagName;
                case IFDFreeOffsets.TagID:
                    return IFDFreeOffsets.TagName;
                case IFDGrayResponseCurve.TagID:
                    return IFDGrayResponseCurve.TagName;
                case IFDGrayResponseUnit.TagID:
                    return IFDGrayResponseUnit.TagName;
                case IFDHostComputer.TagID:
                    return IFDHostComputer.TagName;
                case IFDImageDescription.TagID:
                    return IFDImageDescription.TagName;
                case IFDImageLength.TagID:
                    return IFDImageLength.TagName;
                case IFDImageWidth.TagID:
                    return IFDImageWidth.TagName;
                case IFDMake.TagID:
                    return IFDMake.TagName;
                case IFDMaxSampleValue.TagID:
                    return IFDMaxSampleValue.TagName;
                case IFDMinSampleValue.TagID:
                    return IFDMinSampleValue.TagName;
                case IFDModel.TagID:
                    return IFDModel.TagName;
                case IFDNewSubfileType.TagID:
                    return IFDNewSubfileType.TagName;
                case IFDOrientation.TagID:
                    return IFDOrientation.TagName;
                case IFDPhotometricInterpretation.TagID:
                    return IFDPhotometricInterpretation.TagName;
                case IFDPlanarConfiguration.TagID:
                    return IFDPlanarConfiguration.TagName;
                case IFDResolutionUnit.TagID:
                    return IFDResolutionUnit.TagName;
                case IFDRowsPerStrip.TagID:
                    return IFDRowsPerStrip.TagName;
                case IFDSamplesPerPixel.TagID:
                    return IFDSamplesPerPixel.TagName;
                case IFDSoftware.TagID:
                    return IFDSoftware.TagName;
                case IFDStripByteCounts.TagID:
                    return IFDStripByteCounts.TagName;
                case IFDStripOffsets.TagID:
                    return IFDStripOffsets.TagName;
                case IFDSubfileType.TagID:
                    return IFDSubfileType.TagName;
                case IFDThreshholding.TagID:
                    return IFDThreshholding.TagName;
                case IFDXResolution.TagID:
                    return IFDXResolution.TagName;
                case IFDYResolution.TagID:
                    return IFDYResolution.TagName;
                case IFDExif.TagID:
                    return IFDExif.TagName;
                case IFDGps.TagID:
                    return IFDGps.TagName;
                case IFDJPEGInterchangeFormat.TagID:
                    return IFDJPEGInterchangeFormat.TagName;
                case IFDJPEGInterchangeFormatLength.TagID:
                    return IFDJPEGInterchangeFormatLength.TagName;
                //DNG:
                case IFDDNGVersion.TagID:
                    return IFDDNGVersion.TagName;
                case IFDDNGBackwardVersion.TagID:
                    return IFDDNGBackwardVersion.TagName;
                case IFDDNGBlackLevelRepeatDim.TagID:
                    return IFDDNGBlackLevelRepeatDim.TagName;
                case IFDDNGCFALayout.TagID:
                    return IFDDNGCFALayout.TagName;
                case IFDDNGCFAPlaneColor.TagID:
                    return IFDDNGCFAPlaneColor.TagName;
                case IFDDNGLinearizationTable.TagID:
                    return IFDDNGLinearizationTable.TagName;
                case IFDDNGLocalizedCameraModel.TagID:
                    return IFDDNGLocalizedCameraModel.TagName;
                case IFDDNGUniqueCameraModel.TagID:
                    return IFDDNGUniqueCameraModel.TagName;
                case IFDSubIFDs.TagID:
                    return IFDSubIFDs.TagName;

                case IFDTileWidth.TagID:
                    return IFDTileWidth.TagName;
                case IFDTileLength.TagID:
                    return IFDTileLength.TagName;
                case IFDTileOffsets.TagID:
                    return IFDTileOffsets.TagName;
                case IFDTileByteCounts.TagID:
                    return IFDTileByteCounts.TagName;

                case IFDCFARepeatPatternDim.TagID:
                    return IFDCFARepeatPatternDim.TagName;
                case IFDCFAPattern.TagID:
                    return IFDCFAPattern.TagName;

                case IFDSampleFormat.TagID:
                    return IFDSampleFormat.TagName;
                case IFDDocumentName.TagID:
                    return IFDDocumentName.TagName;

                case IFDFocalLength.TagID:
                    return IFDFocalLength.TagName;
                case IFDDateTimeOriginal.TagID:
                    return IFDDateTimeOriginal.TagName;
                case IFDISOSpeedRatings.TagID:
                    return IFDISOSpeedRatings.TagName;
                case IFDFNumber.TagID:
                    return IFDFNumber.TagName;
                case IFDExposureTime.TagID:
                    return IFDExposureTime.TagName;

                case IFDDNGAnalogBalance.TagID:
                    return IFDDNGAnalogBalance.TagName;
                case IFDDNGAsShotNeutral.TagID:
                    return IFDDNGAsShotNeutral.TagName;
                case IFDDNGAsShotWhiteXY.TagID:
                    return IFDDNGAsShotWhiteXY.TagName;
                case IFDDNGBaselineExposure.TagID:
                    return IFDDNGBaselineExposure.TagName;
                case IFDDNGBaselineNoise.TagID:
                    return IFDDNGBaselineNoise.TagName;
                case IFDDNGBaselineSharpness.TagID:
                    return IFDDNGBaselineSharpness.TagName;
                case IFDDNGCalibrationIlluminant1.TagID:
                    return IFDDNGCalibrationIlluminant1.TagName;
                case IFDDNGCalibrationIlluminant2.TagID:
                    return IFDDNGCalibrationIlluminant2.TagName;
                case IFDDNGColorMatrix1.TagID:
                    return IFDDNGColorMatrix1.TagName;
                case IFDDNGColorMatrix2.TagID:
                    return IFDDNGColorMatrix2.TagName;
                case IFDDNGForwardMatrix1.TagID:
                    return IFDDNGForwardMatrix1.TagName;
                case IFDDNGForwardMatrix2.TagID:
                    return IFDDNGForwardMatrix2.TagName;
                case IFDDNGReductionMatrix1.TagID:
                    return IFDDNGReductionMatrix1.TagName;
                case IFDDNGReductionMatrix2.TagID:
                    return IFDDNGReductionMatrix2.TagName;
                case IFDDNGImageNumber.TagID:
                    return IFDDNGImageNumber.TagName;
                case IFDDNGLensInfo.TagID:
                    return IFDDNGLensInfo.TagName;
                case IFDDNGLinearResponseLimit.TagID:
                    return IFDDNGLinearResponseLimit.TagName;
                case IFDDNGOriginalRawFileName.TagID:
                    return IFDDNGOriginalRawFileName.TagName;
                case IFDDNGPreviewApplicationName.TagID:
                    return IFDDNGPreviewApplicationName.TagName;
                case IFDDNGPreviewApplicationVersion.TagID:
                    return IFDDNGPreviewApplicationVersion.TagName;
                case IFDDNGPreviewColorSpace.TagID:
                    return IFDDNGPreviewColorSpace.TagName;
                case IFDDNGPreviewDateTime.TagID:
                    return IFDDNGPreviewDateTime.TagName;
                case IFDDNGPreviewSettingsDigest.TagID:
                    return IFDDNGPreviewSettingsDigest.TagName;
                case IFDDNGPrivateData.TagID:
                    return IFDDNGPrivateData.TagName;
                case IFDDNGProfileCalibrationSignature.TagID:
                    return IFDDNGProfileCalibrationSignature.TagName;
                case IFDDNGProfileCopyright.TagID:
                    return IFDDNGProfileCopyright.TagName;
                case IFDDNGProfileEmbedPolicy.TagID:
                    return IFDDNGProfileEmbedPolicy.TagName;
                case IFDDNGProfileLookTableData.TagID:
                    return IFDDNGProfileLookTableData.TagName;
                case IFDDNGProfileLookTableDims.TagID:
                    return IFDDNGProfileLookTableDims.TagName;
                case IFDDNGProfileName.TagID:
                    return IFDDNGProfileName.TagName;
                case IFDDNGRawDataUniqueID.TagID:
                    return IFDDNGRawDataUniqueID.TagName;
                case IFDDNGRawImageDigest.TagID:
                    return IFDDNGRawImageDigest.TagName;
                case IFDDNGShadowScale.TagID:
                    return IFDDNGShadowScale.TagName;
                case IFDDNGTimeZoneOffset.TagID:
                    return IFDDNGTimeZoneOffset.TagName;
                case IFDDNGXMPMetaData.TagID:
                    return IFDDNGXMPMetaData.TagName;
                case IFDDNGBlackLevel.TagID:
                    return IFDDNGBlackLevel.TagName;
                case IFDDNGBlackLevelDeltaH.TagID:
                    return IFDDNGBlackLevelDeltaH.TagName;
                case IFDDNGBlackLevelDeltaV.TagID:
                    return IFDDNGBlackLevelDeltaV.TagName;
                case IFDDNGWhiteLevel.TagID:
                    return IFDDNGWhiteLevel.TagName;
                case IFDDNGDefaultScale.TagID:
                    return IFDDNGDefaultScale.TagName;
                case IFDDNGDefaultCropOrigin.TagID:
                    return IFDDNGDefaultCropOrigin.TagName;
                case IFDDNGDefaultCropSize.TagID:
                    return IFDDNGDefaultCropSize.TagName;
                case IFDDNGBayerGreenSplit.TagID:
                    return IFDDNGBayerGreenSplit.TagName;
                case IFDDNGChromaBlurRadius.TagID:
                    return IFDDNGChromaBlurRadius.TagName;
                case IFDDNGActiveArea.TagID:
                    return IFDDNGActiveArea.TagName;
                case IFDDNGBestQualityScale.TagID:
                    return IFDDNGBestQualityScale.TagName;
                case IFDDNGAntiAliasStrength.TagID:
                    return IFDDNGAntiAliasStrength.TagName;
                case IFDDNGNoiseProfile.TagID:
                    return IFDDNGNoiseProfile.TagName;
                case IFDDNGOpcodeList1.TagID:
                    return IFDDNGOpcodeList1.TagName;
                case IFDDNGOpcodeList2.TagID:
                    return IFDDNGOpcodeList2.TagName;
                case IFDDNGOpcodeList3.TagID:
                    return IFDDNGOpcodeList3.TagName;
                case IFDTIFFEPStandardID.TagID:
                    return IFDTIFFEPStandardID.TagName;
                case IFDDNGCameraCalibration1.TagID:
                    return IFDDNGCameraCalibration1.TagName;
                case IFDDNGCameraCalibration2.TagID:
                    return IFDDNGCameraCalibration2.TagName;

                case IFDDNGProfileLookTableEncoding.TagID:
                    return IFDDNGProfileLookTableEncoding.TagName;
                case IFDDNGCameraSerialNumber.TagID:
                    return IFDDNGCameraSerialNumber.TagName;
                case IFDDNGCameraCalibrationSignature.TagID:
                    return IFDDNGCameraCalibrationSignature.TagName;
                case IFDDNGProfileHueSatMapDims.TagID:
                    return IFDDNGProfileHueSatMapDims.TagName;
                case IFDDNGProfileHueSatMapData1.TagID:
                    return IFDDNGProfileHueSatMapData1.TagName;
                case IFDDNGProfileHueSatMapData2.TagID:
                    return IFDDNGProfileHueSatMapData2.TagName;
                case IFDDNGOriginalRawFileData.TagID:
                    return IFDDNGOriginalRawFileData.TagName;
                case IFDDNGMaskedAreas.TagID:
                    return IFDDNGMaskedAreas.TagName;
                case IFDDNGAsShotICCProfile.TagID:
                    return IFDDNGAsShotICCProfile.TagName;
                case IFDDNGAsShotPreProfileMatrix.TagID:
                    return IFDDNGAsShotPreProfileMatrix.TagName;
                case IFDDNGCurrentICCProfile.TagID:
                    return IFDDNGCurrentICCProfile.TagName;
                case IFDDNGCurrentPreProfileMatrix.TagID:
                    return IFDDNGCurrentPreProfileMatrix.TagName;

                case IFDDNGColorimetricReference.TagID:
                    return IFDDNGColorimetricReference.TagName;
                case IFDDNGExtraCameraProfiles.TagID:
                    return IFDDNGExtraCameraProfiles.TagName;
                case IFDDNGAsShotProfileName.TagID:
                    return IFDDNGAsShotProfileName.TagName;
                case IFDDNGNoiseReductionApplied.TagID:
                    return IFDDNGNoiseReductionApplied.TagName;
                case IFDDNGProfileToneCurve.TagID:
                    return IFDDNGProfileToneCurve.TagName;
                case IFDDNGPreviewSettingsName.TagID:
                    return IFDDNGPreviewSettingsName.TagName;
                case IFDDNGOriginalRawFileDigest.TagID:
                    return IFDDNGOriginalRawFileDigest.TagName;

                case IFDDNGSubTileBlockSize.TagID:
                    return IFDDNGSubTileBlockSize.TagName;
                case IFDDNGRowInterleaveFactor.TagID:
                    return IFDDNGRowInterleaveFactor.TagName;

                case IFDDNGDefaultUserCrop.TagID:
                    return IFDDNGDefaultUserCrop.TagName;
                case IFDDNGDefaultBlackRender.TagID:
                    return IFDDNGDefaultBlackRender.TagName;
                case IFDDNGBaselineExposureOffset.TagID:
                    return IFDDNGBaselineExposureOffset.TagName;
                case IFDDNGProfileHueSatMapEncoding.TagID:
                    return IFDDNGProfileHueSatMapEncoding.TagName;
                case IFDDNGOriginalDefaultFinalSize.TagID:
                    return IFDDNGOriginalDefaultFinalSize.TagName;
                case IFDDNGOriginalBestQualityFinalSize.TagID:
                    return IFDDNGOriginalBestQualityFinalSize.TagName;
                case IFDDNGOriginalDefaultCropSize.TagID:
                    return IFDDNGOriginalDefaultCropSize.TagName;
                case IFDDNGNewRawImageDigest.TagID:
                    return IFDDNGNewRawImageDigest.TagName;
                case IFDDNGRawToPreviewGain.TagID:
                    return IFDDNGRawToPreviewGain.TagName;

                default:
                    return tagID.ToString() + ": Unknown Tag";
            }
        }
	}

	#region Typed base classes
	public abstract class StringImageFileDirectoryEntry : ImageFileDirectoryEntry
	{
		protected string mValue;

        public StringImageFileDirectoryEntry(string aValue, ushort aTagID)
            : base(aTagID, new TIFFValueType(TIFFValueTypes.Ascii), 0)
        {
            mValue = aValue;
        }

        public StringImageFileDirectoryEntry(FileReader aPefFile, ushort aTagID)
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

	public abstract class ByteImageFileDirectoryEntry : ImageFileDirectoryEntry
	{
		protected byte[] mValue;

        public ByteImageFileDirectoryEntry(byte[] aValue, ushort aTagID)
            : base(aTagID, new TIFFValueType(TIFFValueTypes.Byte), (uint)aValue.Length)
        {
            mValue = aValue;
        }

        public ByteImageFileDirectoryEntry(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			if (mFieldType.SizeInBytes * mValueCount <= 4)
			{
				unsafe
				{
					fixed (uint* ptr = &mOffset)
					{
						byte* ptrUS = (byte*)ptr;
						mValue = new byte[mFieldType.SizeInBytes * mValueCount];
						for (int i = 0; i < mFieldType.SizeInBytes * mValueCount; i++)
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

	public abstract class SByteImageFileDirectoryEntry : ImageFileDirectoryEntry
	{
		protected sbyte[] mValue;
        public SByteImageFileDirectoryEntry(sbyte[] aValue, ushort aTagID)
            : base(aTagID, new TIFFValueType(TIFFValueTypes.SignedByte), (uint)aValue.Length)
        {
            mValue = aValue;
        }

        public SByteImageFileDirectoryEntry(FileReader aPefFile, ushort aTagID)
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

	public abstract class UShortImageFileDirectoryEntry : ImageFileDirectoryEntry
	{
		protected ushort[] mValue;
        public UShortImageFileDirectoryEntry(ushort[] aValue, ushort aTagID)
            : base(aTagID, new TIFFValueType(TIFFValueTypes.UnsignedShort), (uint)aValue.Length)
        {
            mValue = aValue;
        }

        public UShortImageFileDirectoryEntry(FileReader aPefFile, ushort aTagID)
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

	public abstract class ShortImageFileDirectoryEntry : ImageFileDirectoryEntry
	{
		protected short[] mValue;
        public ShortImageFileDirectoryEntry(short[] aValue, ushort aTagID)
            : base(aTagID, new TIFFValueType(TIFFValueTypes.SignedShort), (uint)aValue.Length)
        {
            mValue = aValue;
        }

        public ShortImageFileDirectoryEntry(FileReader aPefFile, ushort aTagID)
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

	public abstract class IntImageFileDirectoryEntry : ImageFileDirectoryEntry
	{
		protected int[] mValue;
        public IntImageFileDirectoryEntry(int[] aValue, ushort aTagID)
            : base(aTagID, new TIFFValueType(TIFFValueTypes.SignedLong), (uint)aValue.Length)
        { 
            mValue = aValue; 
        }

        public IntImageFileDirectoryEntry(FileReader aPefFile, ushort aTagID)
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

	public abstract class UIntImageFileDirectoryEntry : ImageFileDirectoryEntry
	{
		protected uint[] mValue;
        public UIntImageFileDirectoryEntry(uint[] aValue, ushort aTagID)
            : base(aTagID, new TIFFValueType(TIFFValueTypes.UnsignedLong), (uint)aValue.Length)
        {
            mValue = aValue;
        }

        public UIntImageFileDirectoryEntry(FileReader aPefFile, ushort aTagID)
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

	public abstract class RationalImageFileDirectoryEntry : ImageFileDirectoryEntry
	{
		protected Rational[] mValue;
        public RationalImageFileDirectoryEntry(Rational[] aValue, ushort aTagID)
            : base(aTagID, new TIFFValueType(TIFFValueTypes.Rational), (uint)aValue.Length)
        {
            mValue = aValue;
        }

        public RationalImageFileDirectoryEntry(FileReader aPefFile, ushort aTagID)
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

	public abstract class SRationalImageFileDirectoryEntry : ImageFileDirectoryEntry
	{
		protected SRational[] mValue;
        public SRationalImageFileDirectoryEntry(SRational[] aValue, ushort aTagID)
            : base(aTagID, new TIFFValueType(TIFFValueTypes.SignedRational), (uint)aValue.Length)
        {
            mValue = aValue;
        }

        public SRationalImageFileDirectoryEntry(FileReader aPefFile, ushort aTagID)
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

	public abstract class FloatImageFileDirectoryEntry : ImageFileDirectoryEntry
	{
		protected float[] mValue;
        public FloatImageFileDirectoryEntry(float[] aValue, ushort aTagID)
            : base(aTagID, new TIFFValueType(TIFFValueTypes.Float), (uint)aValue.Length)
        {
            mValue = aValue;
        }

        public FloatImageFileDirectoryEntry(FileReader aPefFile, ushort aTagID)
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

	public abstract class DoubleImageFileDirectoryEntry : ImageFileDirectoryEntry
	{
		protected double[] mValue;
        public DoubleImageFileDirectoryEntry(double[] aValue, ushort aTagID)
            : base(aTagID, new TIFFValueType(TIFFValueTypes.Double), (uint)aValue.Length)
        {
            mValue = aValue;
        }

        public DoubleImageFileDirectoryEntry(FileReader aPefFile, ushort aTagID)
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

    public abstract class UShortOrULongFileDirectoryEntry : ImageFileDirectoryEntry
    {
        protected uint[] mValue;

        public UShortOrULongFileDirectoryEntry(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        {
            if (mFieldType.SizeInBytes == 2)
            {
                if (mFieldType.SizeInBytes * mValueCount <= 4)
                {
                    unsafe
                    {
                        fixed (uint* ptr = &mOffset)
                        {
                            ushort* ptrUS = (ushort*)ptr;
                            mValue = new uint[mValueCount];
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

                    mValue = new uint[mValueCount];
                    for (int i = 0; i < mValueCount; i++)
                    {
                        mValue[i] = mPefFile.ReadUI2();
                    }
                    mPefFile.Seek(currentOffset, System.IO.SeekOrigin.Begin);
                }
            }
            else
            if (mFieldType.SizeInBytes == 4)
            {
                if (mFieldType.SizeInBytes * mValueCount <= 4)
                {
                    unsafe
                    {
                        fixed (uint* ptr = &mOffset)
                        {
                            mValue = new uint[mValueCount];
                            for (int i = 0; i < mValueCount; i++)
                            {
                                mValue[i] = ptr[i];
                            }
                        }
                    }
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
        }

        public UShortOrULongFileDirectoryEntry(uint[] aValue, ushort aTagID)
            : base(aTagID, new TIFFValueType(TIFFValueTypes.UnsignedLong), (uint)aValue.Length)
        {
            mValue = aValue;
        }

        public override string ToString()
        {
            return mValue[0].ToString() + " + " + (mValue.Length - 1).ToString() + " more...";
        }

        public override void SavePass1(Stream stream)
        {
            if (mValue.Length <= 1)
            {                
                BinaryWriter br = new BinaryWriter(stream, Encoding.ASCII, true);
                br.Write(mTagID);
                br.Write((ushort)TIFFValueTypes.UnsignedLong);
                br.Write(mValue.Length);
                br.Write(mValue[0]);
                br.Dispose();
                mOffsetInStream = 0;                
            }
            else
            {
                BinaryWriter br = new BinaryWriter(stream, Encoding.ASCII, true);
                br.Write(mTagID);
                br.Write((ushort)TIFFValueTypes.UnsignedLong);
                br.Write(mValue.Length);
                mOffsetInStream = (uint)stream.Position;
                br.Write(0);
                br.Dispose();
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

    public abstract class UShortOrULongOrRationalFileDirectoryEntry : ImageFileDirectoryEntry
    {
        protected Rational[] mValue;

        public UShortOrULongOrRationalFileDirectoryEntry(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        {
            if (mFieldType.SizeInBytes == 2)
            {
                if (mFieldType.SizeInBytes * mValueCount <= 4)
                {
                    unsafe
                    {
                        fixed (uint* ptr = &mOffset)
                        {
                            ushort* ptrUS = (ushort*)ptr;
                            mValue = new Rational[mValueCount];
                            for (int i = 0; i < mValueCount; i++)
                            {
                                if (aPefFile.EndianSwap)
                                    mValue[i] = new Rational(ptrUS[4 / mFieldType.SizeInBytes - i - 1], 1);
                                else
                                    mValue[i] = new Rational(ptrUS[i], 1);
                            }
                        }
                    }
                }
                else
                {
                    uint currentOffset = mPefFile.Position();
                    mPefFile.Seek(mOffset, System.IO.SeekOrigin.Begin);

                    mValue = new Rational[mValueCount];
                    for (int i = 0; i < mValueCount; i++)
                    {
                        mValue[i] = new Rational(mPefFile.ReadUI2(), 1);
                    }
                    mPefFile.Seek(currentOffset, System.IO.SeekOrigin.Begin);
                }
            }
            else
            if (mFieldType.SizeInBytes == 4)
            {
                if (mFieldType.SizeInBytes * mValueCount <= 4)
                {
                    unsafe
                    {
                        fixed (uint* ptr = &mOffset)
                        {
                            mValue = new Rational[mValueCount];
                            for (int i = 0; i < mValueCount; i++)
                            {
                                mValue[i] = new Rational(ptr[i], 1);
                            }
                        }
                    }
                }
                else
                {
                    uint currentOffset = mPefFile.Position();
                    mPefFile.Seek(mOffset, System.IO.SeekOrigin.Begin);

                    mValue = new Rational[mValueCount];
                    for (int i = 0; i < mValueCount; i++)
                    {
                        mValue[i] = new Rational(mPefFile.ReadUI4(), 1);
                    }
                    mPefFile.Seek(currentOffset, System.IO.SeekOrigin.Begin);
                }
            }
            else
            {
                uint currentOffset = mPefFile.Position();
                mPefFile.Seek(mOffset, System.IO.SeekOrigin.Begin);

                mValue = new Rational[mValueCount];
                for (int i = 0; i < mValueCount; i++)
                {
                    mValue[i].numerator = mPefFile.ReadUI4();
                    mValue[i].denominator = mPefFile.ReadUI4();
                }
                mPefFile.Seek(currentOffset, System.IO.SeekOrigin.Begin);
            }
        }

        public UShortOrULongOrRationalFileDirectoryEntry(Rational[] aValue, ushort aTagID)
            : base(aTagID, new TIFFValueType(TIFFValueTypes.Rational), (uint)aValue.Length)
        {
            mValue = aValue;
        }

        public Rational[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return mValue[0].ToString() + " + " + (mValue.Length - 1).ToString() + " more...";
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
    
    public abstract class OpcodeListFileDirectoryEntry : ImageFileDirectoryEntry
    {
        protected List<OpCode> mValue;

        public OpcodeListFileDirectoryEntry(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        {
            uint currentOffset = mPefFile.Position();
            if (mOffset == 0 || mValueCount == 0) //no entries
            {
                mValue = new List<OpCode>();
                return;
            }
            mPefFile.Seek(mOffset, System.IO.SeekOrigin.Begin);

            uint count = mPefFile.ReadUI4BE();
            mValue = new List<OpCode>();

            for (int op = 0; op < count; op++)
            {
                mValue.Add(OpCode.Create(aPefFile));                
            }

            mPefFile.Seek(currentOffset, System.IO.SeekOrigin.Begin);
        }

        public OpcodeListFileDirectoryEntry(List<OpCode> aValue, ushort aTagID)
            : base(aTagID, new TIFFValueType(TIFFValueTypes.Undefined), (uint)aValue.Count)
        { 
        }

        public override void SavePass1(Stream stream)
        {
            BinaryWriter br = new BinaryWriter(stream, Encoding.ASCII, true);
            br.Write(mTagID);
            br.Write(mFieldType.GetValue());
            br.Write(mValueCount);
            mOffsetInStream = (uint)stream.Position;
            if (mValue.Count == 0)
            {
                mOffsetInStream = 0;
            }
            br.Write(0);
            br.Dispose();
        }
        public override void SavePass2(Stream stream)
        {
            if (mOffsetInStream != 0)
            {
                stream.Seek(0, SeekOrigin.End);
                uint offset = (uint)stream.Position;
                
                uint count = (uint)mValue.Count;
                byte[] countBytes = BitConverter.GetBytes(count);

                if (BitConverter.IsLittleEndian)
                {
                    for (int i = 3; i >= 0; i--)
                    {
                        stream.Write(countBytes, i, 1);
                    }
                }
                else
                {
                    stream.Write(countBytes, 0, 4);
                }

                for (int i = 0; i < mValue.Count; i++)
                {
                    byte[] data = mValue[i].GetAsBytes();
                    stream.Write(data, 0, data.Length);
                }
                stream.Seek(mOffsetInStream, SeekOrigin.Begin);
                byte[] bytes = BitConverter.GetBytes(offset);
                stream.Write(bytes, 0, 4);
                stream.Seek(0, SeekOrigin.End);
            }
        }
    }
    #endregion

    #region Baseline TIFF Tags
    public class IFDUnknownTag : ByteImageFileDirectoryEntry
    {
        public IFDUnknownTag(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        {

        }

        public IFDUnknownTag(byte[] aValue, ushort aTagID, TIFFValueType aValueType)
            : base(aValue, aTagID)
        {
            mFieldType = aValueType;
        }

        public override string ToString()
        {
            return "Unknown IFD entry. ID: " + mTagID;
        }
    }


    public class IFDArtist : StringImageFileDirectoryEntry
	{
		public IFDArtist(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

        public IFDArtist(string aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 315;

		public const string TagName = "Artist";

		public string Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue;
		}
	}

	public class IFDBitsPerSample : UShortImageFileDirectoryEntry
	{
		public IFDBitsPerSample(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }
        public IFDBitsPerSample(ushort[] aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 258;

		public const string TagName = "Bits per sample";

		public ushort[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class IFDCellLength : UShortImageFileDirectoryEntry
	{
		public IFDCellLength(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }
        public IFDCellLength(ushort aValue)
            : base(new ushort[] { aValue }, TagID)
        { }

        public const ushort TagID = 265;

		public const string TagName = "Cell length";

		public ushort Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class IFDCellWidth : UShortImageFileDirectoryEntry
	{
		public IFDCellWidth(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }
        public IFDCellWidth(ushort aValue)
            : base(new ushort[] { aValue }, TagID)
        { }

        public const ushort TagID = 264;

		public const string TagName = "Cell width";

		public ushort Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class IFDColorMap : UShortImageFileDirectoryEntry
	{
		public IFDColorMap(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }
        public IFDColorMap(ushort[] aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 320;

		public const string TagName = "Color map";

		public ushort[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue.Length.ToString() + " entries.";
		}
	}

	public class IFDCompression : UShortImageFileDirectoryEntry
	{
		public enum Compression : ushort
		{ 
			NoCompression = 1,
			CCITTGroup3 = 2,
            LosslessJPEG = 7,
            Deflate = 8,
			PackBits = 32773,
			Pentax = 65535
		}

		public IFDCompression(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }
        public IFDCompression(Compression aValue)
            : base(new ushort[] { (ushort)aValue }, TagID)
        { }

        public const ushort TagID = 259;

		public const string TagName = "Compression";

		public Compression Value
		{
			get { return (Compression)mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class IFDCopyright : StringImageFileDirectoryEntry
	{
		public IFDCopyright(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }
        public IFDCopyright(string aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 33432;

		public const string TagName = "Copyright";

		public string Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue;
		}
	}

	public class IFDDateTime : StringImageFileDirectoryEntry
	{
		DateTime dt_value;

		public IFDDateTime(FileReader aPefFile, ushort aTagID)
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
        public IFDDateTime(DateTime aValue)
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

        public const ushort TagID = 306;

		public const string TagName = "Date/Time";

		public DateTime Value
		{
			get { return dt_value; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class IFDExtraSamples : UShortImageFileDirectoryEntry
	{
		public IFDExtraSamples(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }
        public IFDExtraSamples(ushort[] aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 338;

		public const string TagName = "Extra samples";

		public ushort[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue.Length.ToString() + " entries.";
		}
	}

	public class IFDFillOrder : UShortImageFileDirectoryEntry
	{
		public IFDFillOrder(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }
        public IFDFillOrder(ushort aValue)
            : base(new ushort[] { aValue }, TagID)
        { }

        public const ushort TagID = 226;

		public const string TagName = "Fill order";

		public ushort Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue[0].ToString();
		}
	}

	public class IFDFreeByteCounts : UIntImageFileDirectoryEntry
	{
		public IFDFreeByteCounts(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }
        public IFDFreeByteCounts(uint aValue)
            : base(new uint[] { aValue }, TagID)
        { }

        public const ushort TagID = 289;

		public const string TagName = "Free byte counts";

		public uint Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue[0].ToString();
		}
	}

	public class IFDFreeOffsets : UIntImageFileDirectoryEntry
	{
		public IFDFreeOffsets(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }
        public IFDFreeOffsets(uint aValue)
            : base(new uint[] { aValue }, TagID)
        { }

        public const ushort TagID = 288;

		public const string TagName = "Free offsets";

		public uint Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue[0].ToString();
		}
	}

	public class IFDGrayResponseCurve : UShortImageFileDirectoryEntry
	{
		public IFDGrayResponseCurve(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }
        public IFDGrayResponseCurve(ushort[] aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 291;

		public const string TagName = "Gray response curve";

		public ushort[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue.Length.ToString() + " entries.";
		}
	}

	public class IFDGrayResponseUnit : UShortImageFileDirectoryEntry
	{
		public IFDGrayResponseUnit(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }
        public IFDGrayResponseUnit(ushort aValue)
            : base(new ushort[] { aValue }, TagID)
        { }

        public const ushort TagID = 290;

		public const string TagName = "Gray response unit";

		public ushort Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue[0].ToString();
		}
	}

	public class IFDHostComputer : StringImageFileDirectoryEntry
	{
		public IFDHostComputer(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }
        public IFDHostComputer(string aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 316;

		public const string TagName = "Host computer";

		public string Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue;
		}
	}

	public class IFDImageDescription : StringImageFileDirectoryEntry
	{
		public IFDImageDescription(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }
        public IFDImageDescription(string aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 270;

		public const string TagName = "Image description";

		public string Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue;
		}
	}

	public class IFDImageLength : ImageFileDirectoryEntry
	{
		uint mValue;

		public IFDImageLength(FileReader aPefFile, ushort aTagID)
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

        public IFDImageLength(uint aValue)
            : base(TagID, new TIFFValueType(TIFFValueTypes.UnsignedLong), 1)
        {
            mValue = aValue;
        }

        public const ushort TagID = 257;

		public const string TagName = "Image length";

		public uint Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue.ToString();
        }

        public override void SavePass1(Stream stream)
        {
            BinaryWriter br = new BinaryWriter(stream, Encoding.ASCII, true);
            br.Write(mTagID);
            br.Write((ushort)TIFFValueTypes.UnsignedLong);
            br.Write(1);
            
            br.Write(mValue);
            br.Dispose();
        }
    }

	public class IFDImageWidth : ImageFileDirectoryEntry
	{
		uint mValue;

		public IFDImageWidth(FileReader aPefFile, ushort aTagID)
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

        public IFDImageWidth(uint aValue)
            : base(TagID, new TIFFValueType(TIFFValueTypes.UnsignedLong), 1)
        {
            mValue = aValue;
        }

        public const ushort TagID = 256;

		public const string TagName = "Image width";

		public uint Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue.ToString();
        }

        public override void SavePass1(Stream stream)
        {
            BinaryWriter br = new BinaryWriter(stream, Encoding.ASCII, true);
            br.Write(mTagID);
            br.Write((ushort)TIFFValueTypes.UnsignedLong);
            br.Write(1);

            br.Write(mValue);
            br.Dispose();
        }
    }

	public class IFDMake : StringImageFileDirectoryEntry
	{
		public IFDMake(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }
        public IFDMake(string aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 271;

		public const string TagName = "Make";

		public string Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue;
		}
	}

	public class IFDMaxSampleValue : UShortImageFileDirectoryEntry
	{
		public IFDMaxSampleValue(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }
        public IFDMaxSampleValue(ushort[] aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 281;

		public const string TagName = "Max sample value";

		public ushort[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue.Length.ToString() + " entries.";
		}
	}

	public class IFDMinSampleValue : UShortImageFileDirectoryEntry
	{
		public IFDMinSampleValue(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }
        public IFDMinSampleValue(ushort[] aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 280;

		public const string TagName = "Min sample value";

		public ushort[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue.Length.ToString() + " entries.";
		}
	}

	public class IFDModel : StringImageFileDirectoryEntry
	{
		public IFDModel(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }
        public IFDModel(string aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 272;

		public const string TagName = "Model";

		public string Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue;
		}
	}

	public class IFDNewSubfileType : UIntImageFileDirectoryEntry
	{
		public IFDNewSubfileType(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }
        public IFDNewSubfileType(uint aValue)
            : base(new uint[] { aValue }, TagID)
        { }

        public const ushort TagID = 254;

		public const string TagName = "New subfile type";

		public uint Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue[0].ToString();
		}
	}

	public class IFDOrientation : UShortImageFileDirectoryEntry
	{
		public IFDOrientation(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }
        public IFDOrientation(ushort aValue)
            : base(new ushort[] { aValue }, TagID)
        { }

        public const ushort TagID = 274;

		public const string TagName = "Orientation";

		public ushort Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue[0].ToString();
		}
	}

	public class IFDPhotometricInterpretation : UShortImageFileDirectoryEntry
	{
		public enum PhotometricInterpretation : ushort
		{ 
			WhiteIsZero = 0,
			BlackIsZero = 1,
			RGB = 2,
			Palette = 3,
			TransparencyMask = 4,
			CFA = 32803		
		}

		public IFDPhotometricInterpretation(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }
        public IFDPhotometricInterpretation(PhotometricInterpretation aValue)
            : base(new ushort[] { (ushort)aValue }, TagID)
        { }

        public const ushort TagID = 262;

		public const string TagName = "Photometric interpretation";

		public PhotometricInterpretation Value
		{
			get { return (PhotometricInterpretation)mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class IFDPlanarConfiguration : UShortImageFileDirectoryEntry
	{
		public enum PlanarConfigurartion : ushort
		{ 
			Chunky = 1,
			Planar = 2
		}

		public IFDPlanarConfiguration(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }
        public IFDPlanarConfiguration(PlanarConfigurartion aValue)
            : base(new ushort[] { (ushort)aValue }, TagID)
        { }

        public const ushort TagID = 284;

		public const string TagName = "Planar configuration";

		public PlanarConfigurartion Value
		{
			get { return (PlanarConfigurartion)mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class IFDResolutionUnit : UShortImageFileDirectoryEntry
	{
		public enum ResolutionUnit
		{ 
			None = 1,
			Inch = 2,
			Centimeter = 3
		}

		public IFDResolutionUnit(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }
        public IFDResolutionUnit(ResolutionUnit aValue)
            : base(new ushort[] { (ushort)aValue }, TagID)
        { }

        public const ushort TagID = 296;

		public const string TagName = "Resolution unit";

		public ResolutionUnit Value
		{
			get { return (ResolutionUnit)mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class IFDRowsPerStrip : ImageFileDirectoryEntry
	{
		uint mValue;

		public IFDRowsPerStrip(FileReader aPefFile, ushort aTagID)
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

        public IFDRowsPerStrip(uint aValue)
            : base(TagID, new TIFFValueType(TIFFValueTypes.UnsignedLong), 1)
        {
            mValue = aValue;
        }

        public const ushort TagID = 278;

		public const string TagName = "Rows per strip";

		public uint Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue.ToString();
        }

        public override void SavePass1(Stream stream)
        {
            BinaryWriter br = new BinaryWriter(stream, Encoding.ASCII, true);
            br.Write(mTagID);
            br.Write((ushort)TIFFValueTypes.UnsignedLong);
            br.Write(1);

            br.Write(mValue);
            br.Dispose();
        }
    }

	public class IFDSamplesPerPixel : UShortImageFileDirectoryEntry
	{
		public IFDSamplesPerPixel(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }
        public IFDSamplesPerPixel(ushort aValue)
            : base(new ushort[] { aValue }, TagID)
        { }

        public const ushort TagID = 277;

		public const string TagName = "Samples per pixel";

		public ushort Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue[0].ToString();
		}
	}

	public class IFDSoftware : StringImageFileDirectoryEntry
	{
		public IFDSoftware(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }
        public IFDSoftware(string aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 305;

		public const string TagName = "Software";

		public string Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue;
		}
	}

	public class IFDStripByteCounts : UShortOrULongFileDirectoryEntry
	{
		public IFDStripByteCounts(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
        }

        public IFDStripByteCounts(uint[] aValue)
            : base(aValue, TagID)
        {
        }

        public const ushort TagID = 279;

		public const string TagName = "Strip byte counts";

        public uint[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
		{
			return TagName + ": " + mValue.Length.ToString() + " entries.";
        }
    }

	public class IFDStripOffsets : UShortOrULongFileDirectoryEntry
    {
		public IFDStripOffsets(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
        }

        public IFDStripOffsets(uint[] aValue)
            : base(aValue, TagID)
        {
        }

        public const ushort TagID = 273;

		public const string TagName = "Strip offsets";
        
        public uint[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
		{
			return TagName + ": " + mValue.Length.ToString() + " entries.";
        }

        public override void SavePass1(Stream stream)
        {
            if (mValue.Length <= 1)
            {
                BinaryWriter br = new BinaryWriter(stream, Encoding.ASCII, true);
                br.Write(mTagID);
                br.Write((ushort)TIFFValueTypes.UnsignedLong);
                br.Write(mValue.Length);
                mOffsetInStream = (uint)stream.Position;
                br.Write(mValue[0]);
                br.Dispose();
            }
            else
            {
                BinaryWriter br = new BinaryWriter(stream, Encoding.ASCII, true);
                br.Write(mTagID);
                br.Write((ushort)TIFFValueTypes.UnsignedLong);
                br.Write(mValue.Length);
                mOffsetInStream = (uint)stream.Position;
                br.Write(0);
                br.Dispose();
            }
        }
        public override void SavePass2(Stream stream)
        {
            if (mValue.Length == 1)
                return;

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

        public Stream SaveFinalOffsets(Stream stream, uint[] finalOffsets)
        {
            if (mValue.Length != finalOffsets.Length)
                throw new ArgumentException("Length of offset array isn't the same!");

            BinaryWriter br = new BinaryWriter(stream, Encoding.ASCII, true);

            br.BaseStream.Seek(mOffsetInStream, SeekOrigin.Begin);
            for (int i = 0; i < finalOffsets.Length; i++)
            {
                br.Write(finalOffsets[i]);
            }
            br.BaseStream.Seek(0, SeekOrigin.End);
            br.Dispose();
            return stream;
        }
    }

	public class IFDSubfileType : UShortImageFileDirectoryEntry
	{
		public IFDSubfileType(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }
        public IFDSubfileType(ushort aValue)
            : base(new ushort[] { aValue }, TagID)
        { }

        public const ushort TagID = 255;

		public const string TagName = "Subfile type";

		public ushort Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue[0].ToString();
		}
	}

	public class IFDThreshholding : UShortImageFileDirectoryEntry
	{
		public IFDThreshholding(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }
        public IFDThreshholding(ushort aValue)
            : base(new ushort[] { aValue }, TagID)
        { }

        public const ushort TagID = 263;

		public const string TagName = "Threshholding";

		public ushort Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return mValue[0].ToString();
		}
	}

	public class IFDXResolution : RationalImageFileDirectoryEntry
	{
		public IFDXResolution(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }
        public IFDXResolution(Rational aValue)
            : base(new Rational[] { aValue }, TagID)
        { }

        public const ushort TagID = 282;

		public const string TagName = "X-Resolution";

		public Rational Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue[0].ToString();
		}
	}

	public class IFDYResolution : RationalImageFileDirectoryEntry
	{
		public IFDYResolution(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }
        public IFDYResolution(Rational aValue)
            : base(new Rational[] { aValue }, TagID)
        { }

        public const ushort TagID = 283;

		public const string TagName = "Y-Resolution";

		public Rational Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue[0].ToString();
		}
	}

	public class IFDExif : UIntImageFileDirectoryEntry
	{
		List<ExifEntry> mExif;

		public IFDExif(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			mExif = new List<ExifEntry>();
			uint currPos = mPefFile.Position();
			mPefFile.Seek(mValue[0], System.IO.SeekOrigin.Begin);

			ushort entryCount = mPefFile.ReadUI2();
			for (ushort i = 0; i < entryCount; i++)
			{
				ExifEntry entry = ExifEntry.CreateExifEntry(mPefFile);
				mExif.Add(entry);
			}
			mPefFile.Seek(currPos, System.IO.SeekOrigin.Begin);
		}

        public IFDExif(List<ExifEntry> aValue)
            :base(new uint[] { 0 }, TagID)
        {
            mExif = aValue;
        }

		public const ushort TagID = 34665;

		public const string TagName = "Exif";

		public List<ExifEntry> Value
		{
			get { return mExif; }
		}

		public T GetEntry<T>() where T : ExifEntry
		{
			Type t = typeof(T);
			foreach (var item in mExif)
			{
				if (item is T)
					return (T)item;
			}
			return null;
		}

		public override string ToString()
		{
			return TagName + ": " + Value.Count.ToString() + " entries";
        }

        public override void SavePass1(Stream stream)
        {
            BinaryWriter br = new BinaryWriter(stream, Encoding.ASCII, true);
            br.Write(mTagID);
            br.Write(mFieldType.GetValue());
            br.Write(mValueCount);
            mOffsetInStream = (uint)stream.Position;
            br.Write(0);
            br.Dispose();
        }
        public override void SavePass2(Stream stream)
        {
            stream.Seek(0, SeekOrigin.End);
            uint offset = (uint)stream.Position;



            byte[] count = BitConverter.GetBytes((ushort)mExif.Count);
            stream.Write(count, 0, 2);
            foreach (var item in mExif)
            {
                item.SavePass1(stream);
            }
            stream.Write(new byte[] { 0, 0, 0, 0 }, 0, 4);
            foreach (var item in mExif)
            {
                item.SavePass2(stream);
            }


            stream.Seek(mOffsetInStream, SeekOrigin.Begin);
            byte[] bytes = BitConverter.GetBytes(offset);
            stream.Write(bytes, 0, 4);
            stream.Seek(0, SeekOrigin.End);            
        }
    }

	public class IFDGps : UIntImageFileDirectoryEntry
	{
		List<GPSDirectoryEntry> mGPS;

		public IFDGps(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			mGPS = new List<GPSDirectoryEntry>();
			uint currPos = mPefFile.Position();
			mPefFile.Seek(mValue[0], System.IO.SeekOrigin.Begin);
			
			ushort entryCount = mPefFile.ReadUI2();
			for (ushort i = 0; i < entryCount; i++)
			{
				GPSDirectoryEntry entry = GPSDirectoryEntry.CreateGPSDirectoryEntry(mPefFile);
				mGPS.Add(entry);
			}
			mPefFile.Seek(currPos, System.IO.SeekOrigin.Begin);
        }

        public IFDGps(List<GPSDirectoryEntry> aValue)
            : base(new uint[] { 0 }, TagID)
        {
            mGPS = aValue;
        }

        public const ushort TagID = 34853;

		public const string TagName = "GPS";

		public List<GPSDirectoryEntry> Value
		{
			get { return mGPS; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.Count.ToString() + " entries";
        }

        public override void SavePass1(Stream stream)
        {
            BinaryWriter br = new BinaryWriter(stream, Encoding.ASCII, true);
            br.Write(mTagID);
            br.Write(mFieldType.GetValue());
            br.Write(mValueCount);
            mOffsetInStream = (uint)stream.Position;
            br.Write(0);
            br.Dispose();
        }
        public override void SavePass2(Stream stream)
        {
            stream.Seek(0, SeekOrigin.End);
            uint offset = (uint)stream.Position;



            byte[] count = BitConverter.GetBytes((ushort)mGPS.Count);
            stream.Write(count, 0, 2);
            foreach (var item in mGPS)
            {
                item.SavePass1(stream);
            }
            stream.Write(new byte[] { 0, 0, 0, 0 }, 0, 4);
            foreach (var item in mGPS)
            {
                item.SavePass2(stream);
            }


            stream.Seek(mOffsetInStream, SeekOrigin.Begin);
            byte[] bytes = BitConverter.GetBytes(offset);
            stream.Write(bytes, 0, 4);
            stream.Seek(0, SeekOrigin.End);
        }
    }

	public class IFDJPEGInterchangeFormat : UIntImageFileDirectoryEntry
	{
		public IFDJPEGInterchangeFormat(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }
        public IFDJPEGInterchangeFormat(uint aValue)
            : base(new uint[] { aValue }, TagID)
        { }

        public const ushort TagID = 513;

		public const string TagName = "JPEG Interchange format";

		public uint Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue[0].ToString();
		}
	}

	public class IFDJPEGInterchangeFormatLength : UIntImageFileDirectoryEntry
	{
		public IFDJPEGInterchangeFormatLength(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }
        public IFDJPEGInterchangeFormatLength(uint aValue)
            : base(new uint[] { aValue }, TagID)
        { }

        public const ushort TagID = 514;

		public const string TagName = "JPEG Interchange format length";

		public uint Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue[0].ToString();
		}
	}


    public class IFDDocumentName : StringImageFileDirectoryEntry
    {
        public IFDDocumentName(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDocumentName(string aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 269;

        public const string TagName = "Document Name";

        public string Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue;
        }
    }

    public class IFDSampleFormat : UShortImageFileDirectoryEntry
    {
        public enum SampleFormat : ushort
        {
            UINT = 1,
            INT = 2,
            IEEEFP = 3,
            VOID = 4,
            COMPLEXINT = 5,
            COMPLEXIEEEFP = 6,
        }

        public IFDSampleFormat(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDSampleFormat(SampleFormat aValue)
            : base(new ushort[] { (ushort)aValue }, TagID)
        { }

        public const ushort TagID = 339;

        public const string TagName = "Sample Format";

        public SampleFormat Value
        {
            get { return (SampleFormat)mValue[0]; }
        }

        public override string ToString()
        {
            return TagName + ": " + ((SampleFormat)mValue[0]).ToString();
        }
    }

    public class IFDExposureTime : RationalImageFileDirectoryEntry
    {
        public IFDExposureTime(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDExposureTime(Rational aValue)
            : base(new Rational[] { aValue }, TagID)
        { }

        public const ushort TagID = 33434;

        public const string TagName = "Exposure Time";

        public Rational Value
        {
            get { return mValue[0]; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString();
        }
    }

    public class IFDFNumber : RationalImageFileDirectoryEntry
    {
        public IFDFNumber(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDFNumber(Rational aValue)
            : base(new Rational[] { aValue }, TagID)
        { }

        public const ushort TagID = 33437;

        public const string TagName = "FNumber";

        public Rational Value
        {
            get { return mValue[0]; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString();
        }
    }

    public class IFDISOSpeedRatings : UShortImageFileDirectoryEntry
    {
        public IFDISOSpeedRatings(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDISOSpeedRatings(ushort aValue)
            : base(new ushort[] { aValue }, TagID)
        { }

        public const ushort TagID = 34855;

        public const string TagName = "ISO Speed Ratings";

        public ushort Value
        {
            get { return mValue[0]; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString();
        }
    }

    public class IFDDateTimeOriginal : StringImageFileDirectoryEntry
    {
        DateTime dt_value;

        public IFDDateTimeOriginal(FileReader aPefFile, ushort aTagID)
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
        public IFDDateTimeOriginal(DateTime aValue)
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

        public const string TagName = "Date/Time Original";

        public DateTime Value
        {
            get { return dt_value; }
        }

        public override string ToString()
        {
            return TagName + ": " + Value.ToString();
        }
    }

    public class IFDFocalLength : RationalImageFileDirectoryEntry
    {
        public IFDFocalLength(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDFocalLength(Rational aValue)
            : base(new Rational[] { aValue }, TagID)
        { }

        public const ushort TagID = 37386;

        public const string TagName = "Focal Length";

        public Rational Value
        {
            get { return mValue[0]; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString();
        }
    }
    #endregion

    #region DNG Tags
    public class IFDDNGVersion : ByteImageFileDirectoryEntry
    {
        public IFDDNGVersion(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public IFDDNGVersion(byte v0, byte v1, byte v2, byte v3)
            : base(new byte[] {v0, v1, v2, v3 }, TagID)
        { }

        public const ushort TagID = 50706;

        public const string TagName = "DNGVersion";

        public byte[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + Value[0].ToString() + Value[1].ToString() + Value[2].ToString() + Value[3].ToString();
        }
    }

    public class IFDDNGBackwardVersion : ByteImageFileDirectoryEntry
    {
        public IFDDNGBackwardVersion(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public IFDDNGBackwardVersion(byte v0, byte v1, byte v2, byte v3)
            : base(new byte[] { v0, v1, v2, v3 }, TagID)
        { }

        public const ushort TagID = 50707;

        public const string TagName = "DNGBackwardVersion";

        public byte[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + Value[0].ToString() + Value[1].ToString() + Value[2].ToString() + Value[3].ToString();
        }
    }

    public class IFDDNGUniqueCameraModel : StringImageFileDirectoryEntry
    {
        public IFDDNGUniqueCameraModel(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGUniqueCameraModel(string aUniqueCameraModel)
            : base(aUniqueCameraModel, TagID)
        { }

        public const ushort TagID = 50708;

        public const string TagName = "Unique Camera Model";

        public string Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue;
        }
    }

    public class IFDDNGLocalizedCameraModel : StringImageFileDirectoryEntry
    {
        public IFDDNGLocalizedCameraModel(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGLocalizedCameraModel(string aLocalizedCameraModel)
            : base(aLocalizedCameraModel, TagID)
        { }

        public const ushort TagID = 50709;

        public const string TagName = "Localized Camera Model";

        public string Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue;
        }
    }

    public class IFDDNGCFAPlaneColor : ByteImageFileDirectoryEntry
    {
        public IFDDNGCFAPlaneColor(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGCFAPlaneColor(byte[] aCFAPlaneColor)
            : base(aCFAPlaneColor, TagID)
        { }

        public const ushort TagID = 50710;

        public const string TagName = "CFA Plane Color";

        public byte[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            string ret = TagName + ": ";
            for (int i = 0; i < mValue.Length; i++)
            {
                ret += Value[i].ToString();
            }

            return ret;
        }
    }

    public class IFDDNGCFALayout : UShortImageFileDirectoryEntry
    {
        public IFDDNGCFALayout(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGCFALayout(ushort aCFALayout)
            : base(new ushort[] { aCFALayout }, TagID)
        { }

        public const ushort TagID = 50711;

        public const string TagName = "CFA Layout";

        public ushort Value
        {
            get { return mValue[0]; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString();
        }
    }

    public class IFDDNGLinearizationTable : UShortImageFileDirectoryEntry
    {
        public IFDDNGLinearizationTable(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGLinearizationTable(ushort[] aLinearizationTable)
            : base(aLinearizationTable, TagID)
        { }

        public const ushort TagID = 50712;

        public const string TagName = "Linearization Table";

        public ushort[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue.Length.ToString() + " entries.";
        }
    }

    public class IFDDNGBlackLevelRepeatDim : UShortImageFileDirectoryEntry
    {
        public IFDDNGBlackLevelRepeatDim(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGBlackLevelRepeatDim(ushort [] aBlackLevelRepeatDim)
            : base(aBlackLevelRepeatDim, TagID)
        { }

        public const ushort TagID = 50713;

        public const string TagName = "Black Level Repeat Dim";

        public ushort[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString() + " " + mValue[1].ToString();
        }
    }

    public class IFDTileWidth : UShortOrULongFileDirectoryEntry
    {
        public IFDTileWidth(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        {
        }

        public IFDTileWidth(uint aTileWidth)
            : base(new uint[] { aTileWidth }, TagID)
        { 
        }

        public const ushort TagID = 322;

        public const string TagName = "Tile Width";
        
        public uint Value
        {
            get { return mValue[0]; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString();
        }
    }

    public class IFDTileLength : UShortOrULongFileDirectoryEntry
    {
        public IFDTileLength(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        {

        }

        public IFDTileLength(uint aTileLength)
            : base(new uint[] { aTileLength }, TagID)
        {
        }

        public const ushort TagID = 323;

        public const string TagName = "Tile Length";

        public uint Value
        {
            get { return mValue[0]; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString();
        }
    }

    public class IFDTileByteCounts : UShortOrULongFileDirectoryEntry
    {
        public IFDTileByteCounts(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        {
        }
        public IFDTileByteCounts(uint[] aTileByteCounts)
            : base(aTileByteCounts, TagID)
        {
        }

        public const ushort TagID = 325;

        public const string TagName = "Tile Byte Counts";

        public uint[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString() + " + " + (mValue.Length - 1).ToString() + " more...";
        }
    }

    public class IFDTileOffsets : UIntImageFileDirectoryEntry
    {
        public IFDTileOffsets(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDTileOffsets(uint[] aTileOffsets)
            : base(aTileOffsets, TagID)
        { }

        public const ushort TagID = 324;

        public const string TagName = "Tile Offsets";

        public uint[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString() + " + " + (mValue.Length - 1).ToString() + " more...";
        }
        

        public override void SavePass1(Stream stream)
        {
            if (mValue.Length <= 1)
            {
                BinaryWriter br = new BinaryWriter(stream, Encoding.ASCII, true);
                br.Write(mTagID);
                br.Write((ushort)TIFFValueTypes.UnsignedLong);
                br.Write(mValue.Length);
                mOffsetInStream = (uint)stream.Position;
                br.Write(mValue[0]);
                br.Dispose();
            }
            else
            {
                BinaryWriter br = new BinaryWriter(stream, Encoding.ASCII, true);
                br.Write(mTagID);
                br.Write((ushort)TIFFValueTypes.UnsignedLong);
                br.Write(mValue.Length);
                mOffsetInStream = (uint)stream.Position;
                br.Write(0);
                br.Dispose();
            }
        }
        public override void SavePass2(Stream stream)
        {
            if (mValue.Length == 1)
                return;

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

        public Stream SaveFinalOffsets(Stream stream, uint[] finalOffsets)
        {
            if (mValue.Length != finalOffsets.Length)
                throw new ArgumentException("Length of offset array isn't the same!");

            BinaryWriter br = new BinaryWriter(stream, Encoding.ASCII, true);

            br.BaseStream.Seek(mOffsetInStream, SeekOrigin.Begin);
            for (int i = 0; i < finalOffsets.Length; i++)
            {
                br.Write(finalOffsets[i]);
            }
            br.BaseStream.Seek(0, SeekOrigin.End);
            br.Dispose();
            return stream;
        }
    }

    public class IFDSubIFDs : ImageFileDirectoryEntry
    {
        List<ImageFileDirectory> mValue;

        public IFDSubIFDs(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        {
            uint[] offsets = new uint[mValueCount];
            uint currentOffset;

            if (mFieldType.SizeInBytes == 2)
            {
                if (mFieldType.SizeInBytes * mValueCount <= 4)
                {
                    unsafe
                    {
                        fixed (uint* ptr = &mOffset)
                        {
                            ushort* ptrUS = (ushort*)ptr;
                            offsets = new uint[mValueCount];
                            for (int i = 0; i < mValueCount; i++)
                            {
                                if (aPefFile.EndianSwap)
                                    offsets[i] = ptrUS[4 / mFieldType.SizeInBytes - i - 1];
                                else
                                    offsets[i] = ptrUS[i];
                            }
                        }
                    }
                }
                else
                {
                    currentOffset = mPefFile.Position();
                    mPefFile.Seek(mOffset, System.IO.SeekOrigin.Begin);

                    offsets = new uint[mValueCount];
                    for (int i = 0; i < mValueCount; i++)
                    {
                        offsets[i] = mPefFile.ReadUI2();
                    }
                    mPefFile.Seek(currentOffset, System.IO.SeekOrigin.Begin);
                }
            }
            else
            {
                if (mFieldType.SizeInBytes * mValueCount <= 4)
                {
                    offsets = new uint[mValueCount];
                    offsets[0] = mOffset;
                }
                else
                {
                    currentOffset = mPefFile.Position();
                    mPefFile.Seek(mOffset, System.IO.SeekOrigin.Begin);

                    offsets = new uint[mValueCount];
                    for (int i = 0; i < mValueCount; i++)
                    {
                        offsets[i] = mPefFile.ReadUI4();
                    }
                    mPefFile.Seek(currentOffset, System.IO.SeekOrigin.Begin);
                }
            }


            currentOffset = mPefFile.Position();

            mValue = new List<ImageFileDirectory>();

            foreach (var item in offsets)
            {
                mPefFile.Seek(item, System.IO.SeekOrigin.Begin);
                ImageFileDirectory ifd = new ImageFileDirectory(aPefFile);
                mValue.Add(ifd);
            }

            mPefFile.Seek(currentOffset, System.IO.SeekOrigin.Begin);

        }
        public IFDSubIFDs(List<ImageFileDirectory> aSubIFDs)
            : base(TagID, new TIFFValueType(TIFFValueTypes.UnsignedLong), (uint)aSubIFDs.Count)
        { 
        }

        public const ushort TagID = 330;

        public const string TagName = "SubIFDs";

        public List<ImageFileDirectory> Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue.ToString();
        }


        public override void SavePass1(Stream stream)
        {
            BinaryWriter br = new BinaryWriter(stream, Encoding.ASCII, true);
            br.Write(mTagID);
            br.Write(mFieldType.GetValue());
            br.Write(mValueCount);
            mOffsetInStream = (uint)stream.Position;
            br.Write(0);
            br.Dispose();
        }
        public override void SavePass2(Stream stream)
        {
            stream.Seek(0, SeekOrigin.End);
            uint offset = (uint)stream.Position;

            if (mValueCount > 1)
            {
                uint[] offsets = new uint[mValueCount];

                stream.Write(new byte[mValueCount * 4], 0, (int)mValueCount * 4);

                for (int i = 0; i < mValueCount; i++)
                {
                    offsets[i] = (uint)stream.Position;
                    mValue[i].SavePass1(stream);
                }

                //stream.Write(new byte[] { 0, 0, 0, 0 }, 0, 4);
                foreach (var item in mValue)
                {
                    item.SavePass2(stream);
                }

                stream.Seek(offset, SeekOrigin.Begin);
                for (int i = 0; i < mValueCount; i++)
                {
                    byte[] ifd = BitConverter.GetBytes(offsets[i]);
                    stream.Write(ifd, 0, 4);
                }


                stream.Seek(mOffsetInStream, SeekOrigin.Begin);
                byte[] bytes = BitConverter.GetBytes(offset);
                stream.Write(bytes, 0, 4);
                stream.Seek(0, SeekOrigin.End);
            }
            else
            {
                offset = (uint)stream.Position;
                foreach (var item in mValue)
                {
                    item.SavePass1(stream);
                }
                foreach (var item in mValue)
                {
                    item.SavePass2(stream);
                }


                stream.Seek(mOffsetInStream, SeekOrigin.Begin);
                byte[] bytes = BitConverter.GetBytes(offset);
                stream.Write(bytes, 0, 4);
                stream.Seek(0, SeekOrigin.End);
            }

            
        }
    }

    public class IFDCFARepeatPatternDim : UShortImageFileDirectoryEntry
    {
        public IFDCFARepeatPatternDim(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDCFARepeatPatternDim(ushort[] aRepeatPatternDim)
            : base(aRepeatPatternDim, TagID)
        { }

        public const ushort TagID = 33421;

        public const string TagName = "CFA Repeat Pattern Dim";

        public ushort[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString() + ", " + mValue[1].ToString();
        }
    }

    public class IFDCFAPattern : ByteImageFileDirectoryEntry
    {
        public IFDCFAPattern(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDCFAPattern(ExifCFAPattern.BayerColor[] aCFAPattern)
            : base(new byte[aCFAPattern.Length], TagID)
        {
            for (int i = 0; i < aCFAPattern.Length; i++)
            {
                mValue[i] = (byte)aCFAPattern[i];
            }
        }

        public const ushort TagID = 33422;

        public const string TagName = "CFA Pattern";

        public ExifCFAPattern.BayerColor[] Value
        {
            get
            {
                ExifCFAPattern.BayerColor[] ret = new ExifCFAPattern.BayerColor[mValue.Length];
                for (int i = 0; i < mValue.Length; i++)
                {
                    ret[i] = (ExifCFAPattern.BayerColor)mValue[i];
                }
                return ret;
            }
        }

        public override string ToString()
        {
            string ret = TagName + ": ";

            for (int i = 0; i < mValue.Length; i++)
            {
                ret += mValue[i];
                if (i < mValue.Length - 1)
                    ret += ", ";
            }

            return ret;
        }
    }

    public class IFDDNGTimeZoneOffset : ShortImageFileDirectoryEntry
    {
        public IFDDNGTimeZoneOffset(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGTimeZoneOffset(short[] aTimeZoneOffset)
            : base(aTimeZoneOffset, TagID)
        { }

        public const ushort TagID = 34858;

        public const string TagName = "Time Zone Offset";

        public short[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString();
        }
    }

    public class IFDDNGImageNumber : IntImageFileDirectoryEntry
    {
        public IFDDNGImageNumber(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGImageNumber(int aImageNumber)
            : base(new int[] { aImageNumber }, TagID)
        { }

        public const ushort TagID = 37393;

        public const string TagName = "Image Number";

        public int Value
        {
            get { return mValue[0]; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString();
        }
    }

    public abstract class IFDDNGMatrixTag : SRationalImageFileDirectoryEntry
    {
        public IFDDNGMatrixTag(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGMatrixTag(SRational[] aMatrix, ushort aTagID)
            : base(aMatrix, aTagID)
        { }
        public IFDDNGMatrixTag(DNGMatrix aMatrix, ushort aTagID)
            : base(new SRational[aMatrix.Cols * aMatrix.Rows], aTagID)
        {
            for (uint row = 0; row < aMatrix.Rows; row++)
            {
                for (uint col = 0; col < 3; col++)
                {
                    int denominator = 1000000;
                    mValue[row * 3 + col] = new SRational(
                        (int)Math.Round(aMatrix[row, col] * denominator),
                        denominator);
                }
            }
        }

        public SRational[] Value
        {
            get { return mValue; }
        }

        public DNGMatrix Matrix
        {
            get 
            {
                uint colorPlanes = (uint)Value.Length / 3;
                DNGMatrix matrix = new DNGMatrix(colorPlanes, 3);
                for (uint c = 0; c < colorPlanes; c++)
                {
                    for (uint i = 0; i < 3; i++)
                    {
                        matrix[c, i] = Value[c * 3 + i].Value;
                    }
                }
                return matrix;
            }
        }

        public override string ToString()
        {
            int colorPlanes = Value.Length / 3;
            string ret = "{{";
            for (int y = 0; y < colorPlanes; y++)
            {
                for (int x = 0; x < 3; x++)
                {
                    ret += mValue[y * 3 + x].ToString();
                    if (x < 2) ret += "; ";
                }
                if (y < colorPlanes - 1) ret += "}, {";
            }
            ret += "}}";
            return ret;
        }
    }

    public class IFDDNGColorMatrix1 : IFDDNGMatrixTag
    {
        public IFDDNGColorMatrix1(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGColorMatrix1(SRational[] aMatrix)
            : base(aMatrix, TagID)
        { }
        public IFDDNGColorMatrix1(DNGMatrix aMatrix)
            : base(aMatrix, TagID)
        { }

        public const ushort TagID = 50721;

        public const string TagName = "Color Matrix 1";
        
        public override string ToString()
        {
            string ret = TagName + ": " + base.ToString();
            return ret;
        }
    }

    public class IFDDNGColorMatrix2 : IFDDNGMatrixTag
    {
        public IFDDNGColorMatrix2(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGColorMatrix2(SRational[] aMatrix)
            : base(aMatrix, TagID)
        { }
        public IFDDNGColorMatrix2(DNGMatrix aMatrix)
            : base(aMatrix, TagID)
        { }

        public const ushort TagID = 50722;

        public const string TagName = "Color Matrix 2";
        
        public override string ToString()
        {
            string ret = TagName + ": " + base.ToString();
            return ret;
        }
    }

    public class IFDDNGAnalogBalance : RationalImageFileDirectoryEntry
    {
        public IFDDNGAnalogBalance(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGAnalogBalance(Rational[] aAnalogBalance)
            : base(aAnalogBalance, TagID)
        { }
        public IFDDNGAnalogBalance(DNGVector aAnalogBalance)
            : base(new Rational[aAnalogBalance.Count], TagID)
        {
            for (uint i = 0; i < 3; i++)
            {
                uint denominator = 1000000;
                mValue[i] = new Rational(
                    (uint)Math.Round(aAnalogBalance[i] * denominator),
                    denominator);
            }
        }

        public const ushort TagID = 50727;

        public const string TagName = "Analog Balance";

        public Rational[] Value
        {
            get { return mValue; }
        }

        public DNGVector Vector
        {
            get 
            {
                DNGVector vec = new DNGVector((uint)mValue.Length);
                for (uint i = 0; i < mValue.Length; i++)
                {
                    vec[i] = mValue[i].Value;
                }
                return vec;
            }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue.Length.ToString() + " entries";
        }
    }

    public class IFDDNGAsShotNeutral : UShortOrULongOrRationalFileDirectoryEntry
    {
        public IFDDNGAsShotNeutral(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        {
        }

        public IFDDNGAsShotNeutral(Rational[] aAsShotNeutral)
            : base(aAsShotNeutral, TagID)
        {
        }

        public const ushort TagID = 50728;

        public const string TagName = "As Shot Neutral";
        
        public override string ToString()
        {
            if (mValue.Length == 3)
            {
                return TagName + ": (" + mValue[0].ToString() + ", " + mValue[1].ToString() + ", " + mValue[2].ToString() + ")";
            }
            return TagName + ": " + mValue[0].ToString() + " + " + (mValue.Length - 1).ToString() + " more...";
        }
    }

    public class IFDDNGAsShotWhiteXY : RationalImageFileDirectoryEntry
    {
        public IFDDNGAsShotWhiteXY(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGAsShotWhiteXY(Rational[] aAsShotWhiteXY)
            : base(aAsShotWhiteXY, TagID)
        { }

        public const ushort TagID = 50729;

        public const string TagName = "AsShot White XY";

        public Rational[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue.Length.ToString() + " entries";
        }
    }

    public class IFDDNGBaselineExposure : SRationalImageFileDirectoryEntry
    {
        public IFDDNGBaselineExposure(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGBaselineExposure(SRational aValue)
            : base(new SRational[] { aValue }, TagID)
        { }

        public const ushort TagID = 50730;

        public const string TagName = "Baseline Exposure";

        public SRational Value
        {
            get { return mValue[0]; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString() + " ev";
        }
    }

    public class IFDDNGBaselineNoise : RationalImageFileDirectoryEntry
    {
        public IFDDNGBaselineNoise(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGBaselineNoise(Rational aValue)
            : base(new Rational[] { aValue }, TagID)
        { }

        public const ushort TagID = 50731;

        public const string TagName = "Baseline Noise";

        public Rational Value
        {
            get { return mValue[0]; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString();
        }
    }

    public class IFDDNGBaselineSharpness : RationalImageFileDirectoryEntry
    {
        public IFDDNGBaselineSharpness(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGBaselineSharpness(Rational aValue)
            : base(new Rational[] { aValue }, TagID)
        { }

        public const ushort TagID = 50732;

        public const string TagName = "Baseline Sharpness";

        public Rational Value
        {
            get { return mValue[0]; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString();
        }
    }

    public class IFDDNGLinearResponseLimit : RationalImageFileDirectoryEntry
    {
        public IFDDNGLinearResponseLimit(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGLinearResponseLimit(Rational aValue)
            : base(new Rational[] { aValue }, TagID)
        { }

        public const ushort TagID = 50734;

        public const string TagName = "Linear Response Limit";

        public Rational Value
        {
            get { return mValue[0]; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString();
        }
    }

    public class IFDDNGLensInfo : RationalImageFileDirectoryEntry
    {
        public IFDDNGLensInfo(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGLensInfo(Rational[] aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 50736;

        public const string TagName = "Lens Info";

        public Rational[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString() + " - " + mValue[1].ToString() +
                ", " + mValue[2].ToString() +" - " + mValue[3].ToString();
        }
    }

    public class IFDDNGShadowScale : RationalImageFileDirectoryEntry
    {
        public IFDDNGShadowScale(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGShadowScale(Rational aValue)
            : base(new Rational[] { aValue }, TagID)
        { }

        public const ushort TagID = 50739;

        public const string TagName = "Shadow Scale";

        public Rational Value
        {
            get { return mValue[0]; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString();
        }
    }

    public class IFDDNGPrivateData : ByteImageFileDirectoryEntry
    {
        PentaxMakerNotes _pentaxMakerNotes;
        public IFDDNGPrivateData(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        {
            _pentaxMakerNotes = new PentaxMakerNotes(mValue, mOffset);
        }

        public IFDDNGPrivateData(byte[] aValue)
            : base(aValue, TagID)
        { 
            
        }

        public const ushort TagID = 50740;

        public const string TagName = "DNG Private Data";

        public byte[] Value
        {
            get { return mValue; }
        }

        //the makernotes will be empty if it is not a PENTAX file.
        public PentaxMakerNotes PentaxMakerNotes
        {
            get { return _pentaxMakerNotes; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue.Length.ToString();
        }
    }

    public abstract class IFDDNGCalibrationIlluminant : UShortImageFileDirectoryEntry
    {
        public enum Illuminant : ushort
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
            WarmWhiteFluorescent = 16,
            StandardLightA = 17,
            StandardLightB = 18,
            StandardLightC = 19,
            D55 = 20,
            D65 = 21,
            D75 = 22,
            D50 = 23,
            ISOStudioTungsten = 24,
            OtherLightSource
        }

        public IFDDNGCalibrationIlluminant(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public IFDDNGCalibrationIlluminant(Illuminant aValue, ushort aTagID)
            : base(new ushort[] { (ushort)aValue }, aTagID)
        { }
        
        public Illuminant Value
        {
            get { return (Illuminant)mValue[0]; }
        }
    }

    public class IFDDNGCalibrationIlluminant1 : IFDDNGCalibrationIlluminant
    {
        public IFDDNGCalibrationIlluminant1(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGCalibrationIlluminant1(Illuminant aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 50778;

        public const string TagName = "Calibration Illuminant 1";
        
        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString();
        }
    }

    public class IFDDNGCalibrationIlluminant2 : IFDDNGCalibrationIlluminant
    {
        public IFDDNGCalibrationIlluminant2(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGCalibrationIlluminant2(Illuminant aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 50779;

        public const string TagName = "Calibration Illuminant 2";
        
        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString();
        }
    }

    public class IFDDNGRawDataUniqueID : ByteImageFileDirectoryEntry
    {
        public IFDDNGRawDataUniqueID(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGRawDataUniqueID(byte[] aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 50781;

        public const string TagName = "Raw Data Unique ID";

        public byte[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue.Length.ToString() + " entries...";
        }
    }

    public class IFDDNGOriginalRawFileName : StringImageFileDirectoryEntry
    {
        public IFDDNGOriginalRawFileName(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGOriginalRawFileName(string aFileName)
            : base(aFileName, TagID)
        { }

        public const ushort TagID = 50827;

        public const string TagName = "Original Raw File Name";

        public string Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue;
        }
    }

    public class IFDDNGProfileCalibrationSignature : StringImageFileDirectoryEntry
    {
        public IFDDNGProfileCalibrationSignature(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGProfileCalibrationSignature(string aFileName)
            : base(aFileName, TagID)
        { }

        public const ushort TagID = 50932;

        public const string TagName = "Profile Calibration Signature";

        public string Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue;
        }
    }

    public class IFDDNGProfileName : StringImageFileDirectoryEntry
    {
        public IFDDNGProfileName(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGProfileName(string aFileName)
            : base(aFileName, TagID)
        { }

        public const ushort TagID = 50936;

        public const string TagName = "Profile Name";

        public string Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue;
        }
    }

    public class IFDDNGProfileEmbedPolicy : UIntImageFileDirectoryEntry
    {
        public IFDDNGProfileEmbedPolicy(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGProfileEmbedPolicy(uint aValue)
            : base(new uint[] { aValue }, TagID)
        { }

        public const ushort TagID = 50941;

        public const string TagName = "Profile Embed Policy";

        public uint Value
        {
            get { return mValue[0]; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString();
        }
    }

    public class IFDDNGProfileCopyright : StringImageFileDirectoryEntry
    {
        public IFDDNGProfileCopyright(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGProfileCopyright(string aFileName)
            : base(aFileName, TagID)
        { }

        public const ushort TagID = 50942;

        public const string TagName = "Profile Copyright";

        public string Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue;
        }
    }

    public class IFDDNGForwardMatrix1 : IFDDNGMatrixTag
    {
        public IFDDNGForwardMatrix1(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGForwardMatrix1(SRational[] aMatrix)
            : base(aMatrix, TagID)
        { }
        public IFDDNGForwardMatrix1(DNGMatrix aMatrix)
            : base(aMatrix, TagID)
        { }

        public const ushort TagID = 50964;

        public const string TagName = "Forward Matrix 1";
        
        public override string ToString()
        {
            string ret = TagName + ": " + base.ToString();
            return ret;
        }
    }

    public class IFDDNGForwardMatrix2 : IFDDNGMatrixTag
    {
        public IFDDNGForwardMatrix2(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGForwardMatrix2(SRational[] aMatrix)
            : base(aMatrix, TagID)
        { }
        public IFDDNGForwardMatrix2(DNGMatrix aMatrix)
            : base(aMatrix, TagID)
        { }

        public const ushort TagID = 50965;

        public const string TagName = "Forward Matrix 2";
        
        public override string ToString()
        {
            string ret = TagName + ": " + base.ToString();
            return ret;
        }
    }

    public class IFDDNGReductionMatrix1 : IFDDNGMatrixTag
    {
        public IFDDNGReductionMatrix1(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGReductionMatrix1(SRational[] aMatrix)
            : base(aMatrix, TagID)
        { }
        public IFDDNGReductionMatrix1(DNGMatrix aMatrix)
            : base(aMatrix, TagID)
        { }

        public const ushort TagID = 50725;

        public const string TagName = "Reduction Matrix 1";

        public override string ToString()
        {
            string ret = TagName + ": " + base.ToString();
            return ret;
        }
    }

    public class IFDDNGReductionMatrix2 : IFDDNGMatrixTag
    {
        public IFDDNGReductionMatrix2(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGReductionMatrix2(SRational[] aMatrix)
            : base(aMatrix, TagID)
        { }
        public IFDDNGReductionMatrix2(DNGMatrix aMatrix)
            : base(aMatrix, TagID)
        { }

        public const ushort TagID = 50726;

        public const string TagName = "Reduction Matrix 2";

        public override string ToString()
        {
            string ret = TagName + ": " + base.ToString();
            return ret;
        }
    }

    public class IFDDNGPreviewApplicationName : StringImageFileDirectoryEntry
    {
        public IFDDNGPreviewApplicationName(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGPreviewApplicationName(string aFileName)
            : base(aFileName, TagID)
        { }

        public const ushort TagID = 50966;

        public const string TagName = "Preview Application Name";

        public string Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue;
        }
    }

    public class IFDDNGPreviewApplicationVersion : StringImageFileDirectoryEntry
    {
        public IFDDNGPreviewApplicationVersion(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGPreviewApplicationVersion(string aFileName)
            : base(aFileName, TagID)
        { }

        public const ushort TagID = 50967;

        public const string TagName = "Preview Application Version";

        public string Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue;
        }
    }

    public class IFDDNGPreviewSettingsDigest : ByteImageFileDirectoryEntry
    {
        public IFDDNGPreviewSettingsDigest(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGPreviewSettingsDigest(byte[] aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 50969;

        public const string TagName = "Preview Settings Digest";

        public byte[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue.Length.ToString() + " entries...";
        }
    }

    public class IFDDNGPreviewColorSpace : UIntImageFileDirectoryEntry
    {
        public enum PreviewColorSpace : uint
        { 
            Unknown = 0,
            GrayGamma2_2 = 1,
            sRGB = 2,
            AdobeRGB = 3,
            ProPhotoRGB = 4
        }

        public IFDDNGPreviewColorSpace(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGPreviewColorSpace(PreviewColorSpace aValue)
            : base(new uint[] { (uint)aValue }, TagID)
        { }

        public const ushort TagID = 50970;

        public const string TagName = "Preview Color Space";

        public PreviewColorSpace Value
        {
            get { return (PreviewColorSpace)mValue[0]; }
        }

        public override string ToString()
        {
            return TagName + ": " + Value.ToString();
        }
    }

    public class IFDDNGPreviewDateTime : StringImageFileDirectoryEntry
    {
        public IFDDNGPreviewDateTime(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGPreviewDateTime(string aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 50971;

        public const string TagName = "Preview Date Time";

        public string Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue;
        }
    }

    public class IFDDNGRawImageDigest : ByteImageFileDirectoryEntry
    {
        public IFDDNGRawImageDigest(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGRawImageDigest(byte[] aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 50972;

        public const string TagName = "RawImageDigest";

        public byte[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue.Length.ToString() + " entries...";
        }
    }

    public class IFDDNGProfileLookTableDims : UIntImageFileDirectoryEntry
    {
        public IFDDNGProfileLookTableDims(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGProfileLookTableDims(uint[] aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 50981;

        public const string TagName = "Profile Look Table Dims";

        public uint[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString() + ", " + mValue[2].ToString() + ", " + mValue[3].ToString();
        }
    }

    public class IFDDNGProfileLookTableData : FloatImageFileDirectoryEntry
    {
        public IFDDNGProfileLookTableData(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGProfileLookTableData(float[] aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 50982;

        public const string TagName = "Profile Look Table Data";

        public float[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue.Length.ToString() + " entries...";
        }
    }

    public class IFDDNGXMPMetaData : StringImageFileDirectoryEntry
    {
        public IFDDNGXMPMetaData(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGXMPMetaData(string aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 700;

        public const string TagName = "XMP Meta Data";

        public string Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue;
        }
    }

    public class IFDDNGBlackLevel : UShortOrULongOrRationalFileDirectoryEntry
    {
        public IFDDNGBlackLevel(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        {
            
        }
        public IFDDNGBlackLevel(Rational[] aValue)
            : base(aValue, TagID)
        {
            
        }

        public const ushort TagID = 50714;

        public const string TagName = "Black Level";
        
        public override string ToString()
        {
            return TagName + ": " + base.ToString();
        }
    }

    public class IFDDNGBlackLevelDeltaH : SRationalImageFileDirectoryEntry
    {
        public IFDDNGBlackLevelDeltaH(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGBlackLevelDeltaH(SRational[] aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 50715;

        public const string TagName = "Black Level Delta H";

        public SRational[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue.Length.ToString() + " entries...";
        }
    }

    public class IFDDNGBlackLevelDeltaV : SRationalImageFileDirectoryEntry
    {
        public IFDDNGBlackLevelDeltaV(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGBlackLevelDeltaV(SRational[] aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 50716;

        public const string TagName = "Black Level Delta V";

        public SRational[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue.Length.ToString() + " entries...";
        }
    }

    public class IFDDNGWhiteLevel : UShortOrULongFileDirectoryEntry
    {
        public IFDDNGWhiteLevel(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        {

        }
        public IFDDNGWhiteLevel(uint[] aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 50717;

        public const string TagName = "White Level";

        public uint[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + base.ToString();
        }
    }

    public class IFDDNGDefaultScale : RationalImageFileDirectoryEntry
    {
        public IFDDNGDefaultScale(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGDefaultScale(Rational[] aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 50718;

        public const string TagName = "Default Scale";

        public Rational[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString() + ", " + mValue[1].ToString();
        }
    }

    public class IFDDNGDefaultCropOrigin : UShortOrULongOrRationalFileDirectoryEntry
    {
        public IFDDNGDefaultCropOrigin(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGDefaultCropOrigin(Rational[] aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 50719;

        public const string TagName = "Default Crop Origin";
        
        public override string ToString()
        {
            return TagName + ": " + base.ToString();
        }
    }

    public class IFDDNGDefaultCropSize : UShortOrULongOrRationalFileDirectoryEntry
    {
        public IFDDNGDefaultCropSize(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGDefaultCropSize(Rational[] aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 50720;

        public const string TagName = "Default Crop Size";
        
        public override string ToString()
        {
            return TagName + ": " + base.ToString();
        }
    }

    public class IFDDNGBayerGreenSplit : UIntImageFileDirectoryEntry
    {
        public IFDDNGBayerGreenSplit(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGBayerGreenSplit(uint aValue)
            : base(new uint[] { aValue }, TagID)
        { }

        public const ushort TagID = 50733;

        public const string TagName = "Bayer Green Split";

        public uint Value
        {
            get { return mValue[0]; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString();
        }
    }

    public class IFDDNGChromaBlurRadius : RationalImageFileDirectoryEntry
    {
        public IFDDNGChromaBlurRadius(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGChromaBlurRadius(Rational aValue)
            : base(new Rational[] { aValue }, TagID)
        { }

        public const ushort TagID = 50737;

        public const string TagName = "Chroma Blur Radius";

        public Rational Value
        {
            get { return mValue[0]; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString();
        }
    }

    public class IFDDNGAntiAliasStrength : RationalImageFileDirectoryEntry
    {
        public IFDDNGAntiAliasStrength(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGAntiAliasStrength(Rational aValue)
            : base(new Rational[] { aValue }, TagID)
        { }

        public const ushort TagID = 50738;

        public const string TagName = "Anti Alias Strength";

        public Rational Value
        {
            get { return mValue[0]; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString();
        }
    }

    public class IFDDNGActiveArea : UShortOrULongFileDirectoryEntry
    {
        public IFDDNGActiveArea(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        {

        }
        public IFDDNGActiveArea(uint[] aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 50829;

        public const string TagName = "Active Area";

        public uint[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": (" + mValue[0].ToString() + ", " + mValue[1].ToString() + ") - (" + mValue[2].ToString() + ", " + mValue[3].ToString() + ")";
        }
    }

    public class IFDDNGBestQualityScale : RationalImageFileDirectoryEntry
    {
        public IFDDNGBestQualityScale(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGBestQualityScale(Rational aValue)
            : base(new Rational[] { aValue }, TagID)
        { }

        public const ushort TagID = 50780;

        public const string TagName = "Best Quality Scale";

        public Rational Value
        {
            get { return mValue[0]; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString();
        }
    }

    public class IFDDNGNoiseProfile : DoubleImageFileDirectoryEntry
    {
        public IFDDNGNoiseProfile(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGNoiseProfile(double[] aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 51041;

        public const string TagName = "Noise Profile";

        public Double[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString();
        }
    }

    public class IFDDNGOpcodeList1 : OpcodeListFileDirectoryEntry
    {
        public IFDDNGOpcodeList1(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public IFDDNGOpcodeList1(List<OpCode> aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 51008;

        public const string TagName = "Opcode List 1";

        public List<OpCode> Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": ";
        }
    }

    public class IFDDNGOpcodeList2 : OpcodeListFileDirectoryEntry
    {
        public IFDDNGOpcodeList2(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public IFDDNGOpcodeList2(List<OpCode> aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 51009;

        public const string TagName = "Opcode List 2";

        public List<OpCode> Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " ;
        }
    }

    public class IFDDNGOpcodeList3 : OpcodeListFileDirectoryEntry
    {
        public IFDDNGOpcodeList3(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public IFDDNGOpcodeList3(List<OpCode> aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 51022;

        public const string TagName = "Opcode List 3";

        public List<OpCode> Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": ";
        }
    }

    public class IFDTIFFEPStandardID : ByteImageFileDirectoryEntry
    {
        public IFDTIFFEPStandardID(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDTIFFEPStandardID(byte[] aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 37398;

        public const string TagName = "TIFF/EP Standard ID";

        public byte[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0] + " " + mValue[1] + " " + mValue[2] + " " + mValue[3];
        }
    }

    public class IFDDNGCameraCalibration1 : IFDDNGMatrixTag
    {
        public IFDDNGCameraCalibration1(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGCameraCalibration1(SRational[] aMatrix)
            : base(aMatrix, TagID)
        { }
        public IFDDNGCameraCalibration1(DNGMatrix aMatrix)
            : base(aMatrix, TagID)
        { }

        public const ushort TagID = 50723;

        public const string TagName = "Camera Calibration 1";
        
        public override string ToString()
        {
            return TagName + ": " + base.ToString();
        }
    }

    public class IFDDNGCameraCalibration2 : IFDDNGMatrixTag
    {
        public IFDDNGCameraCalibration2(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGCameraCalibration2(SRational[] aMatrix)
            : base(aMatrix, TagID)
        { }
        public IFDDNGCameraCalibration2(DNGMatrix aMatrix)
            : base(aMatrix, TagID)
        { }

        public const ushort TagID = 50724;

        public const string TagName = "Camera Calibration 2";

        public override string ToString()
        {
            return TagName + ": " + base.ToString();
        }
    }

    public class IFDDNGProfileLookTableEncoding : IntImageFileDirectoryEntry
    {
        public enum ProfileLookTableEncoding : int
        { 
            LinearEncoding = 0,
            sRGBEncoding = 1
        }

        public IFDDNGProfileLookTableEncoding(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 51108;

        public const string TagName = "Profile Look Table Encoding";

        public ProfileLookTableEncoding Value
        {
            get { return (ProfileLookTableEncoding)mValue[0]; }
        }

        public override string ToString()
        {
            return TagName + ": " + ((ProfileLookTableEncoding)mValue[0]).ToString();
        }
    }

    public class IFDDNGCameraSerialNumber : StringImageFileDirectoryEntry
    {
        public IFDDNGCameraSerialNumber(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50735;

        public const string TagName = "Camera Serial Number";

        public string Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue;
        }
    }

    public class IFDDNGCameraCalibrationSignature : StringImageFileDirectoryEntry
    {
        public IFDDNGCameraCalibrationSignature(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50931;

        public const string TagName = "Camera Calibration Signature";

        public string Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue;
        }
    }

    public class IFDDNGProfileHueSatMapDims : IntImageFileDirectoryEntry
    {
        public IFDDNGProfileHueSatMapDims(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50937;

        public const string TagName = "Profile Hue Sat Map Dims";

        public int[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue.Length;
        }
    }

    public class IFDDNGProfileHueSatMapData1 : FloatImageFileDirectoryEntry
    {
        public IFDDNGProfileHueSatMapData1(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50938;

        public const string TagName = "Profile Hue Sat Map Data 1";

        public float[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue.Length.ToString() + " entries...";
        }
    }

    public class IFDDNGProfileHueSatMapData2 : FloatImageFileDirectoryEntry
    {
        public IFDDNGProfileHueSatMapData2(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50939;

        public const string TagName = "Profile Hue Sat Map Data 2";

        public float[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue.Length.ToString() + " entries...";
        }
    }

    public class IFDDNGOriginalRawFileData : ByteImageFileDirectoryEntry
    {
        public IFDDNGOriginalRawFileData(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public IFDDNGOriginalRawFileData(byte[] aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 50828;

        public const string TagName = "Original Raw File Data";

        public byte[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue.Length.ToString() + " entries...";
        }
    }

    public class IFDDNGMaskedAreas : UShortOrULongFileDirectoryEntry
    {
        public IFDDNGMaskedAreas(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGMaskedAreas(uint[] aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 50830;

        public const string TagName = "Masked Areas";

        public uint[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + base.ToString();
        }
    }

    public class IFDDNGAsShotICCProfile : ByteImageFileDirectoryEntry
    {
        public IFDDNGAsShotICCProfile(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public IFDDNGAsShotICCProfile(byte[] aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 50831;

        public const string TagName = "AsShot ICC Profile";

        public byte[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue.Length.ToString() + " entries...";
        }
    }

    public class IFDDNGAsShotPreProfileMatrix : IFDDNGMatrixTag
    {
        public IFDDNGAsShotPreProfileMatrix(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGAsShotPreProfileMatrix(SRational[] aMatrix)
            : base(aMatrix, TagID)
        { }
        public IFDDNGAsShotPreProfileMatrix(DNGMatrix aMatrix)
            : base(aMatrix, TagID)
        { }

        public const ushort TagID = 50832;

        public const string TagName = "AsShot Pre-Profile Matrix";
        
        public override string ToString()
        {
            return TagName + ": " + mValue.Length.ToString() + " entries...";
        }
    }

    public class IFDDNGCurrentICCProfile : ByteImageFileDirectoryEntry
    {
        public IFDDNGCurrentICCProfile(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public IFDDNGCurrentICCProfile(byte[] aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 50833;

        public const string TagName = "Current ICC Profile";

        public byte[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue.Length.ToString() + " entries...";
        }
    }

    public class IFDDNGCurrentPreProfileMatrix : IFDDNGMatrixTag
    {
        public IFDDNGCurrentPreProfileMatrix(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGCurrentPreProfileMatrix(SRational[] aMatrix)
            : base(aMatrix, TagID)
        { }
        public IFDDNGCurrentPreProfileMatrix(DNGMatrix aMatrix)
            : base(aMatrix, TagID)
        { }

        public const ushort TagID = 50834;

        public const string TagName = "Current Pre-Profile Matrix";

        public override string ToString()
        {
            return TagName + ": " + mValue.Length.ToString() + " entries...";
        }
    }

    public class IFDDNGColorimetricReference : UShortImageFileDirectoryEntry
    {
        public IFDDNGColorimetricReference(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGColorimetricReference(ushort aValue)
            : base(new ushort[] { aValue }, TagID)
        { }

        public const ushort TagID = 50879;

        public const string TagName = "Colorimetric Reference";

        public ushort Value
        {
            get { return mValue[0]; }
        }

        public override string ToString()
        {
            return TagName + ": " + Value.ToString();
        }
    }

    public class IFDDNGExtraCameraProfiles : UIntImageFileDirectoryEntry
    {
        public IFDDNGExtraCameraProfiles(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGExtraCameraProfiles(uint[] aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 50933;

        public const string TagName = "Extra Camera Profiles";

        public uint[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue.Length.ToString() + " entries...";
        }
    }

    public class IFDDNGAsShotProfileName : StringImageFileDirectoryEntry
    {
        public IFDDNGAsShotProfileName(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public IFDDNGAsShotProfileName(string aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 50934;

        public const string TagName = "AsShot Profile Name";

        public string Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue;
        }
    }

    public class IFDDNGNoiseReductionApplied : RationalImageFileDirectoryEntry
    {
        public IFDDNGNoiseReductionApplied(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public IFDDNGNoiseReductionApplied(Rational aValue)
            : base(new Rational[] { aValue }, TagID)
        { }

        public const ushort TagID = 50935;

        public const string TagName = "Noise Reduction Applied";

        public Rational Value
        {
            get { return mValue[0]; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].Value.ToString();
        }
    }

    public class IFDDNGProfileToneCurve : FloatImageFileDirectoryEntry
    {
        public IFDDNGProfileToneCurve(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public IFDDNGProfileToneCurve(float[] aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 50940;

        public const string TagName = "Profile Tone Curve";

        public float[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue.Length.ToString() + " entries...";
        }
    }

    public class IFDDNGPreviewSettingsName : StringImageFileDirectoryEntry
    {
        public IFDDNGPreviewSettingsName(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public IFDDNGPreviewSettingsName(string aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 50968;

        public const string TagName = "Preview Settings Name";

        public string Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue;
        }
    }

    public class IFDDNGOriginalRawFileDigest : ByteImageFileDirectoryEntry
    {
        public IFDDNGOriginalRawFileDigest(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public IFDDNGOriginalRawFileDigest(byte[] aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 50973;

        public const string TagName = "Original Raw File Digest";

        public byte[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            string hex = string.Empty;
            for (int i = 0; i < mValue.Length; i++)
            {
                hex += mValue[i].ToString("X2");
                if (i < mValue.Length - 1)
                {
                    hex += " ";
                }
            }
            return TagName + ": " + hex;
        }
    }

    public class IFDDNGSubTileBlockSize : UShortOrULongFileDirectoryEntry
    {
        public IFDDNGSubTileBlockSize(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGSubTileBlockSize(uint[] aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 50974;

        public const string TagName = "Sub Tile Block Size";

        public uint[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString() + "; " + mValue[1].ToString();
        }
    }

    public class IFDDNGRowInterleaveFactor : UShortOrULongFileDirectoryEntry
    {
        public IFDDNGRowInterleaveFactor(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGRowInterleaveFactor(uint aValue)
            : base(new uint[] { aValue }, TagID)
        { }

        public const ushort TagID = 50975;

        public const string TagName = "Row Interleave Factor";

        public uint Value
        {
            get { return mValue[0]; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString();
        }
    }

    public class IFDDNGDefaultUserCrop : RationalImageFileDirectoryEntry
    {
        public IFDDNGDefaultUserCrop(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public IFDDNGDefaultUserCrop(Rational[] aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 51125;

        public const string TagName = "Default User Crop";

        public Rational[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].Value.ToString() + ", " + mValue[1].Value.ToString() + ", " + mValue[2].Value.ToString() + ", " + mValue[3].Value.ToString();
        }
    }

    public class IFDDNGDefaultBlackRender : UIntImageFileDirectoryEntry
    {
        public IFDDNGDefaultBlackRender(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGDefaultBlackRender(uint aValue)
            : base(new uint[] { aValue }, TagID)
        { }

        public const ushort TagID = 51110;

        public const string TagName = "Default Black Render";

        public uint Value
        {
            get { return mValue[0]; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0];
        }
    }

    public class IFDDNGBaselineExposureOffset : RationalImageFileDirectoryEntry
    {
        public IFDDNGBaselineExposureOffset(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public IFDDNGBaselineExposureOffset(Rational aValue)
            : base(new Rational[] { aValue }, TagID)
        { }

        public const ushort TagID = 51109;

        public const string TagName = "Baseline Exposure Offset";

        public Rational Value
        {
            get { return mValue[0]; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].Value.ToString();
        }
    }

    public class IFDDNGProfileHueSatMapEncoding : UIntImageFileDirectoryEntry
    {
        public enum ProfileHueSatMapEncoding
        { 
            Linear = 0,
            sRGB = 1
        }

        public IFDDNGProfileHueSatMapEncoding(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGProfileHueSatMapEncoding(ProfileHueSatMapEncoding aValue)
            : base(new uint[] { (uint)aValue }, TagID)
        { }

        public const ushort TagID = 51107;

        public const string TagName = "Profile Hue Sat Map Encoding";

        public ProfileHueSatMapEncoding Value
        {
            get { return (ProfileHueSatMapEncoding)mValue[0]; }
        }

        public override string ToString()
        {
            return TagName + ": " + Value.ToString();
        }
    }

    public class IFDDNGOriginalDefaultFinalSize : UShortOrULongFileDirectoryEntry
    {
        public IFDDNGOriginalDefaultFinalSize(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGOriginalDefaultFinalSize(uint[] aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 51089;

        public const string TagName = "Original Default Final Size";

        public uint[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString() + ", " + mValue[1].ToString();
        }
    }

    public class IFDDNGOriginalBestQualityFinalSize : UShortOrULongFileDirectoryEntry
    {
        public IFDDNGOriginalBestQualityFinalSize(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGOriginalBestQualityFinalSize(uint[] aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 51090;

        public const string TagName = "Original Best Quality Final Size";

        public uint[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString() + ", " + mValue[1].ToString();
        }
    }

    public class IFDDNGOriginalDefaultCropSize : UShortOrULongOrRationalFileDirectoryEntry
    {
        public IFDDNGOriginalDefaultCropSize(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }
        public IFDDNGOriginalDefaultCropSize(Rational[] aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 51091;

        public const string TagName = "Original Default Crop Size";
        
        public override string ToString()
        {
            return TagName + ": " + mValue[0].Value.ToString() + ", " + mValue[1].Value.ToString();
        }
    }

    public class IFDDNGNewRawImageDigest : ByteImageFileDirectoryEntry
    {
        public IFDDNGNewRawImageDigest(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public IFDDNGNewRawImageDigest(byte[] aValue)
            : base(aValue, TagID)
        { }

        public const ushort TagID = 51111;

        public const string TagName = "New Raw Image Digest";

        public byte[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            string hex = string.Empty;
            for (int i = 0; i < mValue.Length; i++)
            {
                hex += mValue[i].ToString("X2");
                if (i < mValue.Length - 1)
                {
                    hex += " ";
                }
            }
            return TagName + ": " + hex;
        }
    }

    public class IFDDNGRawToPreviewGain : DoubleImageFileDirectoryEntry
    {
        public IFDDNGRawToPreviewGain(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public IFDDNGRawToPreviewGain(double aValue)
            : base(new double[] { aValue}, TagID)
        { }

        public const ushort TagID = 51112;

        public const string TagName = "Raw To Preview Gain";

        public double Value
        {
            get { return mValue[0]; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0];
        }
    }

    #endregion
}
