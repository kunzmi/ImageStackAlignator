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
using System.IO;

namespace PentaxPefFile
{
	public class ImageFileDirectory
	{
		private FileReader mPefFile;
		private ushort mEntryCount;
		public List<ImageFileDirectoryEntry> mEntries;

		public ImageFileDirectory(FileReader aPefFile)
		{
			mPefFile = aPefFile;
			mEntries = new List<ImageFileDirectoryEntry>();
			mEntryCount = mPefFile.ReadUI2();
			for (ushort i = 0; i < mEntryCount; i++)
			{
				ImageFileDirectoryEntry entry = ImageFileDirectoryEntry.CreateImageFileDirectoryEntry(mPefFile);
				mEntries.Add(entry);
			}
		}

		public T GetEntry<T>() where T : ImageFileDirectoryEntry
		{
			Type t = typeof(T);
			foreach (var item in mEntries)
			{
				if (item is T)
					return (T)item;
			}
			return null;
		}

        internal void SavePass1(Stream stream)
        {            
            stream.Write(BitConverter.GetBytes((ushort)mEntries.Count), 0, 2);
            for (int i = 0; i < mEntries.Count; i++)
            {
                mEntries[i].SavePass1(stream);
            }
        }

        internal void SavePass2(Stream stream)
        {
            //end of IFDs marker
            //stream.Write(new byte[] { 0, 0, 0, 0 }, 0, 4);
            for (int i = 0; i < mEntries.Count; i++)
            {
                mEntries[i].SavePass2(stream);
            }
        }

        public ImageFileDirectory(uint width, uint height)
        {
            mEntries = new List<ImageFileDirectoryEntry>();

            mEntries.Add(new IFDImageWidth(width));
            mEntries.Add(new IFDImageLength(height));
            mEntries.Add(new IFDBitsPerSample(new ushort[] { 16, 16, 16}));
            mEntries.Add(new IFDCompression(IFDCompression.Compression.NoCompression));
            mEntries.Add(new IFDPhotometricInterpretation(IFDPhotometricInterpretation.PhotometricInterpretation.RGB));
            mEntries.Add(new IFDStripOffsets(new uint[] { 0 }));
            mEntries.Add(new IFDSamplesPerPixel(3));
            mEntries.Add(new IFDRowsPerStrip(height));
            mEntries.Add(new IFDStripByteCounts(new uint[] { width * height * 3 * 2 }));
            mEntries.Add(new IFDXResolution(new Rational(720, 10)));
            mEntries.Add(new IFDYResolution(new Rational(720, 10)));
            mEntries.Add(new IFDResolutionUnit(IFDResolutionUnit.ResolutionUnit.None));
            mEntries.Add(new IFDMake("MK TIFF"));
            mEntryCount = (ushort)mEntries.Count;
        }

        public void SaveAsTiff(string aFilename, ushort[] data)
        {
            FileStream fs = new FileStream(aFilename, FileMode.Create, FileAccess.Write);
            fs.Write(new byte[] { 0x49, 0x49, 0x2A, 00, 8, 0, 0, 0 }, 0, 8); //Tiff header with offset to first IFD=8
            fs.Write(BitConverter.GetBytes(mEntryCount), 0, 2);
            foreach (var item in mEntries)
            {
                fs.Flush();
                item.SavePass1(fs);
            }
            fs.Write(new byte[] { 0, 0, 0, 0 }, 0, 4);
            foreach (var item in mEntries)
            {
                item.SavePass2(fs);
            }
            fs.Seek(0, SeekOrigin.End);
            uint finalImageOffset = (uint)fs.Position;
            GetEntry<IFDStripOffsets>().SaveFinalOffsets(fs, new uint[] { finalImageOffset });
            fs.Seek(finalImageOffset, SeekOrigin.Begin);
            BinaryWriter br = new BinaryWriter(fs);
            for (int i = 0; i < data.Length; i++)
            {
                br.Write(data[i]);
            }
            br.Close();
            br.Dispose();
            fs.Close();
        }
	}
}
