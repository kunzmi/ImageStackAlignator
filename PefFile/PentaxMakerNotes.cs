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
	public class PentaxMakerNotes : FileReader
	{
		List<PentaxMakerNotesEntry> mEntries;
		ushort mEntryCount;
		public bool K3Specific = false;
        //the "AOC" makenotes uses absolute offsets to file beginning, not in makernote.
        //hence subtract that offset...
        int mAdditionalOffset;

        public void SeekWithOffset(uint aPosition, SeekOrigin aOrigin)
        {
            base.Seek((uint)((int)aPosition - mAdditionalOffset), aOrigin);
        }

        public PentaxMakerNotes(byte[] aData, uint offset)
			: base(new MemoryStream(aData))
		{
            //FileStream fs = new FileStream("dumpMakerNote.bin", FileMode.Create, FileAccess.Write);
            //fs.Write(aData, 0, aData.Length);
            //fs.Close();


            mAdditionalOffset = 0;
			mEntries = new List<PentaxMakerNotesEntry>();

            string testVersion = ReadStr(4);
			if (testVersion == "AOC\0")
			{
				byte a = mFileReader.ReadByte();
				byte b = mFileReader.ReadByte();

				bool fileIsLittleEndian;
				if (a == b && b == 'I')
					fileIsLittleEndian = true;
				else
					if (a == b && b == 'M')
						fileIsLittleEndian = false;
					else
						throw new FileLoadException("Could not determine file endianess for maker notes");

				mEndianSwap = fileIsLittleEndian != BitConverter.IsLittleEndian;
                mAdditionalOffset = (int)offset;
			}
			else
			{
				Seek(0, SeekOrigin.Begin);
				testVersion = ReadStr(8);
                if (testVersion == "PENTAX \0")
                {
                    byte a = mFileReader.ReadByte();
                    byte b = mFileReader.ReadByte();

                    bool fileIsLittleEndian;
                    if (a == b && b == 'I')
                        fileIsLittleEndian = true;
                    else
                        if (a == b && b == 'M')
                        fileIsLittleEndian = false;
                    else
                        throw new FileLoadException("Could not determine file endianess for maker notes");

                    mEndianSwap = fileIsLittleEndian != BitConverter.IsLittleEndian;
                }
                else
                {
                    Seek(0, SeekOrigin.Begin);
                    testVersion = ReadStr(10);
                    const int SizeAdobeHeader = 20;
                    if (testVersion == "Adobe\0MakN")
                    {
                        uint byteCount = mFileReader.ReadUInt32();
                        if (BitConverter.IsLittleEndian)
                        {
                            byteCount = ((byteCount >> 24) |
                            ((byteCount << 8) & 0x00FF0000) |
                            ((byteCount >> 8) & 0x0000FF00) |
                            (byteCount << 24));
                        }

                        byte a = mFileReader.ReadByte();
                        byte b = mFileReader.ReadByte();

                        bool fileIsLittleEndian;
                        if (a == b && b == 'I')
                            fileIsLittleEndian = true;
                        else
                            if (a == b && b == 'M')
                            fileIsLittleEndian = false;
                        else
                            throw new FileLoadException("Could not determine file endianess for maker notes");

                        mEndianSwap = fileIsLittleEndian != BitConverter.IsLittleEndian;

                        uint temp = mFileReader.ReadUInt32();
                        if (mEndianSwap)
                        {
                            temp = ((temp >> 24) |
                            ((temp << 8) & 0x00FF0000) |
                            ((temp >> 8) & 0x0000FF00) |
                            (temp << 24));
                        }
                        mAdditionalOffset = (int)temp;





                        testVersion = ReadStr(4);
                        if (testVersion == "AOC\0")
                        {
                            a = mFileReader.ReadByte();
                            b = mFileReader.ReadByte();

                            
                            if (a == b && b == 'I')
                                fileIsLittleEndian = true;
                            else
                                if (a == b && b == 'M')
                                fileIsLittleEndian = false;
                            else
                                throw new FileLoadException("Could not determine file endianess for maker notes");

                            mEndianSwap = fileIsLittleEndian != BitConverter.IsLittleEndian;
                            mAdditionalOffset -= SizeAdobeHeader;
                        }
                        else
                        {
                            Seek(Position() - 4, SeekOrigin.Begin);
                            testVersion = ReadStr(8);
                            if (testVersion == "PENTAX \0")
                            {
                                a = mFileReader.ReadByte();
                                b = mFileReader.ReadByte();

                                
                                if (a == b && b == 'I')
                                    fileIsLittleEndian = true;
                                else
                                    if (a == b && b == 'M')
                                    fileIsLittleEndian = false;
                                else
                                    throw new FileLoadException("Could not determine file endianess for maker notes");

                                mEndianSwap = fileIsLittleEndian != BitConverter.IsLittleEndian;
                                mAdditionalOffset = -SizeAdobeHeader;
                            }
                        }
                            /* Following Exiftool:
       1. Six bytes containing the zero-terminated string "Adobe". (The DNG specification calls for the DNGPrivateData tag to start with an ASCII string identifying the creator/format).
       2. 4 bytes: an ASCII string ("MakN" for a Makernote),  indicating what sort of data is being stored here. Note that this is not zero-terminated.
       3. A four-byte count (number of data bytes following); this is the length of the original MakerNote data. (This is always in "most significant byte first" format).
       4. 2 bytes: the byte-order indicator from the original file (the usual 'MM'/4D4D or 'II'/4949).
       5. 4 bytes: the original file offset for the MakerNote tag data (stored according to the byte order given above).
       6. The contents of the MakerNote tag. This is a simple byte-for-byte copy, with no modification.
                             */
                        }
                    else
                    {
                        return; //Not a Pentax make note, can't handle it...
                    }
                }
			}

			mEntryCount = ReadUI2();
			for (ushort i = 0; i < mEntryCount; i++)
			{
				PentaxMakerNotesEntry entry = PentaxMakerNotesEntry.CreatePentaxMakerNotesEntry(this);
				mEntries.Add(entry);
			}

			//MNPreviewImageSize imagesize = GetEntry<MNPreviewImageSize>();
			//MNPreviewImageLength imagelength = GetEntry<MNPreviewImageLength>();
			//MNPreviewImageStart imagestart = GetEntry<MNPreviewImageStart>();
			//MNPreviewImageBorders imageborder = GetEntry<MNPreviewImageBorders>();

			//uint curPos = Position();
			//Seek(imagestart.Value, SeekOrigin.Begin);
			//byte[] data = mFileReader.ReadBytes((int)imagelength.Value);
			//Seek(curPos, SeekOrigin.Begin);

		}

		public T GetEntry<T>() where T : PentaxMakerNotesEntry
		{
			Type t = typeof(T);
			foreach (var item in mEntries)
			{
				if (item is T)
					return (T)item;
			}
			return null;
		}
	}
}
