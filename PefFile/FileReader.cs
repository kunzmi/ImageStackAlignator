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
using System.Text;
using System.IO;
using System.Runtime.InteropServices;

namespace PentaxPefFile
{
	public class FileReader
	{
		protected Stream mFile;
		protected BinaryReader mFileReader;
		protected string mFileName;
		protected bool mEndianSwap;

		public FileReader(string aFileName)
		{
			mFileName = aFileName;
			mFile = new FileStream(aFileName, FileMode.Open, FileAccess.Read);
			mFileReader = new BinaryReader(mFile);
			mEndianSwap = false;
		}

		public FileReader(Stream aFileStream)
		{
			mFile = aFileStream;
			mFileReader = new BinaryReader(mFile);
			mEndianSwap = false;
		}

		protected ushort Endian_swap(ushort x)
		{
			return (ushort)((x >> 8) | (x << 8));
		}

		protected uint Endian_swap(uint x)
		{
			return (uint)((x >> 24) |
							((x << 8) & 0x00FF0000) |
							((x >> 8) & 0x0000FF00) |
							(x << 24));
		}

		protected ulong Endian_swap(ulong x)
		{
			return (ulong)((x >> 56) |
							((x << 40) & 0x00FF000000000000) |
							((x << 24) & 0x0000FF0000000000) |
							((x << 8) & 0x000000FF00000000) |
							((x >> 8) & 0x00000000FF000000) |
							((x >> 24) & 0x0000000000FF0000) |
							((x >> 40) & 0x000000000000FF00) |
							(x << 56));
		}

		protected unsafe short Endian_swap(short x)
        {
            ushort i = *(ushort*)&x;
            return (short)((i >> 8) | (i << 8));
		}

		protected unsafe int Endian_swap(int x)
        {
            uint i = *(uint*)&x;
            return (int)((i >> 24) |
							((i << 8) & 0x00FF0000) |
							((i >> 8) & 0x0000FF00) |
							(i << 24));
		}

		protected unsafe long Endian_swap(long x)
		{
            ulong i = *(ulong*)&x;
            return (long)((i >> 56) |
							((i << 40) & 0x00FF000000000000) |
							((i << 24) & 0x0000FF0000000000) |
							((i << 8) & 0x000000FF00000000) |
							((i >> 8) & 0x00000000FF000000) |
							((i >> 24) & 0x0000000000FF0000) |
							((i >> 40) & 0x000000000000FF00) |
							(i << 56));
		}

		protected unsafe double Endian_swap(double x)
		{
			ulong i = *(ulong*)&x;
			i = (i >> 56) |
				((i << 40) & 0x00FF000000000000) |
				((i << 24) & 0x0000FF0000000000) |
				((i << 8) & 0x000000FF00000000) |
				((i >> 8) & 0x00000000FF000000) |
				((i >> 24) & 0x0000000000FF0000) |
				((i >> 40) & 0x000000000000FF00) |
				(i << 56);
			return *(double*)&i;
		}

		protected unsafe float Endian_swap(float x)
		{
			uint i = *(uint*)&x;
			i = (i >> 24) |
				((i << 8) & 0x00FF0000) |
				((i >> 8) & 0x0000FF00) |
				(i << 24);
			return *(float*)&i;
		}

		public long ReadI8()
		{
			long temp = mFileReader.ReadInt64();
			if (mEndianSwap) temp = Endian_swap(temp);
			return temp;
        }

        public long ReadI8BE()
        {
            long temp = mFileReader.ReadInt64();
            if (BitConverter.IsLittleEndian) temp = Endian_swap(temp);
            return temp;
        }

        public int ReadI4()
		{
			int temp = mFileReader.ReadInt32();
			if (mEndianSwap) temp = Endian_swap(temp);
			return temp;
		}

        public int ReadI4BE()
		{
			int temp = mFileReader.ReadInt32();
			if (BitConverter.IsLittleEndian) temp = Endian_swap(temp);
			return temp;
		}

		public short ReadI2()
		{
			short temp = mFileReader.ReadInt16();
			if (mEndianSwap) temp = Endian_swap(temp);
			return temp;
		}

		public short ReadI2BE()
		{
			short temp = mFileReader.ReadInt16();
			if (BitConverter.IsLittleEndian) temp = Endian_swap(temp);
			return temp;
		}

		public sbyte ReadI1()
		{
			sbyte temp = mFileReader.ReadSByte();
			return temp;
		}

		public ulong ReadUI8()
		{
			ulong temp = mFileReader.ReadUInt64();
			if (mEndianSwap) temp = Endian_swap(temp);
			return temp;
		}

		public ulong ReadUI8BE()
		{
			ulong temp = mFileReader.ReadUInt64();
			if (BitConverter.IsLittleEndian) temp = Endian_swap(temp);
			return temp;
		}

		public uint ReadUI4()
		{
			uint temp = mFileReader.ReadUInt32();
			if (mEndianSwap) temp = Endian_swap(temp);
			return temp;
		}

		public uint ReadUI4BE()
		{
			uint temp = mFileReader.ReadUInt32();
			if (BitConverter.IsLittleEndian) temp = Endian_swap(temp);
			return temp;
		}

		public ushort ReadUI2()
		{
			ushort temp = mFileReader.ReadUInt16();
			if (mEndianSwap) temp = Endian_swap(temp);
			return temp;
		}

		public ushort ReadUI2BE()
		{
			ushort temp = mFileReader.ReadUInt16();
			if (BitConverter.IsLittleEndian) temp = Endian_swap(temp);
			return temp;
		}

		public byte ReadUI1()
		{
			byte temp = mFileReader.ReadByte();
			return temp;
		}

		public float ReadF4()
		{
			float temp = mFileReader.ReadSingle();
			if (mEndianSwap) temp = Endian_swap(temp);
			return temp;
		}

		public float ReadF4BE()
		{
			float temp = mFileReader.ReadSingle();
			if (BitConverter.IsLittleEndian) temp = Endian_swap(temp);
			return temp;
		}

		public double ReadF8()
		{
			double temp = mFileReader.ReadDouble();
			if (mEndianSwap) temp = Endian_swap(temp);
			return temp;
		}

		public double ReadF8BE()
		{
			double temp = mFileReader.ReadDouble();
			if (BitConverter.IsLittleEndian) temp = Endian_swap(temp);
			return temp;
		}

		public string ReadStr(int aCountBytes)
		{
			byte[] arr = mFileReader.ReadBytes(aCountBytes);
			System.Text.ASCIIEncoding enc = new System.Text.ASCIIEncoding();
			return enc.GetString(arr);
		}

		public void Close()
		{
			mFileReader.Close();
			mFile.Close();
		}

		public void Seek(uint aPosition, SeekOrigin aOrigin)
		{
			mFile.Seek(aPosition, aOrigin);
		}

		public uint Position()
		{
			return (uint)mFile.Position;
		}

		public bool EndianSwap
		{
			get { return mEndianSwap; }
		}
	}
}
