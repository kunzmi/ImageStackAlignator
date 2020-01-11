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
    public abstract class OpCode
    {
        protected uint _id;
        protected uint _version;
        protected uint _flags;
        protected uint _sizeInBytes;

        #region Byte swap
        protected ushort Endian_swap(ushort x)
        {
            if (BitConverter.IsLittleEndian)
            {
                return (ushort)((x >> 8) | (x << 8));
            }
            return x;
        }

        protected uint Endian_swap(uint x)
        {
            if (BitConverter.IsLittleEndian)
            {
                return (uint)((x >> 24) |
                            ((x << 8) & 0x00FF0000) |
                            ((x >> 8) & 0x0000FF00) |
                            (x << 24));
            }
            return x;
        }

        protected ulong Endian_swap(ulong x)
        {
            if (BitConverter.IsLittleEndian)
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
            return x;
        }

        protected unsafe short Endian_swap(short x)
        {
            if (BitConverter.IsLittleEndian)
            {
                ushort i = *(ushort*)&x;
                return (short)((i >> 8) | (i << 8));
            }
            return x;
        }

        protected unsafe int Endian_swap(int x)
        {
            if (BitConverter.IsLittleEndian)
            {
                uint i = *(uint*)&x;
                return (int)((i >> 24) |
                            ((i << 8) & 0x00FF0000) |
                            ((i >> 8) & 0x0000FF00) |
                            (i << 24));
            }
            return x;
        }

        protected unsafe long Endian_swap(long x)
        {
            if (BitConverter.IsLittleEndian)
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
            return x;
        }

        protected unsafe double Endian_swap(double x)
        {
            if (BitConverter.IsLittleEndian)
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
            return x;
        }

        protected unsafe float Endian_swap(float x)
        {
            if (BitConverter.IsLittleEndian)
            {
                uint i = *(uint*)&x;
                i = (i >> 24) |
                ((i << 8) & 0x00FF0000) |
                ((i >> 8) & 0x0000FF00) |
                (i << 24);
                return *(float*)&i;
            }
            return x;
        }
        #endregion

        protected OpCode(uint aID, uint aVersion, uint aFlags, uint aSizeInBytes)
        {
            _id = aID;
            _version = aVersion;
            _flags = aFlags;
            _sizeInBytes = aSizeInBytes;
        }

        public static OpCode Create(FileReader aPefFile)
        {
            uint ID = aPefFile.ReadUI4BE();
            uint Version = aPefFile.ReadUI4BE();
            uint Flags = aPefFile.ReadUI4BE();
            uint SizeInBytes = aPefFile.ReadUI4BE();

            switch (ID)
            {
                case 1:
                    return new WarpRectilinear(aPefFile, ID, Version, Flags, SizeInBytes);
                case 2:
                    return new WarpFisheye(aPefFile, ID, Version, Flags, SizeInBytes);
                case 3:
                    return new FixVignetteRadial(aPefFile, ID, Version, Flags, SizeInBytes);
                case 4:
                    return new FixBadPixelsConstant(aPefFile, ID, Version, Flags, SizeInBytes);
                case 5:
                    return new FixBadPixelsList(aPefFile, ID, Version, Flags, SizeInBytes);
                case 6:
                    return new TrimBounds(aPefFile, ID, Version, Flags, SizeInBytes);
                case 7:
                    return new MapTable(aPefFile, ID, Version, Flags, SizeInBytes);
                case 8:
                    return new MapPolynomial(aPefFile, ID, Version, Flags, SizeInBytes);
                case 9:
                    return new GainMap(aPefFile, ID, Version, Flags, SizeInBytes);
                case 10:
                    return new DeltaPerRow(aPefFile, ID, Version, Flags, SizeInBytes);
                case 11:
                    return new DeltaPerColumn(aPefFile, ID, Version, Flags, SizeInBytes);
                case 12:
                    return new ScalePerRow(aPefFile, ID, Version, Flags, SizeInBytes);
                case 13:
                    return new ScalePerColumn(aPefFile, ID, Version, Flags, SizeInBytes);
                default:
                    return new Unknown(aPefFile, ID, Version, Flags, SizeInBytes);
            }

        }

        public abstract byte[] GetAsBytes();
    }

    public class WarpRectilinear : OpCode
    {
        private int _coefficientSetCount;
        private List<Coefficients> _coefficients;
        private double _Cx;
        private double _Cy;

        public struct Coefficients
        {
            public double Kr0;
            public double Kr1;
            public double Kr2;
            public double Kr3;
            public double Kt0;
            public double Kt1;
        }

        public WarpRectilinear(FileReader aPefFile, uint aID, uint aVersion, uint aFlags, uint aSizeInBytes)
            : base(aID, aVersion, aFlags, aSizeInBytes)
        {
            _coefficientSetCount = aPefFile.ReadI4BE();
            if (aSizeInBytes != 4 + _coefficientSetCount * 6 * sizeof(double) + 2 * sizeof(double))
            {
                throw new ArgumentException("Opcode parameter length doesn't match opcode.");
            }
            _coefficients = new List<Coefficients>();
            for (int i = 0; i < _coefficientSetCount; i++)
            {
                Coefficients c = new Coefficients();
                c.Kr0 = aPefFile.ReadF8BE();
                c.Kr1 = aPefFile.ReadF8BE();
                c.Kr2 = aPefFile.ReadF8BE();
                c.Kr3 = aPefFile.ReadF8BE();
                c.Kt0 = aPefFile.ReadF8BE();
                c.Kt1 = aPefFile.ReadF8BE();
                _coefficients.Add(c);
            }
            _Cx = aPefFile.ReadF8BE();
            _Cy = aPefFile.ReadF8BE();
        }

        public override byte[] GetAsBytes()
        {
            MemoryStream ms = new MemoryStream();
            BinaryWriter bw = new BinaryWriter(ms);

            bw.Write(Endian_swap(_id));
            bw.Write(Endian_swap(_version));
            bw.Write(Endian_swap(_flags));
            bw.Write(Endian_swap(_sizeInBytes));

            bw.Write(Endian_swap(_coefficientSetCount));

            for (int i = 0; i < _coefficientSetCount; i++)
            {
                Coefficients c = _coefficients[i];
                bw.Write(Endian_swap(c.Kr0));
                bw.Write(Endian_swap(c.Kr1));
                bw.Write(Endian_swap(c.Kr2));
                bw.Write(Endian_swap(c.Kr3));
                bw.Write(Endian_swap(c.Kt0));
                bw.Write(Endian_swap(c.Kt1));
            }
            bw.Write(Endian_swap(_Cx));
            bw.Write(Endian_swap(_Cy));

            bw.Flush();
            ms.Flush();
            byte[] ret = ms.ToArray();
            bw.Close();
            bw.Dispose();
            return ret;
        }
    }

    public class WarpFisheye : OpCode
    {
        private int _coefficientSetCount;
        private List<Coefficients> _coefficients;
        private double _Cx;
        private double _Cy;

        public struct Coefficients
        {
            public double Kr0;
            public double Kr1;
            public double Kr2;
            public double Kr3;
        }

        public WarpFisheye(FileReader aPefFile, uint aID, uint aVersion, uint aFlags, uint aSizeInBytes)
            : base(aID, aVersion, aFlags, aSizeInBytes)
        {
            _coefficientSetCount = aPefFile.ReadI4BE();
            if (aSizeInBytes != _coefficientSetCount * 6 * sizeof(double) + 4)
            {
                throw new ArgumentException("Opcode parameter length doesn't match opcode.");
            }
            _coefficients = new List<Coefficients>();
            for (int i = 0; i < _coefficientSetCount; i++)
            {
                Coefficients c = new Coefficients();
                c.Kr0 = aPefFile.ReadF8BE();
                c.Kr1 = aPefFile.ReadF8BE();
                c.Kr2 = aPefFile.ReadF8BE();
                c.Kr3 = aPefFile.ReadF8BE();
                _coefficients.Add(c);
            }
            _Cx = aPefFile.ReadF8BE();
            _Cy = aPefFile.ReadF8BE();
        }


        public override byte[] GetAsBytes()
        {
            MemoryStream ms = new MemoryStream();
            BinaryWriter bw = new BinaryWriter(ms);

            bw.Write(Endian_swap(_id));
            bw.Write(Endian_swap(_version));
            bw.Write(Endian_swap(_flags));
            bw.Write(Endian_swap(_sizeInBytes));

            bw.Write(Endian_swap(_coefficientSetCount));

            for (int i = 0; i < _coefficientSetCount; i++)
            {
                Coefficients c = _coefficients[i];
                bw.Write(Endian_swap(c.Kr0));
                bw.Write(Endian_swap(c.Kr1));
                bw.Write(Endian_swap(c.Kr2));
                bw.Write(Endian_swap(c.Kr3));
            }
            bw.Write(Endian_swap(_Cx));
            bw.Write(Endian_swap(_Cy));

            bw.Flush();
            ms.Flush();
            byte[] ret = ms.ToArray();
            bw.Close();
            bw.Dispose();
            return ret;
        }
    }

    public class FixVignetteRadial : OpCode
    {
        private Coefficients _coefficients;

        public struct Coefficients
        {
            public double K0;
            public double K1;
            public double K2;
            public double K3;
            public double K4;
            public double Cx;
            public double Cy;
        }

        public FixVignetteRadial(FileReader aPefFile, uint aID, uint aVersion, uint aFlags, uint aSizeInBytes)
            : base(aID, aVersion, aFlags, aSizeInBytes)
        {
            if (aSizeInBytes != 7 * sizeof(double))
            {
                throw new ArgumentException("Opcode parameter length doesn't match opcode.");
            }
            _coefficients = new Coefficients();

            _coefficients.K0 = aPefFile.ReadF8BE();
            _coefficients.K1 = aPefFile.ReadF8BE();
            _coefficients.K2 = aPefFile.ReadF8BE();
            _coefficients.K3 = aPefFile.ReadF8BE();
            _coefficients.K4 = aPefFile.ReadF8BE();
            _coefficients.Cx = aPefFile.ReadF8BE();
            _coefficients.Cy = aPefFile.ReadF8BE();

        }

        public override byte[] GetAsBytes()
        {
            MemoryStream ms = new MemoryStream();
            BinaryWriter bw = new BinaryWriter(ms);

            bw.Write(Endian_swap(_id));
            bw.Write(Endian_swap(_version));
            bw.Write(Endian_swap(_flags));
            bw.Write(Endian_swap(_sizeInBytes));


            bw.Write(Endian_swap(_coefficients.K0));
            bw.Write(Endian_swap(_coefficients.K1));
            bw.Write(Endian_swap(_coefficients.K2));
            bw.Write(Endian_swap(_coefficients.K3));
            bw.Write(Endian_swap(_coefficients.K4));
            bw.Write(Endian_swap(_coefficients.Cx));
            bw.Write(Endian_swap(_coefficients.Cy));

            bw.Flush();
            ms.Flush();
            byte[] ret = ms.ToArray();
            bw.Close();
            bw.Dispose();
            return ret;
        }
    }

    public class FixBadPixelsConstant : OpCode
    {
        private int _constant;
        private int _bayerPhase;

        public FixBadPixelsConstant(FileReader aPefFile, uint aID, uint aVersion, uint aFlags, uint aSizeInBytes)
            : base(aID, aVersion, aFlags, aSizeInBytes)
        {
            if (aSizeInBytes != 2 * sizeof(int))
            {
                throw new ArgumentException("Opcode parameter length doesn't match opcode.");
            }

            _constant = aPefFile.ReadI4BE();
            _bayerPhase = aPefFile.ReadI4BE();

        }

        public override byte[] GetAsBytes()
        {
            MemoryStream ms = new MemoryStream();
            BinaryWriter bw = new BinaryWriter(ms);

            bw.Write(Endian_swap(_id));
            bw.Write(Endian_swap(_version));
            bw.Write(Endian_swap(_flags));
            bw.Write(Endian_swap(_sizeInBytes));


            bw.Write(Endian_swap(_constant));
            bw.Write(Endian_swap(_bayerPhase));

            bw.Flush();
            ms.Flush();
            byte[] ret = ms.ToArray();
            bw.Close();
            bw.Dispose();
            return ret;
        }
    }

    public class FixBadPixelsList : OpCode
    {
        private List<BadPoint> _badPoints;
        private List<BadRect> _badRects;
        private int _bayerPhase;

        public struct BadPoint
        {
            public int Row;
            public int Column;
        }

        public struct BadRect
        {
            public int Top;
            public int Left;
            public int Bottom;
            public int Right;
        }

        public FixBadPixelsList(FileReader aPefFile, uint aID, uint aVersion, uint aFlags, uint aSizeInBytes)
            : base(aID, aVersion, aFlags, aSizeInBytes)
        {
            _bayerPhase = aPefFile.ReadI4BE();
            int badPointCount = aPefFile.ReadI4BE();
            int badRectCount = aPefFile.ReadI4BE();
            _badPoints = new List<BadPoint>();
            _badRects = new List<BadRect>();

            for (int i = 0; i < badPointCount; i++)
            {
                BadPoint p = new BadPoint();
                p.Row = aPefFile.ReadI4BE();
                p.Column = aPefFile.ReadI4BE();
                _badPoints.Add(p);
            }
            for (int i = 0; i < badRectCount; i++)
            {
                BadRect r = new BadRect();
                r.Top = aPefFile.ReadI4BE();
                r.Left = aPefFile.ReadI4BE();
                r.Bottom = aPefFile.ReadI4BE();
                r.Right = aPefFile.ReadI4BE();
                _badRects.Add(r);
            }
        }

        public override byte[] GetAsBytes()
        {
            MemoryStream ms = new MemoryStream();
            BinaryWriter bw = new BinaryWriter(ms);

            bw.Write(Endian_swap(_id));
            bw.Write(Endian_swap(_version));
            bw.Write(Endian_swap(_flags));
            bw.Write(Endian_swap(_sizeInBytes));

            bw.Write(Endian_swap(_bayerPhase));
            bw.Write(Endian_swap(_badPoints.Count));
            bw.Write(Endian_swap(_badRects.Count));

            for (int i = 0; i < _badPoints.Count; i++)
            {
                BadPoint p = _badPoints[i];
                bw.Write(Endian_swap(p.Row));
                bw.Write(Endian_swap(p.Column));
            }
            for (int i = 0; i < _badRects.Count; i++)
            {
                BadRect r = _badRects[i];
                bw.Write(Endian_swap(r.Top));
                bw.Write(Endian_swap(r.Left));
                bw.Write(Endian_swap(r.Bottom));
                bw.Write(Endian_swap(r.Right));
            }

            bw.Flush();
            ms.Flush();
            byte[] ret = ms.ToArray();
            bw.Close();
            bw.Dispose();
            return ret;
        }
    }

    public class TrimBounds : OpCode
    {
        private int _top;
        private int _left;
        private int _bottom;
        private int _right;

        public TrimBounds(FileReader aPefFile, uint aID, uint aVersion, uint aFlags, uint aSizeInBytes)
            : base(aID, aVersion, aFlags, aSizeInBytes)
        {
            if (aSizeInBytes != 4 * sizeof(int))
            {
                throw new ArgumentException("Opcode parameter length doesn't match opcode.");
            }

            _top = aPefFile.ReadI4BE();
            _left = aPefFile.ReadI4BE();
            _bottom = aPefFile.ReadI4BE();
            _right = aPefFile.ReadI4BE();
        }

        public override byte[] GetAsBytes()
        {
            MemoryStream ms = new MemoryStream();
            BinaryWriter bw = new BinaryWriter(ms);

            bw.Write(Endian_swap(_id));
            bw.Write(Endian_swap(_version));
            bw.Write(Endian_swap(_flags));
            bw.Write(Endian_swap(_sizeInBytes));

            bw.Write(Endian_swap(_top));
            bw.Write(Endian_swap(_left));
            bw.Write(Endian_swap(_bottom));
            bw.Write(Endian_swap(_right));

            bw.Flush();
            ms.Flush();
            byte[] ret = ms.ToArray();
            bw.Close();
            bw.Dispose();
            return ret;
        }
    }

    public class MapTable : OpCode
    {
        private int _top;
        private int _left;
        private int _bottom;
        private int _right;
        private int _plane;
        private int _planes;
        private int _rowPitch;
        private int _colPitch;
        private List<short> _tableEntries;


        public MapTable(FileReader aPefFile, uint aID, uint aVersion, uint aFlags, uint aSizeInBytes)
            : base(aID, aVersion, aFlags, aSizeInBytes)
        {
            _top = aPefFile.ReadI4BE();
            _left = aPefFile.ReadI4BE();
            _bottom = aPefFile.ReadI4BE();
            _right = aPefFile.ReadI4BE();
            _plane = aPefFile.ReadI4BE();
            _planes = aPefFile.ReadI4BE();
            _rowPitch = aPefFile.ReadI4BE();
            _colPitch = aPefFile.ReadI4BE();
            int tableSize = aPefFile.ReadI4BE();
            _tableEntries = new List<short>();
            for (int i = 0; i < tableSize; i++)
            {
                short t = aPefFile.ReadI2BE();
                _tableEntries.Add(t);
            }
        }

        public override byte[] GetAsBytes()
        {
            MemoryStream ms = new MemoryStream();
            BinaryWriter bw = new BinaryWriter(ms);

            bw.Write(Endian_swap(_id));
            bw.Write(Endian_swap(_version));
            bw.Write(Endian_swap(_flags));
            bw.Write(Endian_swap(_sizeInBytes));

            bw.Write(Endian_swap(_top));
            bw.Write(Endian_swap(_left));
            bw.Write(Endian_swap(_bottom));
            bw.Write(Endian_swap(_right));
            bw.Write(Endian_swap(_plane));
            bw.Write(Endian_swap(_planes));
            bw.Write(Endian_swap(_rowPitch));
            bw.Write(Endian_swap(_colPitch));
            bw.Write(Endian_swap(_tableEntries.Count));

            for (int i = 0; i < _tableEntries.Count; i++)
            {
                bw.Write(Endian_swap(_tableEntries[i]));
            }

            bw.Flush();
            ms.Flush();
            byte[] ret = ms.ToArray();
            bw.Close();
            bw.Dispose();
            return ret;
        }
    }

    public class MapPolynomial : OpCode
    {
        private int _top;
        private int _left;
        private int _bottom;
        private int _right;
        private int _plane;
        private int _planes;
        private int _rowPitch;
        private int _colPitch;
        private int _degree;
        private List<double> _coefficients;


        public MapPolynomial(FileReader aPefFile, uint aID, uint aVersion, uint aFlags, uint aSizeInBytes)
            : base(aID, aVersion, aFlags, aSizeInBytes)
        {
            _top = aPefFile.ReadI4BE();
            _left = aPefFile.ReadI4BE();
            _bottom = aPefFile.ReadI4BE();
            _right = aPefFile.ReadI4BE();
            _plane = aPefFile.ReadI4BE();
            _planes = aPefFile.ReadI4BE();
            _rowPitch = aPefFile.ReadI4BE();
            _colPitch = aPefFile.ReadI4BE();
            _degree = aPefFile.ReadI4BE();
            _coefficients = new List<double>();
            for (int i = 0; i < _degree+1; i++)
            {
                double t = aPefFile.ReadF8BE();
                _coefficients.Add(t);
            }
        }

        public override byte[] GetAsBytes()
        {
            MemoryStream ms = new MemoryStream();
            BinaryWriter bw = new BinaryWriter(ms);

            bw.Write(Endian_swap(_id));
            bw.Write(Endian_swap(_version));
            bw.Write(Endian_swap(_flags));
            bw.Write(Endian_swap(_sizeInBytes));

            bw.Write(Endian_swap(_top));
            bw.Write(Endian_swap(_left));
            bw.Write(Endian_swap(_bottom));
            bw.Write(Endian_swap(_right));
            bw.Write(Endian_swap(_plane));
            bw.Write(Endian_swap(_planes));
            bw.Write(Endian_swap(_rowPitch));
            bw.Write(Endian_swap(_colPitch));
            bw.Write(Endian_swap(_degree));

            for (int i = 0; i < _degree + 1; i++)
            {
                bw.Write(Endian_swap(_coefficients[i]));
            }

            bw.Flush();
            ms.Flush();
            byte[] ret = ms.ToArray();
            bw.Close();
            bw.Dispose();
            return ret;
        }
    }

    public class GainMap : OpCode
    {
        private int _top;
        private int _left;
        private int _bottom;
        private int _right;
        private int _plane;
        private int _planes;
        private int _rowPitch;
        private int _colPitch;
        private int _mapPointsV;
        private int _mapPointsH;
        private double _mapSpacingV;
        private double _mapSpacingH;
        private double _mapOriginV;
        private double _mapOriginH;
        private int _mapPlanes;
        private List<float> _mapGain;


        public GainMap(FileReader aPefFile, uint aID, uint aVersion, uint aFlags, uint aSizeInBytes)
            : base(aID, aVersion, aFlags, aSizeInBytes)
        {
            _top = aPefFile.ReadI4BE();
            _left = aPefFile.ReadI4BE();
            _bottom = aPefFile.ReadI4BE();
            _right = aPefFile.ReadI4BE();
            _plane = aPefFile.ReadI4BE();
            _planes = aPefFile.ReadI4BE();
            _rowPitch = aPefFile.ReadI4BE();
            _colPitch = aPefFile.ReadI4BE();
            _mapPointsV = aPefFile.ReadI4BE();
            _mapPointsH = aPefFile.ReadI4BE();
            _mapSpacingV = aPefFile.ReadF8BE();
            _mapSpacingH = aPefFile.ReadF8BE();
            _mapOriginV = aPefFile.ReadF8BE();
            _mapOriginH = aPefFile.ReadF8BE();
            _mapPlanes = aPefFile.ReadI4BE();
            
            _mapGain = new List<float>();
            for (int i = 0; i < _mapPointsV * _mapPointsH * _mapPlanes; i++)
            {
                float t = aPefFile.ReadF4BE();
                _mapGain.Add(t);
            }
        }

        public override byte[] GetAsBytes()
        {
            MemoryStream ms = new MemoryStream();
            BinaryWriter bw = new BinaryWriter(ms);

            bw.Write(Endian_swap(_id));
            bw.Write(Endian_swap(_version));
            bw.Write(Endian_swap(_flags));
            bw.Write(Endian_swap(_sizeInBytes));

            bw.Write(Endian_swap(_top));
            bw.Write(Endian_swap(_left));
            bw.Write(Endian_swap(_bottom));
            bw.Write(Endian_swap(_right));
            bw.Write(Endian_swap(_plane));
            bw.Write(Endian_swap(_planes));
            bw.Write(Endian_swap(_rowPitch));
            bw.Write(Endian_swap(_colPitch));
            bw.Write(Endian_swap(_mapPointsV));
            bw.Write(Endian_swap(_mapPointsH));
            bw.Write(Endian_swap(_mapSpacingV));
            bw.Write(Endian_swap(_mapSpacingH));
            bw.Write(Endian_swap(_mapOriginV));
            bw.Write(Endian_swap(_mapOriginH));
            bw.Write(Endian_swap(_mapPlanes));

            for (int i = 0; i < _mapPointsV * _mapPointsH * _mapPlanes; i++)
            {
                bw.Write(Endian_swap(_mapGain[i]));
            }

            bw.Flush();
            ms.Flush();
            byte[] ret = ms.ToArray();
            bw.Close();
            bw.Dispose();
            return ret;
        }
    }

    public class DeltaPerRow : OpCode
    {
        private int _top;
        private int _left;
        private int _bottom;
        private int _right;
        private int _plane;
        private int _planes;
        private int _rowPitch;
        private int _colPitch;

        private List<float> _deltas;


        public DeltaPerRow(FileReader aPefFile, uint aID, uint aVersion, uint aFlags, uint aSizeInBytes)
            : base(aID, aVersion, aFlags, aSizeInBytes)
        {
            _top = aPefFile.ReadI4BE();
            _left = aPefFile.ReadI4BE();
            _bottom = aPefFile.ReadI4BE();
            _right = aPefFile.ReadI4BE();
            _plane = aPefFile.ReadI4BE();
            _planes = aPefFile.ReadI4BE();
            _rowPitch = aPefFile.ReadI4BE();
            _colPitch = aPefFile.ReadI4BE();
            int count = aPefFile.ReadI4BE();

            _deltas = new List<float>();
            for (int i = 0; i < count; i++)
            {
                float t = aPefFile.ReadF4BE();
                _deltas.Add(t);
            }
        }

        public override byte[] GetAsBytes()
        {
            MemoryStream ms = new MemoryStream();
            BinaryWriter bw = new BinaryWriter(ms);

            bw.Write(Endian_swap(_id));
            bw.Write(Endian_swap(_version));
            bw.Write(Endian_swap(_flags));
            bw.Write(Endian_swap(_sizeInBytes));

            bw.Write(Endian_swap(_top));
            bw.Write(Endian_swap(_left));
            bw.Write(Endian_swap(_bottom));
            bw.Write(Endian_swap(_right));
            bw.Write(Endian_swap(_plane));
            bw.Write(Endian_swap(_planes));
            bw.Write(Endian_swap(_rowPitch));
            bw.Write(Endian_swap(_colPitch));
            bw.Write(Endian_swap(_deltas.Count));

            for (int i = 0; i < _deltas.Count; i++)
            {
                bw.Write(Endian_swap(_deltas[i]));
            }

            bw.Flush();
            ms.Flush();
            byte[] ret = ms.ToArray();
            bw.Close();
            bw.Dispose();
            return ret;
        }
    }

    public class DeltaPerColumn : OpCode
    {
        private int _top;
        private int _left;
        private int _bottom;
        private int _right;
        private int _plane;
        private int _planes;
        private int _rowPitch;
        private int _colPitch;

        private List<float> _deltas;


        public DeltaPerColumn(FileReader aPefFile, uint aID, uint aVersion, uint aFlags, uint aSizeInBytes)
            : base(aID, aVersion, aFlags, aSizeInBytes)
        {
            _top = aPefFile.ReadI4BE();
            _left = aPefFile.ReadI4BE();
            _bottom = aPefFile.ReadI4BE();
            _right = aPefFile.ReadI4BE();
            _plane = aPefFile.ReadI4BE();
            _planes = aPefFile.ReadI4BE();
            _rowPitch = aPefFile.ReadI4BE();
            _colPitch = aPefFile.ReadI4BE();
            int count = aPefFile.ReadI4BE();

            _deltas = new List<float>();
            for (int i = 0; i < count; i++)
            {
                float t = aPefFile.ReadF4BE();
                _deltas.Add(t);
            }
        }

        public override byte[] GetAsBytes()
        {
            MemoryStream ms = new MemoryStream();
            BinaryWriter bw = new BinaryWriter(ms);

            bw.Write(Endian_swap(_id));
            bw.Write(Endian_swap(_version));
            bw.Write(Endian_swap(_flags));
            bw.Write(Endian_swap(_sizeInBytes));

            bw.Write(Endian_swap(_top));
            bw.Write(Endian_swap(_left));
            bw.Write(Endian_swap(_bottom));
            bw.Write(Endian_swap(_right));
            bw.Write(Endian_swap(_plane));
            bw.Write(Endian_swap(_planes));
            bw.Write(Endian_swap(_rowPitch));
            bw.Write(Endian_swap(_colPitch));
            bw.Write(Endian_swap(_deltas.Count));

            for (int i = 0; i < _deltas.Count; i++)
            {
                bw.Write(Endian_swap(_deltas[i]));
            }

            bw.Flush();
            ms.Flush();
            byte[] ret = ms.ToArray();
            bw.Close();
            bw.Dispose();
            return ret;
        }
    }

    public class ScalePerRow : OpCode
    {
        private int _top;
        private int _left;
        private int _bottom;
        private int _right;
        private int _plane;
        private int _planes;
        private int _rowPitch;
        private int _colPitch;

        private List<float> _scale;


        public ScalePerRow(FileReader aPefFile, uint aID, uint aVersion, uint aFlags, uint aSizeInBytes)
            : base(aID, aVersion, aFlags, aSizeInBytes)
        {
            _top = aPefFile.ReadI4BE();
            _left = aPefFile.ReadI4BE();
            _bottom = aPefFile.ReadI4BE();
            _right = aPefFile.ReadI4BE();
            _plane = aPefFile.ReadI4BE();
            _planes = aPefFile.ReadI4BE();
            _rowPitch = aPefFile.ReadI4BE();
            _colPitch = aPefFile.ReadI4BE();
            int count = aPefFile.ReadI4BE();

            _scale = new List<float>();
            for (int i = 0; i < count; i++)
            {
                float t = aPefFile.ReadF4BE();
                _scale.Add(t);
            }
        }

        public override byte[] GetAsBytes()
        {
            MemoryStream ms = new MemoryStream();
            BinaryWriter bw = new BinaryWriter(ms);

            bw.Write(Endian_swap(_id));
            bw.Write(Endian_swap(_version));
            bw.Write(Endian_swap(_flags));
            bw.Write(Endian_swap(_sizeInBytes));

            bw.Write(Endian_swap(_top));
            bw.Write(Endian_swap(_left));
            bw.Write(Endian_swap(_bottom));
            bw.Write(Endian_swap(_right));
            bw.Write(Endian_swap(_plane));
            bw.Write(Endian_swap(_planes));
            bw.Write(Endian_swap(_rowPitch));
            bw.Write(Endian_swap(_colPitch));
            bw.Write(Endian_swap(_scale.Count));

            for (int i = 0; i < _scale.Count; i++)
            {
                bw.Write(Endian_swap(_scale[i]));
            }

            bw.Flush();
            ms.Flush();
            byte[] ret = ms.ToArray();
            bw.Close();
            bw.Dispose();
            return ret;
        }
    }

    public class ScalePerColumn : OpCode
    {
        private int _top;
        private int _left;
        private int _bottom;
        private int _right;
        private int _plane;
        private int _planes;
        private int _rowPitch;
        private int _colPitch;

        private List<float> _scale;


        public ScalePerColumn(FileReader aPefFile, uint aID, uint aVersion, uint aFlags, uint aSizeInBytes)
            : base(aID, aVersion, aFlags, aSizeInBytes)
        {
            _top = aPefFile.ReadI4BE();
            _left = aPefFile.ReadI4BE();
            _bottom = aPefFile.ReadI4BE();
            _right = aPefFile.ReadI4BE();
            _plane = aPefFile.ReadI4BE();
            _planes = aPefFile.ReadI4BE();
            _rowPitch = aPefFile.ReadI4BE();
            _colPitch = aPefFile.ReadI4BE();
            int count = aPefFile.ReadI4BE();

            _scale = new List<float>();
            for (int i = 0; i < count; i++)
            {
                float t = aPefFile.ReadF4BE();
                _scale.Add(t);
            }
        }

        public override byte[] GetAsBytes()
        {
            MemoryStream ms = new MemoryStream();
            BinaryWriter bw = new BinaryWriter(ms);

            bw.Write(Endian_swap(_id));
            bw.Write(Endian_swap(_version));
            bw.Write(Endian_swap(_flags));
            bw.Write(Endian_swap(_sizeInBytes));

            bw.Write(Endian_swap(_top));
            bw.Write(Endian_swap(_left));
            bw.Write(Endian_swap(_bottom));
            bw.Write(Endian_swap(_right));
            bw.Write(Endian_swap(_plane));
            bw.Write(Endian_swap(_planes));
            bw.Write(Endian_swap(_rowPitch));
            bw.Write(Endian_swap(_colPitch));
            bw.Write(Endian_swap(_scale.Count));

            for (int i = 0; i < _scale.Count; i++)
            {
                bw.Write(Endian_swap(_scale[i]));
            }

            bw.Flush();
            ms.Flush();
            byte[] ret = ms.ToArray();
            bw.Close();
            bw.Dispose();
            return ret;
        }
    }

    public class Unknown : OpCode
    {
        private byte[] _data;

        public Unknown(FileReader aPefFile, uint aID, uint aVersion, uint aFlags, uint aSizeInBytes)
            : base(aID, aVersion, aFlags, aSizeInBytes)
        {
            _data = new byte[aSizeInBytes];
            for (uint i = 0; i < aSizeInBytes; i++)
            {
                _data[i] = aPefFile.ReadUI1();
            }
        }

        public override byte[] GetAsBytes()
        {
            MemoryStream ms = new MemoryStream();
            BinaryWriter bw = new BinaryWriter(ms);

            bw.Write(Endian_swap(_id));
            bw.Write(Endian_swap(_version));
            bw.Write(Endian_swap(_flags));
            bw.Write(Endian_swap(_sizeInBytes));
            bw.Write(_data);


            bw.Flush();
            ms.Flush();
            byte[] ret = ms.ToArray();
            bw.Close();
            bw.Dispose();
            return ret;
        }
    }


}
