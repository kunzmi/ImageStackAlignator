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

namespace PentaxPefFile
{
	public enum TIFFValueTypes : ushort
	{
		Byte = 1,
		Ascii = 2,
		UnsignedShort = 3,
		UnsignedLong = 4,
		Rational = 5,
		SignedByte = 6,
		Undefined = 7,
		SignedShort = 8,
		SignedLong = 9,
		SignedRational = 10,
		Float = 11,
		Double = 12
	}

	public struct Rational
	{
		public uint numerator;
		public uint denominator;

		public Rational(uint[] aValues)
		{
			if (aValues.Length < 2)
				throw new ArgumentException("aValues must contain at least two values.");

			numerator = aValues[0];
			denominator = aValues[1];
		}

		public Rational(uint aNominator, uint aDenominator)
		{
			numerator = aNominator;
			denominator = aDenominator;
		}

		public override string ToString()
		{
			return numerator.ToString() + " / " + denominator.ToString();
		}

        public double Value
        {
            get { return (double)numerator / (double)denominator; }
        }
	}

	public struct SRational
	{
		public int numerator;
		public int denominator;

		public SRational(int[] aValues)
		{
			if (aValues.Length < 2)
				throw new ArgumentException("aValues must contain at least two values.");

			numerator = aValues[0];
			denominator = aValues[1];
		}

		public SRational(int aNominator, int aDenominator)
		{
			numerator = aNominator;
			denominator = aDenominator;
		}

		public override string ToString()
		{
			return numerator.ToString() + " / " + denominator.ToString();
        }

        public double Value
        {
            get { return (double)numerator / (double)denominator; }
        }
    }

	public struct TIFFValueType
	{
		TIFFValueTypes mType;
		public TIFFValueType(TIFFValueTypes aType)
		{
			mType = aType;
		}

		public int SizeInBytes
		{
			get 
			{
				switch (mType)
				{
					case TIFFValueTypes.Byte:
						return 1;
					case TIFFValueTypes.Ascii:
						return 1;
					case TIFFValueTypes.UnsignedShort:
						return 2;
					case TIFFValueTypes.UnsignedLong:
						return 4;
					case TIFFValueTypes.Rational:
						return 8;
					case TIFFValueTypes.SignedByte:
						return 1;
					case TIFFValueTypes.Undefined:
						return 1;
					case TIFFValueTypes.SignedShort:
						return 2;
					case TIFFValueTypes.SignedLong:
						return 4;
					case TIFFValueTypes.SignedRational:
						return 8;
					case TIFFValueTypes.Float:
						return 4;
					case TIFFValueTypes.Double:
						return 8;
					default:
                        throw new ArgumentException(mType.ToString() + " is not a valid type."); //Never happens...
				}
			}
		}

        public ushort GetValue()
        {
            return (ushort)mType;
        }
	}
}
