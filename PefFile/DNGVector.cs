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
    //More or less a 1:1 copy of the vector classes in Adobe DNG SDK
    //https://www.adobe.com/support/downloads/dng/dng_sdk.html

    public class DNGVector
    {
        protected const int MAX_COLOR_PLANES = 4;
        protected uint _count;

        protected double[] _data = new double[MAX_COLOR_PLANES];

        public DNGVector()
        {
        }

        public DNGVector(uint aCount)
        {
            if (aCount > MAX_COLOR_PLANES)
            {
                throw new ArgumentOutOfRangeException("Vector dimensions must be smaller or equal to " + MAX_COLOR_PLANES);
            }

            _count = aCount;
        }

        public DNGVector(DNGVector aVector)
        {
            _count = aVector.Count;

            for (uint index = 0; index < _count; index++)
            {
                _data[index] = aVector._data[index];
            }
        }

        public static DNGVector Identity(uint size)
        {
            DNGVector mat = new DNGVector();
            mat.SetIdentity(size);
            return mat;
        }

        public uint Count
        {
            get { return _count; }
        }

        #region Override Methods
        public override bool Equals(object obj)
        {
            if (obj == null) return false;
            if (!(obj is DNGVector)) return false;

            DNGVector value = (DNGVector)obj;

            return Equals(value);
        }

        public bool Equals(DNGVector value)
        {
            if (_count != value._count)
            {
                return false;
            }

            for (uint index = 0; index < _count; index++)
            {
                if (_data[index] != value._data[index])
                {
                    return false;
                }                
            }
            return true;
        }

        public override int GetHashCode()
        {
            return _data.GetHashCode();
        }

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            sb.Append("(");
            for (int index = 0; index < _count; index++)
            {
                sb.Append(_data[index].ToString("0.000000"));
                if (index < _count - 1)
                {
                    sb.Append("; ");
                }
            }
            sb.Append(")");
            return sb.ToString();
        }
        #endregion

        public void Clear()
        {
            for (int index = 0; index < MAX_COLOR_PLANES; index++)
            {
                _data[index] = 0;                
            }
            _count = 0;
        }

        public void SetIdentity(uint count)
        {
            if (count > MAX_COLOR_PLANES)
            {
                throw new ArgumentOutOfRangeException("Vector size must be smaller or equal to " + MAX_COLOR_PLANES);
            }

            Clear();
            _count = count;

            for (int i = 0; i < count; i++)
            {
                _data[i] = 1.0;
            }
        }

        public double this[uint index]
        {
            get { return _data[index]; }
            set { _data[index] = value; }
        }

        public static double Dot(DNGVector a, DNGVector b)
        {
            uint count = a.Count;
            if (b.Count != count)
            {
                throw new ArgumentException("Vectors don't have same size.");
            }

            double sum = 0.0;
            for (uint i = 0; i < count; i++)
            {
                sum += a[i] * b[i];
            }

            return sum;
        }

        public static double Distance(DNGVector a, DNGVector b)
        {
            DNGVector c = a - b;

            return Math.Sqrt(Dot(c, c));
        }

        #region Operators

        public static bool operator ==(DNGVector a, DNGVector b)
        {
            return a.Equals(b);
        }

        public static bool operator !=(DNGVector a, DNGVector b)
        {
            return !a.Equals(b);
        }

        public static DNGVector operator *(DNGMatrix a, DNGVector b)
        {
            if (a.Cols != b.Count)
            {
                throw new ArgumentException("Matrix/Vector dimensions don't match.");
            }

            DNGVector ret = new DNGVector(a.Rows);

            for (uint j = 0; j < ret.Count; j++)
            {
                ret[j] = 0.0;
                for (uint m = 0; m < a.Cols; m++)
                {
                    double aa = a[j, m];
                    double bb = b[m];
                    ret[j] += aa * bb;
                }
            }

            return ret;
        }

        public static DNGVector operator *(double scale, DNGVector vec)
        {
            DNGVector ret = new DNGVector(vec);
            ret.Scale(scale);
            return ret;
        }

        public static DNGVector operator *(DNGVector vec, double scale)
        {
            DNGVector ret = new DNGVector(vec);
            ret.Scale(scale);
            return ret;
        }

        public static DNGVector operator +(DNGVector a, DNGVector b)
        {
            uint count = a.Count;
            if (b.Count != count)
            {
                throw new ArgumentException("Vectors don't have same size.");
            }

            DNGVector ret = new DNGVector(count);

            for (uint i = 0; i < count; i++)
            {
                ret[i] = a[i] + b[i];
            }

            return ret;
        }

        public static DNGVector operator -(DNGVector a, DNGVector b)
        {
            uint count = a.Count;
            if (b.Count != count)
            {
                throw new ArgumentException("Vectors don't have same size.");
            }

            DNGVector ret = new DNGVector(count);

            for (uint i = 0; i < count; i++)
            {
                ret[i] = a[i] - b[i];
            }

            return ret;
        }
        #endregion

        public bool IsEmpty()
        {
            return _count == 0;
        }

        public bool NotEmpty()
        {
            return !IsEmpty();
        }


        public double MaxEntry()
        {
            if (IsEmpty())
            {

                return 0.0;

            }

            return _data.Max();
        }

        public double MinEntry()
        {
            if (IsEmpty())
            {

                return 0.0;

            }

            return _data.Min();
        }

        public void Scale(double factor)
        {
            for (uint index = 0; index < _count; index++)
            {
                _data[index] *= factor;
            }                
        }

        public void Round(double factor)
        {
            double invFactor = 1.0 / factor;

            for (uint index = 0; index < _count; index++)
            {
                _data[index] = Math.Round(_data[index] * factor) * invFactor;
            }
        }

        public DNGMatrix AsDiagonal()
	    {
            DNGMatrix M = new DNGMatrix(Count, Count);
	
	        for (uint j = 0; j<Count; j++)
		    {		
		        M[j,j] = _data[j];		
		    }
		
	        return M;
        }

        public DNGMatrix AsColumn()
        {
            DNGMatrix M = new DNGMatrix(Count, 1);

            for (uint j = 0; j < Count; j++)
            {
                M[j, 0] = _data[j];
            }

            return M;
        }
    }

    public class DNGVector3 : DNGVector
    {
        public DNGVector3()
            : base(3)
        {

        }
        public DNGVector3(DNGVector v)
            : base(v)
        {
            if (Count != 3)
            {
                throw new ArgumentException("This is not a three element vector.");
            }
        }
        public DNGVector3(double a0, double a1, double a2)
            : base(3)
        {
            _data[0] = a0;
            _data[1] = a1;
            _data[2] = a2;
        }
    }

    public class DNGVector4 : DNGVector
    {
        public DNGVector4()
            : base(4)
        {

        }
        public DNGVector4(DNGVector v)
            : base(v)
        {
            if (Count != 4)
            {
                throw new ArgumentException("This is not a four element vector.");
            }
        }
        public DNGVector4(double a0, double a1, double a2, double a3)
            : base(4)
        {
            _data[0] = a0;
            _data[1] = a1;
            _data[2] = a2;
            _data[3] = a3;
        }
    }
}
