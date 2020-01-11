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
using System.Globalization;

namespace PentaxPefFile
{
    //More or less a 1:1 copy of the matrix classes in Adobe DNG SDK
    //https://www.adobe.com/support/downloads/dng/dng_sdk.html

    public class DNGMatrix
    {
        protected const int MAX_COLOR_PLANES = 4;
        protected const double NEAR_ZERO = 1.0E-10;
        protected uint _rows;
        protected uint _cols;

        protected double[,] _data = new double[MAX_COLOR_PLANES, MAX_COLOR_PLANES];

        public DNGMatrix()
        {
        }

        public DNGMatrix(uint aRows, uint aCols)
        {
            if (aRows > MAX_COLOR_PLANES || aCols > MAX_COLOR_PLANES)
            {
                throw new ArgumentOutOfRangeException("Matrix dimensions must be smaller or equal to " + MAX_COLOR_PLANES);
            }

            _rows = aRows;
            _cols = aCols;
        }

        public DNGMatrix(DNGMatrix aMatrix)
        {
            _rows = aMatrix.Rows;
            _cols = aMatrix.Cols;

            for (int row = 0; row < _rows; row++)
            {
                for (int col = 0; col < _cols; col++)
                {
                    _data[row, col] = aMatrix._data[row, col];
                }
            }
        }

        public static DNGMatrix Identity(uint size)
        {
            DNGMatrix mat = new DNGMatrix();
            mat.SetIdentity(size);
            return mat;
        }

        public uint Rows
        {
            get { return _rows; }
        }

        public uint Cols
        {
            get { return _cols; }
        }

        #region Override Methods
        public override bool Equals(object obj)
        {
            if (obj == null) return false;
            if (!(obj is DNGMatrix)) return false;

            DNGMatrix value = (DNGMatrix)obj;

            return Equals(value);
        }

        public bool Equals(DNGMatrix value)
        {
            if (_rows != value._rows || _cols != value._cols)
            {
                return false;
            }

            for (int row = 0; row < _rows; row++)
            {
                for (int col = 0; col < _cols; col++)
                {
                    if (_data[row, col] != value._data[row, col])
                    {
                        return false;
                    }
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
            sb.Append("{");
            for (int row = 0; row < _rows; row++)
            {
                sb.Append("{");
                for (int col = 0; col < _cols; col++)
                {
                    sb.Append(_data[row, col].ToString());
                    if (col < _cols - 1)
                    {
                        sb.Append("; ");
                    }
                }
                sb.Append("}");
            }
            sb.Append("}");
            return sb.ToString();
        }
        #endregion

        public void Clear()
        {
            for (int row = 0; row < MAX_COLOR_PLANES; row++)
            {
                for (int col = 0; col < MAX_COLOR_PLANES; col++)
                {
                    _data[row, col] = 0;
                }
            }
            _rows = 0;
            _cols = 0;
        }

        public void SetIdentity(uint count)
        {
            if (count > MAX_COLOR_PLANES)
            {
                throw new ArgumentOutOfRangeException("Matrix dimensions must be smaller or equal to " + MAX_COLOR_PLANES);
            }

            Clear();
            _rows = count;
            _cols = count;

            for (int i = 0; i < count; i++)
            {
                _data[i, i] = 1.0;
            }
        }

        public double this[uint row, uint col]
        {
            get { return _data[row, col]; }
            set { _data[row, col] = value; }
        }

        //for convenience with NPP
        public float[,] GetAs3x4Array()
        {
            float[,] colorTwist = new float[3, 4];
            for (uint i = 0; i < 3; i++)
            {
                for (uint j = 0; j < 3; j++)
                {
                    colorTwist[i, j] = (float)this[i, j];
                }
            }
            return colorTwist;
        }

        #region Operators

        public static bool operator ==(DNGMatrix a, DNGMatrix b)
        {
            if (object.ReferenceEquals(a, null))
            {
                return object.ReferenceEquals(b, null);
            }
            if (object.ReferenceEquals(b, null))
                return false;
            return a.Equals(b);
        }

        public static bool operator !=(DNGMatrix a, DNGMatrix b)
        {
            if (object.ReferenceEquals(a, null))
            {
                return !object.ReferenceEquals(b, null);
            }
            if (object.ReferenceEquals(b, null))
                return true;

            return !a.Equals(b);
        }
        
        public static DNGMatrix operator *(double scale, DNGMatrix vec)
        {
            DNGMatrix ret = new DNGMatrix(vec);
            ret.Scale(scale);
            return ret;
        }

        public static DNGMatrix operator *(DNGMatrix vec, double scale)
        {
            DNGMatrix ret = new DNGMatrix(vec);
            ret.Scale(scale);
            return ret;
        }
        public static DNGMatrix operator *(DNGMatrix a, DNGMatrix b)
        {
            if (a.Cols != b.Rows)
            {
                throw new ArgumentException("Matrix dimensions don't match.");
            }

            DNGMatrix ret = new DNGMatrix(a.Rows, b.Cols);

            for (uint j = 0; j < ret.Rows; j++)
                for (uint k = 0; k < ret.Cols; k++)
                {
                    ret[j,k] = 0.0;
                    for (uint m = 0; m < a.Cols; m++)
                    {
                        double aa = a[j,m];
                        double bb = b[m,k];
                        ret[j,k] += aa * bb;
                    }

                }
            return ret;
        }
        public static DNGMatrix operator +(DNGMatrix a, DNGMatrix b)
        {
            if (a.Cols != b.Cols || a.Rows != b.Rows)
            {
                throw new ArgumentException("Matrices don't have the same size.");
            }

            DNGMatrix ret = new DNGMatrix(a);

            for (uint j = 0; j < ret.Rows; j++)
                for (uint k = 0; k < ret.Cols; k++)
                {
                    ret[j,k] += b[j,k];
                }
            return ret;
        }
        #endregion

        public bool IsEmpty()
        {
            return _rows == 0 || _cols == 0;
        }

        public bool NotEmpty()
        {
            return !IsEmpty();
        }

        public bool IsDiagonal()
        {
            if (IsEmpty())
            {
                return false;
            }

            if (_rows != _cols)
            {
                return false;
            }

            for (uint j = 0; j < _rows; j++)
                for (uint k = 0; k < _cols; k++)
                {
                    if (j != k)
                    {
                        if (_data[j, k] != 0.0)
                        {
                            return false;
                        }

                    }
                }

            return true;
        }

        public bool IsIdentity()
        {
            if (IsDiagonal())
            {
                for (uint j = 0; j < _rows; j++)
                {
                    if (_data[j, j] != 1.0)
                    {
                        return false;
                    }
                }
                return true;
            }
            return false;
        }


        public double MaxEntry()
        {
            if (IsEmpty())
            {

                return 0.0;

            }

            double m = _data[0, 0];

            for (uint j = 0; j < _rows; j++)
                for (uint k = 0; k < _cols; k++)
                {

                    m = Math.Max(m, _data[j, k]);

                }

            return m;
        }

        public double MinEntry()
        {
            if (IsEmpty())
            {

                return 0.0;

            }

            double m = _data[0, 0];

            for (uint j = 0; j < _rows; j++)
                for (uint k = 0; k < _cols; k++)
                {

                    m = Math.Min(m, _data[j, k]);

                }

            return m;
        }

        public void Scale(double factor)
        {
            for (uint j = 0; j < _rows; j++)
                for (uint k = 0; k < _cols; k++)
                {

                    _data[j, k] *= factor;

                }
        }

        public void Round(double factor)
        {
            double invFactor = 1.0 / factor;

            for (uint j = 0; j < _rows; j++)
                for (uint k = 0; k < _cols; k++)
                {

                    _data[j, k] = Math.Round(_data[j, k] * factor) * invFactor;

                }
        }

        public void SafeRound(double factor)
        {
            double invFactor = 1.0 / factor;

            for (uint j = 0; j < _rows; j++)
            {
                // Round each row to the specified accuracy, but make sure the
                // a rounding does not affect the total of the elements in a row
                // more than necessary.
                double error = 0.0;
                for (uint k = 0; k < _cols; k++)
                {
                    _data[j, k] += error;
                    double rounded = Math.Round(_data[j, k] * factor) * invFactor;
                    error = _data[j, k] - rounded;
                    _data[j, k] = rounded;
                }
            }
        }

        public bool AlmostEqual(DNGMatrix m, double slop = 1.0e-8)
        {
            if (_rows != m._rows || _cols != m._cols)
            {
                return false;
            }

            for (uint j = 0; j < _rows; j++)
            {
                for (uint k = 0; k < _cols; k++)
                {
                    if (Math.Abs(_data[j, k] - m[j, k]) > slop)
                    {
                        return false;
                    }
                }
            }
            return true;
        }

        public bool AlmostIdentity(double slop = 1.0e-8)
        {
            return AlmostEqual(DNGMatrix.Identity(_rows), slop);
        }

        private DNGMatrix Invert3x3()
        {
            if (Cols != 3 && Rows != 3)
            {
                throw new Exception("This only works on 3x3 matrices.");
            }

            double a00 = this[0, 0];
            double a01 = this[0, 1];
            double a02 = this[0, 2];
            double a10 = this[1, 0];
            double a11 = this[1, 1];
            double a12 = this[1, 2];
            double a20 = this[2, 0];
            double a21 = this[2, 1];
            double a22 = this[2, 2];

            double[,] temp = new double[3, 3];

	        temp[0, 0] = a11* a22 - a21* a12;
            temp[0, 1] = a21* a02 - a01* a22;
            temp[0, 2] = a01* a12 - a11* a02;
            temp[1, 0] = a20* a12 - a10* a22;
            temp[1, 1] = a00* a22 - a20* a02;
            temp[1, 2] = a10* a02 - a00* a12;
            temp[2, 0] = a10* a21 - a20* a11;
            temp[2, 1] = a20* a01 - a00* a21;
            temp[2, 2] = a00* a11 - a10* a01;

            double det = (a00 * temp[0, 0] +
                          a01 * temp[1, 0] +
                          a02 * temp[2, 0]);

	        if (Math.Abs(det) < NEAR_ZERO)
		    {		
		        throw new Exception("The matrix determinant is too close to zero.");
            }

            DNGMatrix B = new DNGMatrix(3, 3);

            for (uint j = 0; j < 3; j++)
                for (uint k = 0; k < 3; k++)
                {
                    B[j, k] = temp[j, k] / det;
                }
			
	        return B;
        }

        private DNGMatrix InvertNxN()
        {
            uint i;
            uint j;
            uint k;

            uint n = Rows;
            uint augmented_cols = 2 * n;

            double[,] temp = new double[MAX_COLOR_PLANES, MAX_COLOR_PLANES * 2];

	
	        for (i = 0; i<n; i++)
		        for (j = 0; j<n; j++)
			    {			
			        temp[i, j] = this[i, j];			
			        temp[i, j + n] = (i == j? 1.0 : 0.0);			
			    }
			
	        for (i = 0; i<n; i++)
		    {
		        // Find row iMax with largest absolute entry in column i.
		        uint iMax = i;
                double vMax = -1.0;

		        for (k = i; k<n; k++)
			    {			
			        double v = Math.Abs(this[k, i]);

			        if (v > vMax)
				    {
				        vMax = v;
				        iMax = k;
				    }			
			    }

		        double alpha = temp[iMax, i];
		
		        if (Math.Abs(alpha) < NEAR_ZERO)
			    {			
			        throw new Exception("This matrix is not invertible");						 
			    }
			
		        // Swap rows i and iMax, column by column.
		        if (i != iMax)
			    {
			        for (j = 0; j<augmented_cols; j++)
			        {
                        double t = temp[iMax, j];
                        temp[iMax, j] = temp[i, j];
                        temp[i, j] = t;
				    }
			    }

		        for (j = 0; j<augmented_cols; j++)
			    {			
			        temp[i, j] /= alpha;			
			    }
			
		        for (k = 0; k<n; k++)
			    {			
			        if (i != k)
				    {				
				        double beta = temp[k, i];
				
				        for (j = 0; j<augmented_cols; j++)
					    {					
					        temp[k, j] -= beta* temp[i, j];					
					    }				
				    }			
		        }			
		    }
		
	        DNGMatrix B = new DNGMatrix(n, n);
	
	        for (i = 0; i<n; i++)
		        for (j = 0; j<n; j++)
			    {			
			        B[i, j] = temp[i, j + n];
			    }
			
	        return B;
        }

        public DNGMatrix Transpose()
        {
            DNGMatrix B = new DNGMatrix(Cols, Rows);

            for (uint j = 0; j < B.Rows; j++)
                for (uint k = 0; k < B.Cols; k++)
                {
                    B[j, k] = this[k, j];
                }

            return B;
        }

        public DNGMatrix Invert()
        {
            if (Rows < 2 || Cols < 2)
            {
                throw new Exception("Can't invert a matrix smaller than 2x2");
            }

            if (Rows == Cols)
            {
                if (Rows == 3)
                {
                    return Invert3x3();
                }
                return InvertNxN();
            }
            else
            {
                // Compute the pseudo inverse.
                DNGMatrix B = Transpose();
                return (B * this).Invert() * B;
            }
        }

        public DNGMatrix Invert(DNGMatrix hint)
        {
            if (Rows == Cols ||
                Rows != hint.Cols ||
                Cols != hint.Rows)
            {
                return Invert();
            }
            else
            {
                // Use the specified hint matrix.
                return (hint * this).Invert() * hint;
            }
        }
    }

    public class DNGMatrix3x3 : DNGMatrix
    {
        public DNGMatrix3x3() 
            : base(3,3)
        { }

        public DNGMatrix3x3(DNGMatrix m)
            : base(m)
        {
            if (Rows != 3 || Cols != 3)
            {
                throw new ArgumentException("This is not a 3x3 Matrix");
            }
        }

        public DNGMatrix3x3(double a00, double a01, double a02,
                         double a10, double a11, double a12,
                         double a20, double a21, double a22)
            : base(3, 3)
        {
            _data[0, 0] = a00;
            _data[0, 1] = a01;
            _data[0, 2] = a02;

            _data[1, 0] = a10;
            _data[1, 1] = a11;
            _data[1, 2] = a12;

            _data[2, 0] = a20;
            _data[2, 1] = a21;
            _data[2, 2] = a22;
        }

        public DNGMatrix3x3(double[] aValues)
            : base(3, 3)
        {
            _data[0, 0] = aValues[0];
            _data[0, 1] = aValues[1];
            _data[0, 2] = aValues[2];

            _data[1, 0] = aValues[3];
            _data[1, 1] = aValues[4];
            _data[1, 2] = aValues[5];

            _data[2, 0] = aValues[6];
            _data[2, 1] = aValues[7];
            _data[2, 2] = aValues[8];
        }

        public DNGMatrix3x3(double a00, double a11, double a22)
            : base(3, 3)
        {
            _data[0, 0] = a00;
            _data[1, 1] = a11;
            _data[2, 2] = a22;
        }
    }

    public class DNGMatrix4x3 : DNGMatrix
    {
        public DNGMatrix4x3()
            : base(4, 3)
        { }

        public DNGMatrix4x3(DNGMatrix m)
            : base(m)
        {
            if (Rows != 4 || Cols != 3)
            {
                throw new ArgumentException("This is not a 4x3 Matrix");
            }
        }

        public DNGMatrix4x3(double a00, double a01, double a02,
                            double a10, double a11, double a12,
                            double a20, double a21, double a22,
                            double a30, double a31, double a32)
            : base(4, 3)
        {
            _data[0, 0] = a00;
            _data[0, 1] = a01;
            _data[0, 2] = a02;

            _data[1, 0] = a10;
            _data[1, 1] = a11;
            _data[1, 2] = a12;

            _data[2, 0] = a20;
            _data[2, 1] = a21;
            _data[2, 2] = a22;

            _data[3, 0] = a30;
            _data[3, 1] = a31;
            _data[3, 2] = a32;
        }
    }

    public class DNGMatrix4x4 : DNGMatrix
    {
        public DNGMatrix4x4()
            : base(4, 4)
        { }

        public DNGMatrix4x4(DNGMatrix m)
            : base(m)
        {
            // Input must be either 3x3 or 4x4.

            bool is3by3 = (m.Rows == 3 && m.Cols == 3);
            bool is4by4 = (m.Rows == 4 && m.Cols == 4);

            if (!is3by3 && !is4by4)
            {
                throw new ArgumentException("This is not a 3x3 or 4x4 Matrix");
            }

            // For 3x3 case, pad to 4x4 (equivalent 4x4 matrix).
            if (is3by3)
            {
                _rows = 4;
                _cols = 4;

                _data[0, 3] = 0.0;
                _data[1, 3] = 0.0;
                _data[2, 3] = 0.0;

                _data[3, 0] = 0.0;
                _data[3, 1] = 0.0;
                _data[3, 2] = 0.0;

                _data[3, 3] = 1.0;
            }
        }

        public DNGMatrix4x4(double a00, double a01, double a02, double a03,
                            double a10, double a11, double a12, double a13,
                            double a20, double a21, double a22, double a23,
                            double a30, double a31, double a32, double a33)
            : base(4, 4)
        {
            _data[0, 0] = a00;
            _data[0, 1] = a01;
            _data[0, 2] = a02;
            _data[0, 3] = a03;

            _data[1, 0] = a10;
            _data[1, 1] = a11;
            _data[1, 2] = a12;
            _data[1, 3] = a13;

            _data[2, 0] = a20;
            _data[2, 1] = a21;
            _data[2, 2] = a22;
            _data[2, 3] = a23;

            _data[3, 0] = a30;
            _data[3, 1] = a31;
            _data[3, 2] = a32;
            _data[3, 3] = a33;
        }

        public DNGMatrix4x4(double a00, double a11, double a22, double a33)
            : base(4, 4)
        {
            _data[0, 0] = a00;
            _data[1, 1] = a11;
            _data[2, 2] = a22;
            _data[3, 3] = a33;
        }
    }
}
