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

namespace PEFStudioDX
{
    class Matrix3x3
    {
        private double[] data;
        private int cols = 3;
        private int rows = 3;

        public Matrix3x3()
        {
            data = new double[9];
        }

        public double this[int row, int col]
        {
            get { return data[col + row * cols]; }
            set { data[col + row * cols] = value; }
        }

        public static Matrix3x3 Mul(Matrix3x3 src, Matrix3x3 value)
        {
            Matrix3x3 res = new Matrix3x3();
            for (int retx = 0; retx < src.rows; retx++)
                for (int rety = 0; rety < value.cols; rety++)
                {
                    double val = 0;
                    for (int i = 0; i < src.cols; i++)
                    {
                        val += src[retx, i] * value[i, rety];
                    }
                    res[retx, rety] = val;
                }
            return res;
        }

        public static Matrix3x3 operator *(Matrix3x3 src, Matrix3x3 value)
        {
            return Matrix3x3.Mul(src, value);
        }
        public static Matrix3x3 Unit()
        {
            Matrix3x3 res = new Matrix3x3();
            res[0, 0] = 1;
            res[1, 1] = 1;
            res[2, 2] = 1;
            return res;
        }

        public static Matrix3x3 ShiftAffine(double x, double y)
        {
            Matrix3x3 res = Unit();
            res[0, 2] = x;
            res[1, 2] = y;
            return res;
        }

        public static Matrix3x3 Rotation(double angInDeg)
        {
            Matrix3x3 res = Unit();
            double s = Math.Sin(angInDeg / 180.0 * Math.PI);
            double c = Math.Cos(angInDeg / 180.0 * Math.PI);
            res[0, 0] = c;
            res[1, 0] = -s;
            res[0, 1] = s;
            res[1, 1] = c;
            return res;
        }

        public static Matrix3x3 RotAroundCenter(double angle, double width, double height)
        {
            Matrix3x3 res = Matrix3x3.ShiftAffine(width / 2.0, height / 2.0);
            res = res * Matrix3x3.Rotation(angle);
            res = res * Matrix3x3.ShiftAffine(-width / 2.0, -height / 2.0);
            return res;
        }

        public override string ToString()
        {
            string res = "{";
            for (int c = 0; c < cols; c++)
            {
                for (int r = 0; r < rows; r++)
                {
                    res += this[r, c].ToString("0.000");
                    if (r < rows - 1)
                        res += " ";
                    else
                        res += "; ";
                }
            }
            res += "}";
            return res;
        }

        public double[,] ToAffine()
        {
            double[,] res = new double[2, 3];
            for (int c = 0; c < cols; c++)
            {
                for (int r = 0; r < 2; r++)
                {
                    res[r, c] = this[r, c];
                }
            }
            return res;
        }

    }
}
