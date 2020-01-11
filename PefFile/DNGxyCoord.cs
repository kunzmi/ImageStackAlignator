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
    //More or less a 1:1 copy of the DNGxyCoord class in Adobe DNG SDK
    //https://www.adobe.com/support/downloads/dng/dng_sdk.html

    public class DNGxyCoord
    {
        double x;
        double y;

        public DNGxyCoord()
        {
        }

        public DNGxyCoord(double xx, double yy)
        {
            x = xx;
            y = yy;
        }

        public DNGxyCoord(DNGxyCoord a)
        {
            x = a.x;
            y = a.y;
        }

        public double X
        {
            get { return x; }
            set { x = value; }
        }

        public double Y
        {
            get { return y; }
            set { y = value; }
        }

        public void Clear()
        {
            x = 0.0;
            y = 0.0;
        }

        public bool IsValid()
        {
            return x > 0.0 && y > 0.0;
        }

        public bool NotValid()
        {
            return !IsValid();
        }

        public DNGVector3 XYtoXYZ()
        {
            DNGxyCoord temp = new DNGxyCoord(this);

            // Restrict xy coord to someplace inside the range of real xy coordinates.
            // This prevents math from doing strange things when users specify
            // extreme temperature/tint coordinates.
            temp.x = DNGUtils.Pin(0.000001, temp.x, 0.999999);
            temp.y = DNGUtils.Pin(0.000001, temp.y, 0.999999);

            if (temp.x + temp.y > 0.999999)
            {
                double scale = 0.999999 / (temp.x + temp.y);
                temp.x *= scale;
                temp.y *= scale;
            }

            return new DNGVector3(temp.x / temp.y,
                                 1.0,
                                 (1.0 - temp.x - temp.y) / temp.y);
        }

        public static DNGxyCoord XYZtoXY(DNGVector3 coord)
        {
            double X = coord[0];
            double Y = coord[1];
            double Z = coord[2];

            double total = X + Y + Z;

            if (total > 0.0)
            {
                return new DNGxyCoord(X / total, Y / total);
            }

            return DNGxyCoord.D50;
        }

        public override bool Equals(object obj)
        {
            if (!(obj is DNGxyCoord))
            {
                return false;
            }

            DNGxyCoord t = (DNGxyCoord)obj;
            return this.x == t.x && this.y == t.y;
        }

        public override int GetHashCode()
        {
            return x.GetHashCode() | y.GetHashCode();
        }

        public static bool operator ==(DNGxyCoord a, DNGxyCoord b)
        {
            return a.x == b.x && a.y == b.y;
        }

        public static bool operator !=(DNGxyCoord a, DNGxyCoord b)
        {
            return a.x != b.x && a.y != b.y;
        }

        public static DNGxyCoord operator +(DNGxyCoord a, DNGxyCoord b)
        {
            return new DNGxyCoord(a.x + b.x, a.y + b.y);
        }

        public static DNGxyCoord operator -(DNGxyCoord a, DNGxyCoord b)
        {
            return new DNGxyCoord(a.x - b.x, a.y - b.y);
        }

        public static DNGxyCoord operator *(double scale, DNGxyCoord b)
        {
            return new DNGxyCoord(scale * b.x, scale * b.y);
        }

        public static DNGxyCoord operator *(DNGxyCoord a, double scale)
        {
            return new DNGxyCoord(a.x * scale, a.y * scale);
        }

        public static double operator *(DNGxyCoord a, DNGxyCoord b)
        {
            return a.x * b.x + a.y * a.y;
        }

        public static DNGxyCoord StdA
        {
            get 
            { 
                return new DNGxyCoord(0.4476, 0.4074); 
            }
        }
        public static DNGxyCoord D50
        {
            get 
            {
                return new DNGxyCoord(0.3457, 0.3585);
            }
        }

        public static DNGxyCoord D55
        {
            get 
            {
                return new DNGxyCoord(0.3324, 0.3474);
            }
        }

        public static DNGxyCoord D65
        {
            get 
            {
                return new DNGxyCoord(0.3127, 0.3290);
            }
        }

        public static DNGxyCoord D75
        {
            get 
            {
                return new DNGxyCoord(0.2990, 0.3149);
            }
        }

        public static DNGxyCoord PCStoXY
        {
            get { return D50; } //using ProPhoto as intermediate RGB space
        }
        public static DNGVector3 PCStoXYZ
        {
            get { return D50.XYtoXYZ(); }
        }
    }
}
