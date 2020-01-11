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
    //More or less a 1:1 copy of the DNGColorSpace class in Adobe DNG SDK
    //https://www.adobe.com/support/downloads/dng/dng_sdk.html

    public class DNGColorSpace
    {
        DNGMatrix mToPCS;
        DNGMatrix mFromPCS;

        //XYZ with D50 to sRGB with D65
        public static DNGColorSpace sRGB50 = new DNGColorSpace(new DNGMatrix3x3(0.4360747, 0.3850649, 0.1430804,
                                                                         0.2225045, 0.7168786, 0.0606169,
                                                                         0.0139322, 0.0971045, 0.7141733));
        //XYZ and ProPhoto D50
        public static DNGColorSpace ProPhoto = new DNGColorSpace(new DNGMatrix3x3(0.7976749, 0.1351917, 0.0313534,
                                                                           0.2880402, 0.7118741, 0.0000857,
                                                                           0.0000000, 0.0000000, 0.8252100));

        public DNGColorSpace(DNGMatrix3x3 toPCS)
        {
            // The matrix values are often rounded, so adjust to
            // get them to convert device white exactly to the PCS.
            DNGVector W1 = toPCS * new DNGVector3(1.0, 1.0, 1.0);
            DNGVector W2 = DNGxyCoord.PCStoXYZ;

            double s0 = W2[0] / W1[0];
            double s1 = W2[1] / W1[1];
            double s2 = W2[2] / W1[2];

            DNGMatrix3x3 S = new DNGMatrix3x3(s0,  0,  0,
				   		                    0, s1,  0,
				   		                    0,  0, s2);

            mToPCS = S * toPCS;

            // Find reverse matrix.
            mFromPCS = mToPCS.Invert();
        }

        public DNGMatrix ToPCS
        {
            get { return mToPCS; }
        }

        public DNGMatrix FromPCS
        {
            get { return mFromPCS; }
        }
    }

}
