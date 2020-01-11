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
    /// <summary>
    /// Same as DNG SDK, this is more human readable than TIFF spec.
    /// https://www.adobe.com/support/downloads/dng/dng_sdk.html
    /// </summary>
    public class DNGOrientation
    {
        public enum Orientation
        {
            Normal = 0,
            Rotate90CW = 1,
            Rotate180 = 2,
            Rotate90CCW = 3,
            Mirror = 4,
            Mirror90CW = 5,
            Mirror180 = 6,
            Mirror90CCW = 7,
            Unknown = 8
        }

        private Orientation _value;

        public DNGOrientation(Orientation aValue)
        {
            _value = aValue;
        }

        public DNGOrientation(ushort tiffOrientation)
        {
            switch (tiffOrientation)
            {
                case 1:
                    _value = Orientation.Normal;
                    break;
                case 2:
                    _value = Orientation.Mirror;
                    break;
                case 3:
                    _value = Orientation.Rotate180;
                    break;
                case 4:
                    _value = Orientation.Mirror180;
                    break;
                case 5:
                    _value = Orientation.Mirror90CCW;
                    break;
                case 6:
                    _value = Orientation.Rotate90CW;
                    break;
                case 7:
                    _value = Orientation.Mirror90CW;
                    break;
                case 8:
                    _value = Orientation.Rotate90CCW;
                    break;
                case 9:
                    _value = Orientation.Unknown;
                    break;
                default:
                    _value = Orientation.Normal;
                    break;
            }
        }

        public Orientation Value
        {
            get { return _value; }
        }
    }
}
