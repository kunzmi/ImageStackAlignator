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
    public class PatchTrackingLevel
    {
        int _resizeLevel;
        int _tileSize;
        int _maxShift;

        public PatchTrackingLevel()
        {
            _resizeLevel = 1;
            _tileSize = 32;
            _maxShift = 2;
        }
        public PatchTrackingLevel(int aResizeLevel, int aTileSize, int aMaxShift)
        {
            _resizeLevel = aResizeLevel;
            _tileSize = aTileSize;
            _maxShift = aMaxShift;
        }

        public int ResizeLevel
        {
            get { return _resizeLevel; }
            set { _resizeLevel = value; }
        }

        public int TileSize
        {
            get { return _tileSize; }
            set { _tileSize = value; }
        }

        public int MaxShift
        {
            get { return _maxShift; }
            set { _maxShift = value; }
        }
    }
}
