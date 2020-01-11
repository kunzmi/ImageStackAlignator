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
using ManagedCuda.VectorTypes;

namespace PEFStudioDX
{
    public class PreAlignmentStore
    {
        float2[] _shifts;
        float[] _rotations;
        int _referenceIndex = -1;

        
        
        public PreAlignmentStore(double4[] initialShifts)
        {
            int imageCount = initialShifts.Length;
            _shifts = new float2[imageCount];
            _rotations = new float[imageCount];

            double4[] basicShifts = new double4[initialShifts.Length];

            for (int i = 1; i < imageCount; i++)
            {
                double4 shiftToI = initialShifts[i];
                double4 shiftToIMinus1 = initialShifts[i - 1];

                basicShifts[i - 1] = new double4(shiftToI.x - shiftToIMinus1.x, shiftToI.y - shiftToIMinus1.y, shiftToI.z - shiftToIMinus1.z, 0);
            }
            int minIndex = -1;

            //Reduce total shift:
            double2[] shiftLengthMin = new double2[imageCount - 1];

            for (int i = 0; i < imageCount - 1; i++)
            {
                double2 a = new double2();
                for (int j = 0; j < i; j++)
                {
                    a.x -= basicShifts[j].x;
                    a.y -= basicShifts[j].y;
                }

                double2 b = new double2();
                for (int j = i; j < imageCount - 1; j++)
                {
                    b.x += basicShifts[j].x;
                    b.y += basicShifts[j].y;
                }

                shiftLengthMin[i].x = a.x + b.x;
                shiftLengthMin[i].y = a.y + b.y;
            }

            minIndex = -1;
            double minShift = double.MaxValue;
            for (int i = 0; i < imageCount - 1; i++)
            {
                double2 a = shiftLengthMin[i];
                double d = System.Math.Sqrt(a.x * a.x + a.y * a.y);
                if (d < minShift)
                {
                    minShift = d;
                    minIndex = i;
                }
            }

            for (int i = 0; i < imageCount; i++)
            {
                double4 a = new double4();
                for (int j = i; j < minIndex; j++)
                {
                    a.x -= basicShifts[j].x;
                    a.y -= basicShifts[j].y;
                    a.z -= basicShifts[j].z;
                }

                for (int j = minIndex; j < i; j++)
                {
                    a.x += basicShifts[j].x;
                    a.y += basicShifts[j].y;
                    a.z += basicShifts[j].z;
                }
                float2 shift = new float2((float)a.x, (float)a.y);
                _shifts[i] = shift;
                _rotations[i] = (float)a.z;
            }
            _referenceIndex = minIndex;
        }

        public int ReferenceIndex
        {
            get { return _referenceIndex; }
            set { _referenceIndex = value; }
        }
        

        public float2 GetShift(int from, int to)
        {
            return _shifts[to] - _shifts[from];
        }

        public float2 GetShift(int to)
        {
            if (_referenceIndex < 0)
                return new float2(0, 0);
            return GetShift(_referenceIndex, to);
        }

        public float GetRotation(int from, int to)
        {
            float angle = _rotations[to] - _rotations[from];
            angle = angle / 180.0f * (float)System.Math.PI;
            return angle;
        }

        public float GetRotation(int to)
        {
            if (_referenceIndex < 0)
                return 0.0f;
            return GetRotation(_referenceIndex, to);
        }

        public void Reset()
        {
            for (int i = 0; i < _rotations.Length; i++)
            {
                _rotations[i] = 0;
            }
            for (int i = 0; i < _shifts.Length; i++)
            {
                _shifts[i] = new float2(0,0);
            }
        }
    }
}
