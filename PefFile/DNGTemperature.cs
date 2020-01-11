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
    //More or less a 1:1 copy of the dng_temperature class in Adobe DNG SDK
    //https://www.adobe.com/support/downloads/dng/dng_sdk.html

    public class DNGTemperature
    {
        const double kTintScale = -3000.0;
        struct ruvt
        {
            public double r;
            public double u;
            public double v;
            public double t;

            public ruvt(double aR, double aU, double aV, double aT)
            {
                r = aR;
                u = aU;
                v = aV;
                t = aT;
            }
        };
        
        static ruvt[] kTempTable = new ruvt[]
            {
            new ruvt(   0, 0.18006, 0.26352, -0.24341 ),
            new ruvt(  10, 0.18066, 0.26589, -0.25479 ),
            new ruvt(  20, 0.18133, 0.26846, -0.26876 ),
            new ruvt(  30, 0.18208, 0.27119, -0.28539 ),
            new ruvt(  40, 0.18293, 0.27407, -0.30470 ),
            new ruvt(  50, 0.18388, 0.27709, -0.32675 ),
            new ruvt(  60, 0.18494, 0.28021, -0.35156 ),
            new ruvt(  70, 0.18611, 0.28342, -0.37915 ),
            new ruvt(  80, 0.18740, 0.28668, -0.40955 ),
            new ruvt(  90, 0.18880, 0.28997, -0.44278 ),
            new ruvt( 100, 0.19032, 0.29326, -0.47888 ),
            new ruvt( 125, 0.19462, 0.30141, -0.58204 ),
            new ruvt( 150, 0.19962, 0.30921, -0.70471 ),
            new ruvt( 175, 0.20525, 0.31647, -0.84901 ),
            new ruvt( 200, 0.21142, 0.32312, -1.0182 ),
            new ruvt( 225, 0.21807, 0.32909, -1.2168 ),
            new ruvt( 250, 0.22511, 0.33439, -1.4512 ),
            new ruvt( 275, 0.23247, 0.33904, -1.7298 ),
            new ruvt( 300, 0.24010, 0.34308, -2.0637 ),
            new ruvt( 325, 0.24702, 0.34655, -2.4681 ),
            new ruvt( 350, 0.25591, 0.34951, -2.9641 ),
            new ruvt( 375, 0.26400, 0.35200, -3.5814 ),
            new ruvt( 400, 0.27218, 0.35407, -4.3633 ),
            new ruvt( 425, 0.28039, 0.35577, -5.3762 ),
            new ruvt( 450, 0.28863, 0.35714, -6.7262 ),
            new ruvt( 475, 0.29685, 0.35823, -8.5955 ),
            new ruvt( 500, 0.30505, 0.35907, -11.324 ),
            new ruvt( 525, 0.31320, 0.35968, -15.628 ),
            new ruvt( 550, 0.32129, 0.36011, -23.325 ),
            new ruvt( 575, 0.32931, 0.36038, -40.770 ),
            new ruvt( 600, 0.33724, 0.36051, -116.45 )
            };

        double _temperature;
        double _tint;

        public DNGTemperature()
        { 
        }

        public DNGTemperature(double aTemperature, double aTint)
        {
            _temperature = aTemperature;
            _tint = aTint;
        }

        public DNGTemperature(DNGxyCoord xy)
        {
            xyCoord = xy;
        }

        public double Temperature
        {
            get { return _temperature; }
            set { _temperature = value; }
        }

        public double Tint
        {
            get { return _tint; }
            set { _tint = value; }
        }

        public DNGxyCoord xyCoord
        {
            get 
            {
                DNGxyCoord result = new DNGxyCoord();

                // Find inverse temperature to use as index.
                double r = 1.0E6 / _temperature;

                // Convert tint to offset is uv space.
                double offset = _tint * (1.0 / kTintScale);

                // Search for line pair containing coordinate.
                for (uint index = 0; index <= 29; index++)
                {
                    if (r < kTempTable[index + 1].r || index == 29)
                    {
                        // Find relative weight of first line.
                        double f = (kTempTable[index + 1].r - r) /
                                   (kTempTable[index + 1].r - kTempTable[index].r);

                        // Interpolate the black body coordinates.
                        double u = kTempTable[index].u * f +
                                   kTempTable[index + 1].u * (1.0 - f);

                        double v = kTempTable[index].v * f +
                                   kTempTable[index + 1].v * (1.0 - f);

                        // Find vectors along slope for each line.
                        double uu1 = 1.0;
                        double vv1 = kTempTable[index].t;

                        double uu2 = 1.0;
                        double vv2 = kTempTable[index + 1].t;

                        double len1 = Math.Sqrt(1.0 + vv1 * vv1);
                        double len2 = Math.Sqrt(1.0 + vv2 * vv2);

                        uu1 /= len1;
                        vv1 /= len1;

                        uu2 /= len2;
                        vv2 /= len2;

                        // Find vector from black body point.
                        double uu3 = uu1 * f + uu2 * (1.0 - f);
                        double vv3 = vv1 * f + vv2 * (1.0 - f);

                        double len3 = Math.Sqrt(uu3 * uu3 + vv3 * vv3);

                        uu3 /= len3;
                        vv3 /= len3;

                        // Adjust coordinate along this vector.
                        u += uu3 * offset;
                        v += vv3 * offset;

                        // Convert to xy coordinates.
                        result.X = 1.5 * u / (u - 4.0 * v + 2.0);
                        result.Y = v / (u - 4.0 * v + 2.0);

                        break;
                    }
                }

                return result;
            }

            set 
            {
                // Convert to uv space.
                double u = 2.0 * value.X / (1.5 - value.X + 6.0 * value.Y);
                double v = 3.0 * value.Y / (1.5 - value.X + 6.0 * value.Y);

                // Search for line pair coordinate is between.
                double last_dt = 0.0;
                double last_dv = 0.0;
                double last_du = 0.0;

                for (uint index = 1; index <= 30; index++)
                {
                    // Convert slope to delta-u and delta-v, with length 1.
                    double du = 1.0;
                    double dv = kTempTable[index].t;

                    double len = Math.Sqrt(1.0 + dv * dv);
                    du /= len;
                    dv /= len;

                    // Find delta from black body point to test coordinate.
                    double uu = u - kTempTable[index].u;
                    double vv = v - kTempTable[index].v;

                    // Find distance above or below line.
                    double dt = -uu * dv + vv * du;

                    // If below line, we have found line pair.
                    if (dt <= 0.0 || index == 30)
                    {
                        // Find fractional weight of two lines.
                        if (dt > 0.0)
                            dt = 0.0;

                        dt = -dt;

                        double f;

                        if (index == 1)
                        {
                            f = 0.0;
                        }
                        else
                        {
                            f = dt / (last_dt + dt);
                        }

                        // Interpolate the temperature.
                        _temperature = 1.0E6 / (kTempTable[index - 1].r * f +
                                                kTempTable[index].r * (1.0 - f));

                        // Find delta from black body point to test coordinate.
                        uu = u - (kTempTable[index - 1].u * f +
                                  kTempTable[index].u * (1.0 - f));

                        vv = v - (kTempTable[index - 1].v * f +
                                  kTempTable[index].v * (1.0 - f));

                        // Interpolate vectors along slope.
                        du = du * (1.0 - f) + last_du * f;
                        dv = dv * (1.0 - f) + last_dv * f;

                        len = Math.Sqrt(du * du + dv * dv);

                        du /= len;
                        dv /= len;

                        // Find distance along slope.
                        _tint = (uu * du + vv * dv) * kTintScale;
                        break;
                    }

                    // Try next line pair.
                    last_dt = dt;
                    last_du = du;
                    last_dv = dv;
                }
            }
        }
    }
}
