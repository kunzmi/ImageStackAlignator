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
    //More or less a 1:1 copy of the DNGColorSpec class in Adobe DNG SDK
    //https://www.adobe.com/support/downloads/dng/dng_sdk.html
    /// Color transform taking into account white point and camera calibration and
    /// individual calibration from DNG negative.
    public class DNGColorSpec
    {
        /// \brief Compute a 3x3 matrix which maps colors from white point white1 to
        /// white point white2
        ///
        /// Uses linearized Bradford adaptation matrix to compute a mapping from 
        /// colors measured with one white point (white1) to another (white2).
        public static DNGMatrix3x3 MapWhiteMatrix(DNGxyCoord white1, DNGxyCoord white2)
        {
            // Use the linearized Bradford adaptation matrix.
            DNGMatrix3x3 Mb = new DNGMatrix3x3( 0.8951,  0.2664, -0.1614,
		 		                               -0.7502,  1.7135,  0.0367,
		  			                            0.0389, -0.0685,  1.0296);

            DNGVector w1 = Mb * white1.XYtoXYZ();
            DNGVector w2 = Mb * white2.XYtoXYZ();

            // Negative white coordinates are kind of meaningless.
            w1[0] = Math.Max(w1[0], 0.0);
            w1[1] = Math.Max(w1[1], 0.0);
            w1[2] = Math.Max(w1[2], 0.0);

            w2[0] = Math.Max(w2[0], 0.0);
            w2[1] = Math.Max(w2[1], 0.0);
            w2[2] = Math.Max(w2[2], 0.0);

            // Limit scaling to something reasonable.
            DNGMatrix3x3 A = new DNGMatrix3x3();

            A[0, 0] = DNGUtils.Pin(0.1, w1[0] > 0.0 ? w2[0] / w1[0] : 10.0, 10.0);
            A[1, 1] = DNGUtils.Pin(0.1, w1[1] > 0.0 ? w2[1] / w1[1] : 10.0, 10.0);
            A[2, 2] = DNGUtils.Pin(0.1, w1[2] > 0.0 ? w2[2] / w1[2] : 10.0, 10.0);

            DNGMatrix3x3 B = new DNGMatrix3x3(Mb.Invert() * A * Mb);
            return B;
        }
        
        uint fChannels;

        double fTemperature1;
        double fTemperature2;

        DNGMatrix fColorMatrix1 = new DNGMatrix();
        DNGMatrix fColorMatrix2 = new DNGMatrix();

        DNGMatrix fForwardMatrix1 = new DNGMatrix();
        DNGMatrix fForwardMatrix2 = new DNGMatrix();

        DNGMatrix fReductionMatrix1 = new DNGMatrix();
        DNGMatrix fReductionMatrix2 = new DNGMatrix();

        DNGMatrix fCameraCalibration1 = new DNGMatrix();
        DNGMatrix fCameraCalibration2 = new DNGMatrix();

        DNGMatrix fAnalogBalance = new DNGMatrix();

        DNGxyCoord fWhiteXY = new DNGxyCoord();

        DNGVector fCameraWhite = new DNGVector();

        DNGMatrix fCameraToPCS = new DNGMatrix();
        DNGMatrix fPCStoCamera = new DNGMatrix();

        private T GetTag<T>(ImageFileDirectory ifd0, ImageFileDirectory raw) where T : PentaxPefFile.ImageFileDirectoryEntry
        {
            T tag = ifd0.GetEntry<T>();
            if (tag == null)
            {
                //try raw tag
                tag = raw.GetEntry<T>();
            }
            return tag;
        }

        private DNGMatrix NormalizeForwardMatrix(DNGMatrix m)
        {
            if (m == null)
                return new DNGMatrix();

            if (m.NotEmpty())
            {
                DNGVector cameraOne = DNGVector.Identity(m.Cols);

                DNGVector xyz = m * cameraOne;
                
                m = DNGxyCoord.PCStoXYZ.AsDiagonal() *
                    (xyz.AsDiagonal().Invert()) * m;
            }
            return m;
        }

        private double ConvertIlluminantToTemperature(IFDDNGCalibrationIlluminant.Illuminant illuminant)
        {
            switch (illuminant)
            {
                case IFDDNGCalibrationIlluminant.Illuminant.Unknown:
                    return 0.0;
                case IFDDNGCalibrationIlluminant.Illuminant.Daylight:
                    return 5500.0;
                case IFDDNGCalibrationIlluminant.Illuminant.Fluorescent:
                    return (3800.0 + 4500.0) * 0.5;
                case IFDDNGCalibrationIlluminant.Illuminant.Tungsten:
                    return 2850.0;
                case IFDDNGCalibrationIlluminant.Illuminant.Flash:
                    return 5500.0;
                case IFDDNGCalibrationIlluminant.Illuminant.FineWeather:
                    return 5500.0;
                case IFDDNGCalibrationIlluminant.Illuminant.CloudyWeather:
                    return 6500.0;
                case IFDDNGCalibrationIlluminant.Illuminant.Shade:
                    return 7500.0;
                case IFDDNGCalibrationIlluminant.Illuminant.DaylightFluorescent:
                    return (5700.0 + 7100.0) * 0.5;
                case IFDDNGCalibrationIlluminant.Illuminant.DayWhiteFluorescent:
                    return (4600.0 + 5500.0) * 0.5;
                case IFDDNGCalibrationIlluminant.Illuminant.CoolWhiteFluorescent:
                    return (3800.0 + 4500.0) * 0.5;
                case IFDDNGCalibrationIlluminant.Illuminant.WhiteFluorescent:
                    return (3250.0 + 3800.0) * 0.5;
                case IFDDNGCalibrationIlluminant.Illuminant.WarmWhiteFluorescent:
                    return (2600.0 + 3250.0) * 0.5;
                case IFDDNGCalibrationIlluminant.Illuminant.StandardLightA:
                    return 2850.0;
                case IFDDNGCalibrationIlluminant.Illuminant.StandardLightB:
                    return 5500.0;
                case IFDDNGCalibrationIlluminant.Illuminant.StandardLightC:
                    return 6500.0;
                case IFDDNGCalibrationIlluminant.Illuminant.D55:
                    return 5500.0;
                case IFDDNGCalibrationIlluminant.Illuminant.D65:
                    return 6500.0;
                case IFDDNGCalibrationIlluminant.Illuminant.D75:
                    return 7500.0;
                case IFDDNGCalibrationIlluminant.Illuminant.D50:
                    return 5000.0;
                case IFDDNGCalibrationIlluminant.Illuminant.ISOStudioTungsten:
                    return 3200.0;
                case IFDDNGCalibrationIlluminant.Illuminant.OtherLightSource:
                    return 0.0;
                default:
                    return 0.0;
            }
        }

        public DNGColorSpec(double[] colorMatrix1, double[] colorMatrix2, 
            IFDDNGCalibrationIlluminant.Illuminant illuminant1, IFDDNGCalibrationIlluminant.Illuminant illuminant2, float[] whiteBalance)
        {
            fChannels = 3;
            fTemperature1 = ConvertIlluminantToTemperature(illuminant1);
            fTemperature2 = ConvertIlluminantToTemperature(illuminant2);

            
            if (colorMatrix1 == null)
                fColorMatrix1 = DNGMatrix.Identity(fChannels); //best choice if nothing is given...
            else
                fColorMatrix1 = new DNGMatrix3x3(colorMatrix1);

            if (colorMatrix2 == null)
                fColorMatrix2 = new DNGMatrix();
            else
                fColorMatrix2 = new DNGMatrix3x3(colorMatrix2);

            fForwardMatrix1 = new DNGMatrix();
            fForwardMatrix2 = new DNGMatrix();

            fReductionMatrix1 = new DNGMatrix();
            fReductionMatrix2 = new DNGMatrix();

            fCameraCalibration1 = DNGMatrix.Identity(fChannels);
            fCameraCalibration2 = DNGMatrix.Identity(fChannels);

            fAnalogBalance = DNGMatrix.Identity(fChannels);

            fForwardMatrix1 = NormalizeForwardMatrix(fForwardMatrix1);

            fColorMatrix1 = fAnalogBalance * fCameraCalibration1 * fColorMatrix1;

            if (fColorMatrix2.IsEmpty() ||
                fTemperature1 <= 0.0 ||
                fTemperature2 <= 0.0 ||
                fTemperature1 == fTemperature2)
            {

                fTemperature1 = 5000.0;
                fTemperature2 = 5000.0;

                fColorMatrix2 = fColorMatrix1;
                fForwardMatrix2 = fForwardMatrix1;
                fReductionMatrix2 = fReductionMatrix1;
                fCameraCalibration2 = fCameraCalibration1;
            }
            else
            {
                fForwardMatrix2 = NormalizeForwardMatrix(fForwardMatrix2);
                fColorMatrix2 = fAnalogBalance * fCameraCalibration2 * fColorMatrix2;

                // Swap values if temperatures are out of order.
                if (fTemperature1 > fTemperature2)
                {
                    double temp = fTemperature1;
                    fTemperature1 = fTemperature2;
                    fTemperature2 = temp;

                    DNGMatrix T = fColorMatrix1;
                    fColorMatrix1 = fColorMatrix2;
                    fColorMatrix2 = T;

                    T = fForwardMatrix1;
                    fForwardMatrix1 = fForwardMatrix2;
                    fForwardMatrix2 = T;

                    T = fReductionMatrix1;
                    fReductionMatrix1 = fReductionMatrix2;
                    fReductionMatrix2 = T;

                    T = fCameraCalibration1;
                    fCameraCalibration1 = fCameraCalibration2;
                    fCameraCalibration2 = T;
                }
            }

            DNGxyCoord white;            
            DNGVector vec = new DNGVector((uint)whiteBalance.Length);

            for (uint c = 0; c < whiteBalance.Length; c++)
            {
                //white point is given as a multiplicatice factor
                //actual white point is hence 1/value
                vec[c] = 1.0f / whiteBalance[c];
            }

            double unify = 1.0 / vec.MaxEntry();
            vec = unify * vec;

            white = NeutralToXY(vec);
            
            WhiteXY = white;
        }

        public DNGColorSpec(uint aChannels, ImageFileDirectory ifd0, ImageFileDirectory raw)
        {
            fChannels = aChannels;
            if (GetTag<IFDDNGCalibrationIlluminant1>(ifd0, raw) != null)
                fTemperature1 = ConvertIlluminantToTemperature(GetTag<IFDDNGCalibrationIlluminant1>(ifd0, raw).Value);
            else
                fTemperature1 = 0.0;
            if (GetTag<IFDDNGCalibrationIlluminant2>(ifd0, raw) != null)
                fTemperature2 = ConvertIlluminantToTemperature(GetTag<IFDDNGCalibrationIlluminant2>(ifd0, raw).Value);
            else
                fTemperature2 = 0.0;
            
            fColorMatrix1 = GetTag<IFDDNGColorMatrix1>(ifd0, raw)?.Matrix;
            if (fColorMatrix1 == null) 
                fColorMatrix1 = DNGMatrix.Identity(fChannels); //best choice if nothing is given...
            fColorMatrix2 = GetTag<IFDDNGColorMatrix2>(ifd0, raw)?.Matrix;
            if (fColorMatrix2 == null)
                fColorMatrix2 = new DNGMatrix();

            fForwardMatrix1 = GetTag<IFDDNGForwardMatrix1>(ifd0, raw)?.Matrix;
            if (fForwardMatrix1 == null)
                fForwardMatrix1 = new DNGMatrix();
            fForwardMatrix2 = GetTag<IFDDNGForwardMatrix2>(ifd0, raw)?.Matrix;
            if (fForwardMatrix2 == null)
                fForwardMatrix2 = new DNGMatrix();

            fReductionMatrix1 = GetTag<IFDDNGReductionMatrix1>(ifd0, raw)?.Matrix;
            if (fReductionMatrix1 == null)
                fReductionMatrix1 = new DNGMatrix();
            fReductionMatrix2 = GetTag<IFDDNGReductionMatrix2>(ifd0, raw)?.Matrix;
            if (fReductionMatrix2 == null)
                fReductionMatrix2 = new DNGMatrix();

            fCameraCalibration1 = GetTag<IFDDNGCameraCalibration1>(ifd0, raw)?.Matrix;
            fCameraCalibration2 = GetTag<IFDDNGCameraCalibration1>(ifd0, raw)?.Matrix;
            if (fCameraCalibration1 == null)
                fCameraCalibration1 = DNGMatrix.Identity(fChannels);
            if (fCameraCalibration2 == null)
                fCameraCalibration2 = DNGMatrix.Identity(fChannels);

            fAnalogBalance = GetTag<IFDDNGAnalogBalance>(ifd0, raw)?.Vector.AsDiagonal();
            if (fAnalogBalance == null)
                fAnalogBalance = DNGMatrix.Identity(fChannels);

            fForwardMatrix1 = NormalizeForwardMatrix(fForwardMatrix1);

            fColorMatrix1 = fAnalogBalance * fCameraCalibration1 * fColorMatrix1;

            if (fColorMatrix2.IsEmpty() ||
                fTemperature1 <= 0.0 ||
                fTemperature2 <= 0.0 ||
                fTemperature1 == fTemperature2)
            {

                fTemperature1 = 5000.0;
                fTemperature2 = 5000.0;

                fColorMatrix2 = fColorMatrix1;
                fForwardMatrix2 = fForwardMatrix1;
                fReductionMatrix2 = fReductionMatrix1;
                fCameraCalibration2 = fCameraCalibration1;
            }
            else
            {
                fForwardMatrix2 = NormalizeForwardMatrix(fForwardMatrix2);
                fColorMatrix2 = fAnalogBalance * fCameraCalibration2 * fColorMatrix2;
                
                // Swap values if temperatures are out of order.
                if (fTemperature1 > fTemperature2)
                {
                    double temp = fTemperature1;
                    fTemperature1 = fTemperature2;
                    fTemperature2 = temp;

                    DNGMatrix T = fColorMatrix1;
                    fColorMatrix1 = fColorMatrix2;
                    fColorMatrix2 = T;

                    T = fForwardMatrix1;
                    fForwardMatrix1 = fForwardMatrix2;
                    fForwardMatrix2 = T;

                    T = fReductionMatrix1;
                    fReductionMatrix1 = fReductionMatrix2;
                    fReductionMatrix2 = T;

                    T = fCameraCalibration1;
                    fCameraCalibration1 = fCameraCalibration2;
                    fCameraCalibration2 = T;
                }
            }
            
            IFDDNGAsShotNeutral neutral = GetTag<IFDDNGAsShotNeutral>(ifd0, raw);
            IFDDNGAsShotWhiteXY asShot = GetTag<IFDDNGAsShotWhiteXY>(ifd0, raw);
            DNGxyCoord white;

            if (asShot == null)
            {
                if (neutral == null)
                {
                    throw new ArgumentException("The DNG spec says that one of the As Shot White balance tags must be present.");
                }

                DNGVector vec = new DNGVector((uint)neutral.Value.Length);
                
                for (uint c = 0; c < neutral.Value.Length; c++)
                {
                    vec[c] = neutral.Value[c].Value;
                }

                double unify = 1.0 / vec.MaxEntry();
                vec = unify * vec;

                white = NeutralToXY(vec);
            }
            else
            {
                double x = asShot.Value[0].Value;
                double y = asShot.Value[1].Value;
                white = new DNGxyCoord(x, y);
            }
            WhiteXY = white;
        }

        public uint Channels
        {
            get { return fChannels; }
        }

        public DNGxyCoord WhiteXY
        {
            get 
            {
                return fWhiteXY; 
            }
            set 
            {
                fWhiteXY = value;

                // Deal with monochrome cameras.
                if (fChannels == 1)
                {
                    fCameraWhite.SetIdentity(1);
                    fCameraToPCS = DNGxyCoord.PCStoXYZ.AsColumn();
                    return;
                }

                // Interpolate all matric values for this white point.
                DNGMatrix colorMatrix = new DNGMatrix();
                DNGMatrix forwardMatrix = new DNGMatrix();
                DNGMatrix reductionMatrix = new DNGMatrix();
                DNGMatrix cameraCalibration = new DNGMatrix();

                colorMatrix = FindXYZtoCamera(fWhiteXY,
                                               ref forwardMatrix,
                                               ref reductionMatrix,
                                               ref cameraCalibration);

                // Find the camera white values.
                fCameraWhite = colorMatrix * fWhiteXY.XYtoXYZ();

                double whiteScale = 1.0 / fCameraWhite.MaxEntry();

                for (uint j = 0; j < fChannels; j++)
                {
                    // We don't support non-positive values for camera neutral values.
                    fCameraWhite[j] = DNGUtils.Pin(0.001,
                                                   whiteScale * fCameraWhite[j],
                                                   1.0);
                }

                // Find PCS to Camera transform. Scale matrix so PCS white can just be
                // reached when the first camera channel saturates
                fPCStoCamera = colorMatrix * MapWhiteMatrix(DNGxyCoord.PCStoXY, fWhiteXY);
                double scale = (fPCStoCamera * DNGxyCoord.PCStoXYZ).MaxEntry();
                fPCStoCamera = (1.0 / scale) * fPCStoCamera;

                // If we have a forward matrix, then just use that.
                if (forwardMatrix.NotEmpty())
                {
                    DNGMatrix individualToReference = (fAnalogBalance * cameraCalibration).Invert();

                    DNGVector refCameraWhite = individualToReference * fCameraWhite;

                    fCameraToPCS = forwardMatrix *
                                   refCameraWhite.AsDiagonal().Invert() *
                                   individualToReference;
                }
                // Else we need to use the adapt in XYZ method.
                else
                {
                    // Invert this PCS to camera matrix.  Note that if there are more than three
                    // camera channels, this inversion is non-unique.
                    fCameraToPCS = fPCStoCamera.Invert(reductionMatrix);
                }
            }
        }

        public DNGVector CameraWhite
        {
            get { return fCameraWhite; }
        }

        public DNGMatrix CameraToPCS
        {
            get { return fCameraToPCS; }
        }

        public DNGMatrix PCStoCamera
        {
            get { return fPCStoCamera; }
        }

        /// Return the XY value to use for SetWhiteXY for a given camera color
        /// space coordinate as the white point.
        /// \param neutral A camera color space value to use for white point.
        /// Components range from 0.0 to 1.0 and should be normalized such that
        /// the largest value is 1.0 .
        /// \retval White point in XY space that makes neutral map to this
        /// XY value as closely as possible.
        public DNGxyCoord NeutralToXY(DNGVector neutral)
        {
            const uint kMaxPasses = 30;

            if (fChannels == 1)
            {
                return DNGxyCoord.PCStoXY;
            }

            DNGxyCoord last = DNGxyCoord.D50;

            for (uint pass = 0; pass < kMaxPasses; pass++)
            {
                DNGMatrix nullMat = null;
                DNGMatrix xyzToCamera = FindXYZtoCamera(last, ref nullMat, ref nullMat, ref nullMat);

                DNGMatrix inv = xyzToCamera.Invert();
                DNGVector vec = inv * neutral;
                DNGVector3 vec3 = new DNGVector3(vec);

                DNGxyCoord next = DNGxyCoord.XYZtoXY(new DNGVector3(xyzToCamera.Invert() * neutral));

                if (Math.Abs(next.X - last.X) +
                    Math.Abs(next.Y - last.Y) < 0.0000001)
                {
                    return next;
                }

                // If we reach the limit without converging, we are most likely
                // in a two value oscillation.  So take the average of the last
                // two estimates and give up.
                if (pass == kMaxPasses - 1)
                {
                    next.X = (last.X + next.X) * 0.5;
                    next.Y = (last.Y + next.Y) * 0.5;
                }
                last = next;
            }
            return last;
        }

        private DNGMatrix FindXYZtoCamera(DNGxyCoord white, ref DNGMatrix forwardMatrix,
            ref DNGMatrix reductionMatrix, ref DNGMatrix cameraCalibration)
        {

            // Convert to temperature/offset space.
            DNGTemperature td = new DNGTemperature(white);

            // Find fraction to weight the first calibration.
            double g;

            if (td.Temperature <= fTemperature1)
                g = 1.0;

            else if (td.Temperature >= fTemperature2)
                g = 0.0;

            else
            {
                double invT = 1.0 / td.Temperature;

                g = (invT - (1.0 / fTemperature2)) /
                    ((1.0 / fTemperature1) - (1.0 / fTemperature2));
            }

            // Interpolate the color matrix.

            DNGMatrix colorMatrix;

            if (g >= 1.0)
                colorMatrix = fColorMatrix1;

            else if (g <= 0.0)
                colorMatrix = fColorMatrix2;

            else
                colorMatrix = (g) * fColorMatrix1 +
                              (1.0 - g) * fColorMatrix2;

            // Interpolate forward matrix, if any.
            if (forwardMatrix != null)
            {
                bool has1 = fForwardMatrix1.NotEmpty();
                bool has2 = fForwardMatrix2.NotEmpty();

                if (has1 && has2)
                {
                    if (g >= 1.0)
                        forwardMatrix = fForwardMatrix1;

                    else if (g <= 0.0)
                        forwardMatrix = fForwardMatrix2;

                    else
                        forwardMatrix = (g) * fForwardMatrix1 +
                                         (1.0 - g) * fForwardMatrix2;
                }
                else if (has1)
                {
                    forwardMatrix = fForwardMatrix1;
                }
                else if (has2)
                {
                    forwardMatrix = fForwardMatrix2;
                }
                else
                {
                    forwardMatrix.Clear();
                }
            }

            // Interpolate reduction matrix, if any.
            if (reductionMatrix != null)
            {
                bool has1 = fReductionMatrix1.NotEmpty();
                bool has2 = fReductionMatrix2.NotEmpty();

                if (has1 && has2)
                {
                    if (g >= 1.0)
                        reductionMatrix = fReductionMatrix1;
                    else if (g <= 0.0)
                        reductionMatrix = fReductionMatrix2;
                    else
                        reductionMatrix = (g) * fReductionMatrix1 +
                                           (1.0 - g) * fReductionMatrix2;
                }
                else if (has1)
                {
                    reductionMatrix = fReductionMatrix1;
                }
                else if (has2)
                {
                    reductionMatrix = fReductionMatrix2;
                }
                else
                {
                    reductionMatrix.Clear();
                }
            }

            // Interpolate camera calibration matrix.
            if (cameraCalibration != null)
            {
                if (g >= 1.0)
                    cameraCalibration = fCameraCalibration1;
                else if (g <= 0.0)
                    cameraCalibration = fCameraCalibration2;
                else
                    cameraCalibration = (g) * fCameraCalibration1 +
                                         (1.0 - g) * fCameraCalibration2;
            }

            // Return the interpolated color matrix.
            return colorMatrix;
        }
    }
}
