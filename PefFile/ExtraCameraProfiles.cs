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
using System.Xml.Serialization;

namespace PentaxPefFile
{
    public class ExtraCameraProfiles
    {
        #region Seralization
        /// <summary>
        /// Saves to an xml file
        /// </summary>
        /// <param name="FileName">File path of the new xml file</param>
        public void Save(string FileName)
        {
            using (var writer = new System.IO.StreamWriter(FileName))
            {
                var serializer = new XmlSerializer(this.GetType());
                serializer.Serialize(writer, this);
                writer.Flush();
            }
        }

        /// <summary>
        /// Load an object from an xml file
        /// </summary>
        /// <param name="FileName">Xml file name</param>
        /// <returns>The object created from the xml file</returns>
        public static ExtraCameraProfiles Load(string FileName)
        {
            using (var stream = System.IO.File.OpenRead(FileName))
            {
                var serializer = new XmlSerializer(typeof(ExtraCameraProfiles));
                return serializer.Deserialize(stream) as ExtraCameraProfiles;
            }
        }
        #endregion

        List<ExtraCameraProfile> extraCameraProfiles;

        public List<ExtraCameraProfile> Profiles
        {
            get { return extraCameraProfiles; }
            set { extraCameraProfiles = value; }
        }

        public ExtraCameraProfile GetProfile(string aMake, string aUniqueModel)
        {
            if (extraCameraProfiles == null)
                return null;

            for (int i = 0; i < extraCameraProfiles.Count; i++)
            {
                if (extraCameraProfiles[i].Make == aMake)
                {
                    if (extraCameraProfiles[i].UniqueModel == aUniqueModel)
                    {
                        return extraCameraProfiles[i];
                    }
                }
            }
            return null;
        }
    }

    public class ExtraCameraProfile
    {
        string make;
        string uniqueModel;

        double[] colorMatrix1;
        double[] colorMatrix2;
        IFDDNGCalibrationIlluminant.Illuminant illuminant1;
        IFDDNGCalibrationIlluminant.Illuminant illuminant2;

        NoiseModel noiseModel;
        CropInfo cropInfo;

        public string Make
        {
            get { return make; }
            set { make = value; }
        }
        public string UniqueModel
        {
            get { return uniqueModel; }
            set { uniqueModel = value; }
        }
        public double[] ColorMatrix1
        {
            get { return colorMatrix1; }
            set { colorMatrix1 = value; }
        }
        public double[] ColorMatrix2
        {
            get { return colorMatrix2; }
            set { colorMatrix2 = value; }
        }
        public IFDDNGCalibrationIlluminant.Illuminant Illuminant1
        {
            get { return illuminant1; }
            set { illuminant1 = value; }
        }
        public IFDDNGCalibrationIlluminant.Illuminant Illuminant2
        {
            get { return illuminant2; }
            set { illuminant2 = value; }
        }
        public NoiseModel NoiseModel
        {
            get { return noiseModel; }
            set { noiseModel = value; }
        }
        public CropInfo CropInfo
        {
            get { return cropInfo; }
            set { cropInfo = value; }
        }
    }

    public class NoiseModel
    {
        int[] iso;
        double[] alpha;
        double[] beta;

        public int[] Iso
        {
            get { return iso; }
            set { iso = value; }
        }
        public double[] Alpha
        {
            get { return alpha; }
            set { alpha = value; }
        }
        public double[] Beta
        {
            get { return beta; }
            set { beta = value; }
        }

        public (double, double) GetValue(int aIso)
        {
            if (iso.Length == 0)
            {
                return (0, 0);
            }

            int index = -1;
            for (int i = iso.Length - 1; i >= 0; i--)
            {
                if (iso[i] <= aIso)
                {
                    index = i;
                    break;
                }
            }

            //found at last index
            if (index == iso.Length - 1)
            {
                return (alpha[iso.Length - 1], beta[iso.Length - 1]);
            }
            //or asked ISO is smaller than our model
            if (index < 0)
            { 
                return (alpha[0], beta[0]);
            }

            //linear interpolate between two values
            double weight = ((double)aIso - (double)iso[index + 1]) / ((double)iso[index] - (double)iso[index + 1]);
            double a = weight * alpha[index] + (1.0 - weight) * alpha[index + 1];
            double b = weight * beta[index] + (1.0 - weight) * beta[index + 1];

            return (a, b);
        }
    }

    public class CropInfo
    {
        int top;
        int left;
        int width;
        int height;

        public int Top
        {
            get { return top; }
            set { top = value; }
        }
        public int Left
        {
            get { return left; }
            set { left = value; }
        }
        public int Width
        {
            get { return width; }
            set { width = value; }
        }
        public int Height
        {
            get { return height; }
            set { height = value; }
        }
    }
}
