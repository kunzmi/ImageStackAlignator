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
using System.ComponentModel;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Windows;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using PentaxPefFile;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using ManagedCuda.NPP;
using ManagedCuda.NPP.NPPsExtensions;

namespace PEFStudioDX
{
    public class ImageStackAlignatorController : DependencyObject, INotifyPropertyChanged
    {
        bool _isTestMode = false;

        //for test mode
        float[] preShiftsX = new float[] { 0, -300, -500, -400, -200 };
        float[] preShiftsY = new float[] { 0, 1, 2, 3, 4 };
        float[] preRot = new float[] { 0.00f, 0.2f, 0.4f, 0.6f, 0.8f };
        float[] shiftX = new float[] { 0, -3, 5, -8, -11 };
        float[] shiftY = new float[] { 0, -2, 4, -6, -8 };

        public float[] defaultLUT = new float[256];

        public float[] LUTx = new float[] { 0f,
            0.003921569f,
            0.007843137f,
            0.011764706f,
            0.015686275f,
            0.019607843f,
            0.023529412f,
            0.02745098f,
            0.031372549f,
            0.035294118f,
            0.039215686f,
            0.043137255f,
            0.047058824f,
            0.050980392f,
            0.054901961f,
            0.058823529f,
            0.062745098f,
            0.066666667f,
            0.070588235f,
            0.074509804f,
            0.078431373f,
            0.082352941f,
            0.08627451f,
            0.090196078f,
            0.094117647f,
            0.098039216f,
            0.101960784f,
            0.105882353f,
            0.109803922f,
            0.11372549f,
            0.117647059f,
            0.121568627f,
            0.125490196f,
            0.129411765f,
            0.133333333f,
            0.137254902f,
            0.141176471f,
            0.145098039f,
            0.149019608f,
            0.152941176f,
            0.156862745f,
            0.160784314f,
            0.164705882f,
            0.168627451f,
            0.17254902f,
            0.176470588f,
            0.180392157f,
            0.184313725f,
            0.188235294f,
            0.192156863f,
            0.196078431f,
            0.2f,
            0.203921569f,
            0.207843137f,
            0.211764706f,
            0.215686275f,
            0.219607843f,
            0.223529412f,
            0.22745098f,
            0.231372549f,
            0.235294118f,
            0.239215686f,
            0.243137255f,
            0.247058824f,
            0.250980392f,
            0.254901961f,
            0.258823529f,
            0.262745098f,
            0.266666667f,
            0.270588235f,
            0.274509804f,
            0.278431373f,
            0.282352941f,
            0.28627451f,
            0.290196078f,
            0.294117647f,
            0.298039216f,
            0.301960784f,
            0.305882353f,
            0.309803922f,
            0.31372549f,
            0.317647059f,
            0.321568627f,
            0.325490196f,
            0.329411765f,
            0.333333333f,
            0.337254902f,
            0.341176471f,
            0.345098039f,
            0.349019608f,
            0.352941176f,
            0.356862745f,
            0.360784314f,
            0.364705882f,
            0.368627451f,
            0.37254902f,
            0.376470588f,
            0.380392157f,
            0.384313725f,
            0.388235294f,
            0.392156863f,
            0.396078431f,
            0.4f,
            0.403921569f,
            0.407843137f,
            0.411764706f,
            0.415686275f,
            0.419607843f,
            0.423529412f,
            0.42745098f,
            0.431372549f,
            0.435294118f,
            0.439215686f,
            0.443137255f,
            0.447058824f,
            0.450980392f,
            0.454901961f,
            0.458823529f,
            0.462745098f,
            0.466666667f,
            0.470588235f,
            0.474509804f,
            0.478431373f,
            0.482352941f,
            0.48627451f,
            0.490196078f,
            0.494117647f,
            0.498039216f,
            0.501960784f,
            0.505882353f,
            0.509803922f,
            0.51372549f,
            0.517647059f,
            0.521568627f,
            0.525490196f,
            0.529411765f,
            0.533333333f,
            0.537254902f,
            0.541176471f,
            0.545098039f,
            0.549019608f,
            0.552941176f,
            0.556862745f,
            0.560784314f,
            0.564705882f,
            0.568627451f,
            0.57254902f,
            0.576470588f,
            0.580392157f,
            0.584313725f,
            0.588235294f,
            0.592156863f,
            0.596078431f,
            0.6f,
            0.603921569f,
            0.607843137f,
            0.611764706f,
            0.615686275f,
            0.619607843f,
            0.623529412f,
            0.62745098f,
            0.631372549f,
            0.635294118f,
            0.639215686f,
            0.643137255f,
            0.647058824f,
            0.650980392f,
            0.654901961f,
            0.658823529f,
            0.662745098f,
            0.666666667f,
            0.670588235f,
            0.674509804f,
            0.678431373f,
            0.682352941f,
            0.68627451f,
            0.690196078f,
            0.694117647f,
            0.698039216f,
            0.701960784f,
            0.705882353f,
            0.709803922f,
            0.71372549f,
            0.717647059f,
            0.721568627f,
            0.725490196f,
            0.729411765f,
            0.733333333f,
            0.737254902f,
            0.741176471f,
            0.745098039f,
            0.749019608f,
            0.752941176f,
            0.756862745f,
            0.760784314f,
            0.764705882f,
            0.768627451f,
            0.77254902f,
            0.776470588f,
            0.780392157f,
            0.784313725f,
            0.788235294f,
            0.792156863f,
            0.796078431f,
            0.8f,
            0.803921569f,
            0.807843137f,
            0.811764706f,
            0.815686275f,
            0.819607843f,
            0.823529412f,
            0.82745098f,
            0.831372549f,
            0.835294118f,
            0.839215686f,
            0.843137255f,
            0.847058824f,
            0.850980392f,
            0.854901961f,
            0.858823529f,
            0.862745098f,
            0.866666667f,
            0.870588235f,
            0.874509804f,
            0.878431373f,
            0.882352941f,
            0.88627451f,
            0.890196078f,
            0.894117647f,
            0.898039216f,
            0.901960784f,
            0.905882353f,
            0.909803922f,
            0.91372549f,
            0.917647059f,
            0.921568627f,
            0.925490196f,
            0.929411765f,
            0.933333333f,
            0.937254902f,
            0.941176471f,
            0.945098039f,
            0.949019608f,
            0.952941176f,
            0.956862745f,
            0.960784314f,
            0.964705882f,
            0.968627451f,
            0.97254902f,
            0.976470588f,
            0.980392157f,
            0.984313725f,
            0.988235294f,
            0.992156863f,
            0.996078431f,
            1f
        };
        
        public enum WorkState
        {
            Init,
            PreAlign,
            PatchAlign,
            Accumulate,
            PostProcessing
        }

        [TypeConverter(typeof(EnumDescriptionTypeConverter))]
        public enum WhatToShow
        {
            [Description("Final image")]
            FinalImage,
            [Description("Weightings red")]
            WeightingsRed,
            [Description("Weightings green")]
            WeightingsGreen,
            [Description("Weightings blue")]
            WeightingsBlue,
            [Description("Certainty mask red")]
            CertaintyMaskRed,
            [Description("Certainty mask green")]
            CertaintyMaskGreen,
            [Description("Certainty mask blue")]
            CertaintyMaskBlue,
            [Description("Warped image")]
            WarpedImage
        }

        List<RawFile> _pefFiles;
        List<bool> _selected;
        List<float[]> _debayerdImagesBW;

        WorkState _workState;
        CudaContext _ctx;

        int _selectedItem = -1;
        NPPImage_8uC4 _imageToShow;
        NPPImage_16uC1 _rawImage;
        CudaDeviceVariable<ushort> _rawImageNoPitch;
        NPPImage_32fC1 _rawImageFloat;
        NPPImage_32fC3 _decodedImage;
        NPPImage_32fC3 _finalImage;
        NPPImage_32fC3 _totalWeight;
        NPPImage_32fC3 _structureTensor;
        NPPImage_32fC3 _debayerRefHalfRes;
        NPPImage_32fC3 _debayerTrackHalfRes;
        NPPImage_32fC4 _uncertaintyMask;
        NPPImage_32fC4 _uncertaintyMaskEroded;
        NPPImage_32fC4 _structureTensor4;
        CudaDeviceVariable<float> _LUTx;
        CudaDeviceVariable<float> _LUTy;

        DeBayerGreenKernel deBayerGreenKernel;
        DeBayerRedBlueKernel deBayerRedBlueKernel;
        DeBayersSubSampleKernel debayerSubSample;
        GammasRGBKernel gammaCorrectionKernel;
        ApplyWeightingKernel applyWeightingKernel;
        upsampleShiftsKernel upsampleShifts;
        ComputeDerivatives2Kernel computeDerivatives2Kernel;
        computeStructureTensorKernel computeStructureTensor;
        computeKernelParamKernel computeKernelParam;
        AccumulateImagesKernel accumulateImages;
        AccumulateImagesSuperResKernel accumulateImagesSuperRes;
        RobustnessModellKernel computeMask;

        CudaDeviceVariable<float> filterBuffer;

        PreAlignment _preAlignment = null;
        PreAlignmentStore _preAlignmentStore;
        ShiftCollection _shiftCollection;

        float3[] _kernelParameters;

        PentaxPefFile.DNGColorSpec _colorSpec;

        public ImageStackAlignatorController()
        {
            FileNameList = new ObservableCollection<string>();
            PatchTrackingLevels = new ObservableCollection<PatchTrackingLevel>();
            PatchTrackingLevels.Add(new PatchTrackingLevel());

            System.ComponentModel.DependencyPropertyDescriptor imageColorChanged = System.ComponentModel.DependencyPropertyDescriptor.FromProperty
                (ImageStackAlignatorController.ColorTemperatureProperty, typeof(ImageStackAlignatorController));
            imageColorChanged.AddValueChanged(this, new EventHandler(this.ImageColorChanged));

            imageColorChanged = System.ComponentModel.DependencyPropertyDescriptor.FromProperty
                (ImageStackAlignatorController.ColorTintProperty, typeof(ImageStackAlignatorController));
            imageColorChanged.AddValueChanged(this, new EventHandler(this.ImageColorChanged));

            imageColorChanged = System.ComponentModel.DependencyPropertyDescriptor.FromProperty
                (ImageStackAlignatorController.ExposureProperty, typeof(ImageStackAlignatorController));
            imageColorChanged.AddValueChanged(this, new EventHandler(this.ImageColorChanged));
        }

        private void ImageColorChanged(object sender, EventArgs e)
        {
            if (_colorSpec != null)
            {
                DNGTemperature t = new DNGTemperature(ColorTemperature, ColorTint);
                _colorSpec.WhiteXY = t.xyCoord;
                DeBayerColorVisu(_selectedItem);
            }
        }

        public event PropertyChangedEventHandler PropertyChanged;
        protected virtual void OnPropertyChanged(string propertyName)
        {
            PropertyChangedEventHandler handler = PropertyChanged;
            if (handler != null)
            {
                handler(this, new PropertyChangedEventArgs(propertyName));
            }
        }

        #region DependendyProperties
        public ObservableCollection<string> FileNameList
        {
            get { return (ObservableCollection<string>)GetValue(FileNameListProperty); }
            set { SetValue(FileNameListProperty, value); }
        }

        public static readonly DependencyProperty FileNameListProperty =
            DependencyProperty.Register("FileNameList", typeof(ObservableCollection<string>), typeof(ImageStackAlignatorController));
        

        public string BaseDirectory
        {
            get { return (string)GetValue(BaseDirectoryProperty); }
            set { SetValue(BaseDirectoryProperty, value); }
        }

        public static readonly DependencyProperty BaseDirectoryProperty =
            DependencyProperty.Register("BaseDirectory", typeof(string), typeof(ImageStackAlignatorController));

        public float SigmaDebayerTracking
        {
            get { return (float)GetValue(SigmaDebayerTrackingProperty); }
            set { SetValue(SigmaDebayerTrackingProperty, value); }
        }

        public static readonly DependencyProperty SigmaDebayerTrackingProperty =
            DependencyProperty.Register("SigmaDebayerTracking", typeof(float), typeof(ImageStackAlignatorController), new PropertyMetadata(0.5f));

        public float SigmaDebayerAccumulation
        {
            get { return (float)GetValue(SigmaDebayerAccumulationProperty); }
            set { SetValue(SigmaDebayerAccumulationProperty, value); }
        }

        public static readonly DependencyProperty SigmaDebayerAccumulationProperty =
            DependencyProperty.Register("SigmaDebayerAccumulation", typeof(float), typeof(ImageStackAlignatorController), new PropertyMetadata(0.5f));

        public float HighPass
        {
            get { return (float)GetValue(HighPassProperty); }
            set { SetValue(HighPassProperty, value); }
        }

        public static readonly DependencyProperty HighPassProperty =
            DependencyProperty.Register("HighPass", typeof(float), typeof(ImageStackAlignatorController), new PropertyMetadata(0.01f));

        public float HighPassSigma
        {
            get { return (float)GetValue(HighPassSigmaProperty); }
            set { SetValue(HighPassSigmaProperty, value); }
        }

        public static readonly DependencyProperty HighPassSigmaProperty =
            DependencyProperty.Register("HighPassSigma", typeof(float), typeof(ImageStackAlignatorController), new PropertyMetadata(0.0025f));

        public int ClearAxis
        {
            get { return (int)GetValue(ClearAxisProperty); }
            set { SetValue(ClearAxisProperty, value); }
        }

        public static readonly DependencyProperty ClearAxisProperty =
            DependencyProperty.Register("ClearAxis", typeof(int), typeof(ImageStackAlignatorController), new PropertyMetadata(0));

        public float MatchingThreshold
        {
            get { return (float)GetValue(MatchingThresholdProperty); }
            set { SetValue(MatchingThresholdProperty, value); }
        }

        public static readonly DependencyProperty MatchingThresholdProperty =
            DependencyProperty.Register("MatchingThreshold", typeof(float), typeof(ImageStackAlignatorController), new PropertyMetadata(1.0f));

        public ObservableCollection<PatchTrackingLevel> PatchTrackingLevels
        {
            get { return (ObservableCollection<PatchTrackingLevel>)GetValue(PatchTrackingLevelsProperty); }
            set { SetValue(PatchTrackingLevelsProperty, value); }
        }

        public static readonly DependencyProperty PatchTrackingLevelsProperty =
            DependencyProperty.Register("PatchTrackingLevels", typeof(ObservableCollection<PatchTrackingLevel>), typeof(ImageStackAlignatorController));

        public bool ShowTilesWithPreAlignment
        {
            get { return (bool)GetValue(ShowTilesWithPreAlignmentProperty); }
            set { SetValue(ShowTilesWithPreAlignmentProperty, value); }
        }

        public static readonly DependencyProperty ShowTilesWithPreAlignmentProperty =
            DependencyProperty.Register("ShowTilesWithPreAlignment", typeof(bool), typeof(ImageStackAlignatorController), new PropertyMetadata(true));

        public float SigmaStructureTensor
        {
            get { return (float)GetValue(SigmaStructureTensorProperty); }
            set { SetValue(SigmaStructureTensorProperty, value); }
        }

        public static readonly DependencyProperty SigmaStructureTensorProperty =
            DependencyProperty.Register("SigmaStructureTensor", typeof(float), typeof(ImageStackAlignatorController), new PropertyMetadata(1.0f));

        public float Dth
        {
            get { return (float)GetValue(DthProperty); }
            set { SetValue(DthProperty, value); }
        }

        public static readonly DependencyProperty DthProperty =
            DependencyProperty.Register("Dth", typeof(float), typeof(ImageStackAlignatorController), new PropertyMetadata(0.001f));

        public float Dtr
        {
            get { return (float)GetValue(DtrProperty); }
            set { SetValue(DtrProperty, value); }
        }

        public static readonly DependencyProperty DtrProperty =
            DependencyProperty.Register("Dtr", typeof(float), typeof(ImageStackAlignatorController), new PropertyMetadata(0.006f));

        public float kDetail
        {
            get { return (float)GetValue(kDetailProperty); }
            set { SetValue(kDetailProperty, value); }
        }

        public static readonly DependencyProperty kDetailProperty =
            DependencyProperty.Register("kDetail", typeof(float), typeof(ImageStackAlignatorController), new PropertyMetadata(0.25f));

        public float kDenoise
        {
            get { return (float)GetValue(kDenoiseProperty); }
            set { SetValue(kDenoiseProperty, value); }
        }

        public static readonly DependencyProperty kDenoiseProperty =
            DependencyProperty.Register("kDenoise", typeof(float), typeof(ImageStackAlignatorController), new PropertyMetadata(3.0f));

        public float kStretch
        {
            get { return (float)GetValue(kStretchProperty); }
            set { SetValue(kStretchProperty, value); }
        }

        public static readonly DependencyProperty kStretchProperty =
            DependencyProperty.Register("kStretch", typeof(float), typeof(ImageStackAlignatorController), new PropertyMetadata(4.0f));

        public float kShrink
        {
            get { return (float)GetValue(kShrinkProperty); }
            set { SetValue(kShrinkProperty, value); }
        }

        public static readonly DependencyProperty kShrinkProperty =
            DependencyProperty.Register("kShrink", typeof(float), typeof(ImageStackAlignatorController), new PropertyMetadata(2.0f));

        public int IterationsLK
        {
            get { return (int)GetValue(IterationsLKProperty); }
            set { SetValue(IterationsLKProperty, value); }
        }

        public static readonly DependencyProperty IterationsLKProperty =
            DependencyProperty.Register("IterationsLK", typeof(int), typeof(ImageStackAlignatorController), new PropertyMetadata(3));

        public int WindowSizeLK
        {
            get { return (int)GetValue(WindowSizeLKProperty); }
            set { SetValue(WindowSizeLKProperty, value); }
        }

        public static readonly DependencyProperty WindowSizeLKProperty =
            DependencyProperty.Register("WindowSizeLK", typeof(int), typeof(ImageStackAlignatorController), new PropertyMetadata(11));

        public int ErodeSize
        {
            get { return (int)GetValue(ErodeSizeProperty); }
            set { SetValue(ErodeSizeProperty, value); }
        }

        public static readonly DependencyProperty ErodeSizeProperty =
            DependencyProperty.Register("ErodeSize", typeof(int), typeof(ImageStackAlignatorController), new PropertyMetadata(5));

        public float MinDetLK
        {
            get { return (float)GetValue(MinDetLKProperty); }
            set { SetValue(MinDetLKProperty, value); }
        }

        public static readonly DependencyProperty MinDetLKProperty =
            DependencyProperty.Register("MinDetLK", typeof(float), typeof(ImageStackAlignatorController), new PropertyMetadata(0.01f));

        public float ThresholdM
        {
            get { return (float)GetValue(ThresholdMProperty); }
            set { SetValue(ThresholdMProperty, value); }
        }

        public static readonly DependencyProperty ThresholdMProperty =
            DependencyProperty.Register("ThresholdM", typeof(float), typeof(ImageStackAlignatorController), new PropertyMetadata(100000.0f));

        public bool ClearResults
        {
            get { return (bool)GetValue(ClearResultsProperty); }
            set { SetValue(ClearResultsProperty, value); }
        }

        public static readonly DependencyProperty ClearResultsProperty =
            DependencyProperty.Register("ClearResults", typeof(bool), typeof(ImageStackAlignatorController), new PropertyMetadata(false));

        public WhatToShow ShowWhatResult
        {
            get { return (WhatToShow)GetValue(ShowWhatResultProperty); }
            set { SetValue(ShowWhatResultProperty, value); }
        }

        public static readonly DependencyProperty ShowWhatResultProperty =
            DependencyProperty.Register("ShowWhatResult", typeof(WhatToShow), typeof(ImageStackAlignatorController), new PropertyMetadata(WhatToShow.FinalImage));

        public bool SuperResolution
        {
            get { return (bool)GetValue(SuperResolutionProperty); }
            set { SetValue(SuperResolutionProperty, value); }
        }

        public static readonly DependencyProperty SuperResolutionProperty =
            DependencyProperty.Register("SuperResolution", typeof(bool), typeof(ImageStackAlignatorController), new PropertyMetadata(false));

        public ShiftCollection.TrackingStrategy TrackingStrategy
        {
            get { return (ShiftCollection.TrackingStrategy)GetValue(TrackingStrategyProperty); }
            set { SetValue(TrackingStrategyProperty, value); }
        }

        public static readonly DependencyProperty TrackingStrategyProperty =
            DependencyProperty.Register("TrackingStrategy", typeof(ShiftCollection.TrackingStrategy), typeof(ImageStackAlignatorController), new PropertyMetadata(ShiftCollection.TrackingStrategy.Full));

        public int TrackingBlockSize
        {
            get { return (int)GetValue(TrackingBlockSizeProperty); }
            set { SetValue(TrackingBlockSizeProperty, value); }
        }

        public static readonly DependencyProperty TrackingBlockSizeProperty =
            DependencyProperty.Register("TrackingBlockSize", typeof(int), typeof(ImageStackAlignatorController), new PropertyMetadata(5));

        public bool GreenChannelOnly
        {
            get { return (bool)GetValue(GreenChannelOnlyProperty); }
            set { SetValue(GreenChannelOnlyProperty, value); }
        }

        public static readonly DependencyProperty GreenChannelOnlyProperty =
            DependencyProperty.Register("GreenChannelOnly", typeof(bool), typeof(ImageStackAlignatorController), new PropertyMetadata(false));



        #region ImageInfo

        public bool HasTiltInfo
        {
            get { return (bool)GetValue(HasTiltInfoProperty); }
            set { SetValue(HasTiltInfoProperty, value); }
        }

        public static readonly DependencyProperty HasTiltInfoProperty =
            DependencyProperty.Register("HasTiltInfo", typeof(bool), typeof(ImageStackAlignatorController), new PropertyMetadata(false));

        public float TiltAngleDeg
        {
            get { return (float)GetValue(TiltAngleDegProperty); }
            set { SetValue(TiltAngleDegProperty, value); }
        }

        public static readonly DependencyProperty TiltAngleDegProperty =
            DependencyProperty.Register("TiltAngleDeg", typeof(float), typeof(ImageStackAlignatorController), new PropertyMetadata(0.0f));

        public string Make
        {
            get { return (string)GetValue(MakeProperty); }
            set { SetValue(MakeProperty, value); }
        }

        public static readonly DependencyProperty MakeProperty =
            DependencyProperty.Register("Make", typeof(string), typeof(ImageStackAlignatorController));

        public string Model
        {
            get { return (string)GetValue(ModelProperty); }
            set { SetValue(ModelProperty, value); }
        }

        public static readonly DependencyProperty ModelProperty =
            DependencyProperty.Register("Model", typeof(string), typeof(ImageStackAlignatorController));

        public PentaxPefFile.Rational ExposureTime
        {
            get { return (PentaxPefFile.Rational)GetValue(ExposureTimeProperty); }
            set { SetValue(ExposureTimeProperty, value); }
        }

        public static readonly DependencyProperty ExposureTimeProperty =
            DependencyProperty.Register("ExposureTime", typeof(PentaxPefFile.Rational), typeof(ImageStackAlignatorController), new PropertyMetadata(new PentaxPefFile.Rational(0,0)));


        public DateTime RecordingDate
        {
            get { return (DateTime)GetValue(RecordingDateProperty); }
            set { SetValue(RecordingDateProperty, value); }
        }

        public static readonly DependencyProperty RecordingDateProperty =
            DependencyProperty.Register("RecordingDate", typeof(DateTime), typeof(ImageStackAlignatorController));

        public float NoiseModelA
        {
            get { return (float)GetValue(NoiseModelAProperty); }
            set { SetValue(NoiseModelAProperty, value); }
        }

        public static readonly DependencyProperty NoiseModelAProperty =
            DependencyProperty.Register("NoiseModelA", typeof(float), typeof(ImageStackAlignatorController), new PropertyMetadata(0.0f));

        public float NoiseModelB
        {
            get { return (float)GetValue(NoiseModelBProperty); }
            set { SetValue(NoiseModelBProperty, value); }
        }

        public static readonly DependencyProperty NoiseModelBProperty =
            DependencyProperty.Register("NoiseModelB", typeof(float), typeof(ImageStackAlignatorController), new PropertyMetadata(0.0f));

        public ImagePresenterDX.Rotation Orientation
        {
            get { return (ImagePresenterDX.Rotation)GetValue(OrientationProperty); }
            set { SetValue(OrientationProperty, value); }
        }

        public static readonly DependencyProperty OrientationProperty =
            DependencyProperty.Register("Orientation", typeof(ImagePresenterDX.Rotation), typeof(ImageStackAlignatorController), new PropertyMetadata(ImagePresenterDX.Rotation._0));

        public float RotationSearchRange
        {
            get { return (float)GetValue(RotationSearchRangeProperty); }
            set { SetValue(RotationSearchRangeProperty, value); }
        }

        public static readonly DependencyProperty RotationSearchRangeProperty =
            DependencyProperty.Register("RotationSearchRange", typeof(float), typeof(ImageStackAlignatorController), new PropertyMetadata(1.0f));

        public float RotationSearchIncrement
        {
            get { return (float)GetValue(RotationSearchIncrementProperty); }
            set { SetValue(RotationSearchIncrementProperty, value); }
        }

        public static readonly DependencyProperty RotationSearchIncrementProperty =
            DependencyProperty.Register("RotationSearchIncrement", typeof(float), typeof(ImageStackAlignatorController), new PropertyMetadata(0.1f));



        #endregion

        #region ColorInfo

        public float Exposure
        {
            get { return (float)GetValue(ExposureProperty); }
            set { SetValue(ExposureProperty, value); }
        }

        public static readonly DependencyProperty ExposureProperty =
            DependencyProperty.Register("Exposure", typeof(float), typeof(ImageStackAlignatorController), new PropertyMetadata(0.0f));

        public double ColorTemperature
        {
            get { return (double)GetValue(ColorTemperatureProperty); }
            set 
            {
                SetValue(ColorTemperatureProperty, value);
            }
        }

        public static readonly DependencyProperty ColorTemperatureProperty =
            DependencyProperty.Register("ColorTemperature", typeof(double), typeof(ImageStackAlignatorController), new PropertyMetadata(5000.0));

        public double ColorTint
        {
            get { return (double)GetValue(ColorTintProperty); }
            set
            {
                SetValue(ColorTintProperty, value);
            }
        }

        public static readonly DependencyProperty ColorTintProperty =
            DependencyProperty.Register("ColorTint", typeof(double), typeof(ImageStackAlignatorController), new PropertyMetadata(0.0));
        #endregion


        #endregion

        #region Properties
        public bool IsReady
        {
            get { return _colorSpec != null; }
        }

        public bool IsPreAlignmentReady
        {
            get { return IsReady && _pefFiles.Count > 1; }
        }
        public bool IsPatchAlignmentReady
        {
            get { return IsPreAlignmentReady && _preAlignmentStore != null; }
        }
        public bool IsPrepareAccumulationReady
        {
            get { return IsPatchAlignmentReady && _shiftCollection != null; }
        }
        public bool IsAccumulationReady
        {
            get { return IsPrepareAccumulationReady && _debayerRefHalfRes != null; }
        }
        public bool IsPostProcessingReady
        {
            get { return IsAccumulationReady && _finalImage != null; }
        }

        public NPPImage_8uC4 Image
        {
            get { return _imageToShow; }
        }

        public float2 PreAlignmentShift
        {
            get {
                if (_workState == WorkState.Accumulate)
                    return new float2(0, 0);

                if (_workState == WorkState.PatchAlign && ShowTilesWithPreAlignment)
                    return new float2(0, 0);

                if (_selectedItem >= 0 && _preAlignmentStore != null)
                {
                    return _preAlignmentStore.GetShift(_selectedItem);
                }
                return new float2(0, 0);
            }
        }

        public float PreAlignmentRotation
        {
            get
            {
                if (_workState == WorkState.Accumulate)
                    return 0.0f;

                if (_workState == WorkState.PatchAlign && ShowTilesWithPreAlignment)
                    return 0.0f;

                if (_selectedItem >= 0 && _preAlignmentStore != null)
                {
                    return _preAlignmentStore.GetRotation(_selectedItem);
                }
                return 0;
            }
        }

        public bool CanRemovePatchTrackingLevel
        {
            get 
            {
                if (PatchTrackingLevels == null)
                    return false;

                return PatchTrackingLevels.Count > 1; 
            }
        }

        public int TileCountX
        {
            get
            {
                if (_pefFiles != null)
                {
                    if (_pefFiles.Count > 0)
                    {
                        int tileCountX = ((_pefFiles[0].RawWidth / PatchTrackingLevels[0].ResizeLevel) - PatchTrackingLevels[0].MaxShift * 2) / PatchTrackingLevels[0].TileSize;
                        return tileCountX;
                    }
                }
                return 0;
            }
        }

        public int TileCountY
        {
            get
            {
                if (_pefFiles != null)
                {
                    if (_pefFiles.Count > 0)
                    {
                        int tileCountY = ((_pefFiles[0].RawHeight / PatchTrackingLevels[0].ResizeLevel) - PatchTrackingLevels[0].MaxShift * 2) / PatchTrackingLevels[0].TileSize;
                        return tileCountY;
                    }
                }
                return 0;
            }
        }

        public int TileSize
        {
            get
            {
                return PatchTrackingLevels[0].TileSize;
            }
        }

        public int ResizeLevel
        {
            get
            {
                return PatchTrackingLevels[0].ResizeLevel;
            }
        }

        public int MaxShift
        {
            get
            {
                return PatchTrackingLevels[0].MaxShift;
            }
        }

        public int ReferenceIndex
        {
            get 
            {
                if (_preAlignmentStore != null)
                {
                    return _preAlignmentStore.ReferenceIndex;
                }
                return 0;
            }
            set
            {
                if (_preAlignmentStore != null)
                {
                    if (_preAlignmentStore.ReferenceIndex != value)
                    {
                        _preAlignmentStore.ReferenceIndex = value;
                        OnPropertyChanged("ReferenceIndex");
                    }
                }
                else
                {
                    //erase user input and return 0
                    OnPropertyChanged("ReferenceIndex");
                }
            }
        }

        public int MaxReferenceIndex 
        {
            get 
            { 
                if (_pefFiles != null && _pefFiles.Count > 0)
                    return _pefFiles.Count - 1;
                return 0;
            }
        }
        #endregion

        #region public Methods
        public void InitCuda(CudaContext ctx)
        {
            if (_ctx != null)
                return; //ignore...

            _ctx = ctx;
            CUmodule modDebayer = _ctx.LoadModulePTX("DeBayerKernels.ptx");
            deBayerGreenKernel = new DeBayerGreenKernel(modDebayer, _ctx);
            deBayerRedBlueKernel = new DeBayerRedBlueKernel(modDebayer, _ctx);
            debayerSubSample = new DeBayersSubSampleKernel(ctx, modDebayer);
            accumulateImages = new AccumulateImagesKernel(ctx, modDebayer);
            accumulateImagesSuperRes = new AccumulateImagesSuperResKernel(ctx, modDebayer);

            CUmodule modBasic = _ctx.LoadModulePTX("kernel.ptx");
            gammaCorrectionKernel = new GammasRGBKernel(_ctx, modBasic);
            applyWeightingKernel = new ApplyWeightingKernel(_ctx, modBasic);
            upsampleShifts = new upsampleShiftsKernel(_ctx, modBasic);
            computeStructureTensor = new computeStructureTensorKernel(ctx, modBasic);
            computeKernelParam = new computeKernelParamKernel(ctx, modBasic);

            CUmodule modOF = _ctx.LoadModulePTX("opticalFlow.ptx");
            computeDerivatives2Kernel = new ComputeDerivatives2Kernel(_ctx, modOF);

            CUmodule modRM = _ctx.LoadModulePTX("RobustnessModell.ptx");
            computeMask = new RobustnessModellKernel(_ctx, modRM);

            filterBuffer = new CudaDeviceVariable<float>(100);
        }

        public void ReadFiles()
        {
            _pefFiles = new List<RawFile>();
            _selected = new List<bool>();
            _debayerdImagesBW = new List<float[]>();
            _colorSpec = null;

            if (FileNameList.Count == 0)
            {
                return;
            }

            if (FileNameList.Count < 1)
                return;

            //check that all needed information is present:
            bool isOK = true;
            if (_isTestMode)
            {
                for (int i = 0; i < FileNameList.Count; i++)
                {
                    string filename = System.IO.Path.Combine(BaseDirectory, FileNameList[i]);

                    int idxShifts = 0;
                    if (i < 5)
                    {
                        idxShifts = i;
                    }
                    TestRawFile testRaw = new TestRawFile(filename, 1024, preShiftsX[idxShifts], preShiftsY[idxShifts], preRot[idxShifts], shiftX[idxShifts], shiftY[idxShifts]);
                    _pefFiles.Add(testRaw);
                    _selected.Add(false);
                    _debayerdImagesBW.Add(new float[testRaw.RawWidth * testRaw.RawHeight]);
                }
            }
            else
            {
                if (FileNameList[0].ToLower().EndsWith(".pef"))
                {
                    Task<PEFFile>[] pefs = new Task<PEFFile>[FileNameList.Count];

                    for (int i = 0; i < FileNameList.Count; i++)
                    {
                        string filename = System.IO.Path.Combine(BaseDirectory, FileNameList[i]);
                        Task<PEFFile> task = Task.Run(() =>
                        {
                            return new PEFFile(filename);
                        });

                        pefs[i] = task;
                    }

                    Task.WhenAll(pefs);

                    foreach (var item in pefs)
                    {
                        _pefFiles.Add(item.Result);
                        _selected.Add(false);
                        _debayerdImagesBW.Add(new float[item.Result.RawWidth * item.Result.RawHeight]);
                    }
                    //Load additional camera profiling (only for first raw image)
                    isOK = _pefFiles[0].LoadExtraCameraProfile("ExtraCameraProfiles.xml");
                }
                else
                {
                    Task<DNGFile>[] pefs = new Task<DNGFile>[FileNameList.Count];

                    for (int i = 0; i < FileNameList.Count; i++)
                    {
                        string filename = System.IO.Path.Combine(BaseDirectory, FileNameList[i]);
                        Task<DNGFile> task = Task.Run(() =>
                        {
                            return new DNGFile(filename);
                        });

                        pefs[i] = task;
                    }

                    Task.WhenAll(pefs);

                    foreach (var item in pefs)
                    {
                        _pefFiles.Add(item.Result);
                        _selected.Add(false);
                        _debayerdImagesBW.Add(new float[item.Result.RawWidth * item.Result.RawHeight]);
                    }
                    //Load additional camera profiling (only for first raw image) (noisemodel for dng)
                    _pefFiles[0].LoadExtraCameraProfile("ExtraCameraProfiles.xml");
                }

            }


            if (!isOK)
            {
                MessageBox.Show("Color spec not found.\nCheck for entries '" + _pefFiles[0].Make + "' and '" + _pefFiles[0].UniqueModelName + "' in 'ExtraCameraProfiles.xml' or use DNG");
                isOK = false;
            }

            if (_pefFiles[0].NoiseModelAlpha == 0)
            {
                MessageBox.Show("Noise model not found.\nCheck for entries '" + _pefFiles[0].Make + "' and '" + _pefFiles[0].UniqueModelName + "' in 'ExtraCameraProfiles.xml' or use DNG");
                isOK = false;
            }

            //set the values
            switch (_pefFiles[0].Orientation.Value)
            {
                case DNGOrientation.Orientation.Normal:
                    Orientation = ImagePresenterDX.Rotation._0;
                    break;
                case DNGOrientation.Orientation.Rotate90CW:
                    Orientation = ImagePresenterDX.Rotation._90;
                    break;
                case DNGOrientation.Orientation.Rotate180:
                    Orientation = ImagePresenterDX.Rotation._180;
                    break;
                case DNGOrientation.Orientation.Rotate90CCW:
                    Orientation = ImagePresenterDX.Rotation._270;
                    break;
                case DNGOrientation.Orientation.Mirror:
                    Orientation = ImagePresenterDX.Rotation._0;
                    break;
                case DNGOrientation.Orientation.Mirror90CW:
                    Orientation = ImagePresenterDX.Rotation._0;
                    break;
                case DNGOrientation.Orientation.Mirror180:
                    Orientation = ImagePresenterDX.Rotation._0;
                    break;
                case DNGOrientation.Orientation.Mirror90CCW:
                    Orientation = ImagePresenterDX.Rotation._0;
                    break;
                case DNGOrientation.Orientation.Unknown:
                    Orientation = ImagePresenterDX.Rotation._0;
                    break;
                default:
                    Orientation = ImagePresenterDX.Rotation._0;
                    break;
            }



            HasTiltInfo = _pefFiles[0].RollAnglePresent;

            if (HasTiltInfo)
            {
                //Pentax gives us a roll angle precision of 0.5 degrees, no need to search for much more...
                RotationSearchIncrement = 0.01f;
                RotationSearchRange = 0.8f;
            }
            else
            {
                RotationSearchIncrement = 0.1f;
                RotationSearchRange = 8.0f;
            }

            TiltAngleDeg = _pefFiles[0].RollAngle;
            Make = _pefFiles[0].Make;
            Model = _pefFiles[0].UniqueModelName;
            ExposureTime = _pefFiles[0].ExposureTime;
            RecordingDate = _pefFiles[0].RecordingDate;

            NoiseModelA = _pefFiles[0].NoiseModelAlpha;
            NoiseModelB = _pefFiles[0].NoiseModelBeta;


            if (_pefFiles[0].ColorSpec != null)
            {
                // Convert to temperature/offset space.
                DNGTemperature td = new DNGTemperature(_pefFiles[0].ColorSpec.WhiteXY);
                ColorTemperature = td.Temperature;
                ColorTint = td.Tint;
                _colorSpec = _pefFiles[0].ColorSpec;
            }

            _preAlignmentStore = null;

            AllocateDeviceMemory();
            OnPropertyChanged("IsReady");
            OnPropertyChanged("IsPreAlignmentReady");
            OnPropertyChanged("IsPatchAlignmentReady");
            OnPropertyChanged("IsAccumulationReady");
            OnPropertyChanged("MaxReferenceIndex");
        }

        public void WorkStateChanged(WorkState workState)
        {
            _workState = workState;
        }

        public void SelectedItemsChanged(System.Collections.IList selectedItems)
        {
            int oldIndex = -1;
            for (int i = 0; i < _pefFiles.Count; i++)
            {
                if (_selected[i])
                {
                    oldIndex = i;
                    break;
                }
            }

            for (int i = 0; i < _pefFiles.Count; i++)
            {
                _selected[i] = false;
            }

            foreach (var item in selectedItems)
            {
                for (int i = 0; i < _pefFiles.Count; i++)
                {
                    if (FileNameList[i] == item as string)
                    {
                        _selected[i] = true;
                    }
                }
            }

            int newIndex = -1;
            for (int i = 0; i < _pefFiles.Count; i++)
            {
                if (_selected[i])
                {
                    newIndex = i;
                    break;
                }
            }
            _selectedItem = newIndex;

            if (oldIndex != newIndex && (_workState == WorkState.Init || _workState == WorkState.PreAlign || _workState == WorkState.PatchAlign))
            {

                DeBayerColorVisu(newIndex);
                //DeBayerColorVisu(0);
            }
        }

        private void DeBayerFullRes(int index)
        {
            if (index < 0 || index >= _pefFiles.Count)
                return;

            RawFile pef = _pefFiles[index];
            RawFile forColor = _pefFiles[0];

            float3 whitePoint = new float3(forColor.WhiteLevel[0], forColor.WhiteLevel[1], forColor.WhiteLevel[2]);
            float3 blackPoint = new float3(forColor.BlackLevel[0], forColor.BlackLevel[1], forColor.BlackLevel[2]);
            float3 scaling = new float3(1.0f / (float)_colorSpec.CameraWhite[0], 1.0f / (float)_colorSpec.CameraWhite[1], 1.0f / (float)_colorSpec.CameraWhite[2]);

            _rawImage.CopyToDevice(pef.RawImage);
            _rawImage.Convert(_rawImageFloat);

            if (_isTestMode)
            {
                _imageToShow.Set(new byte[] { 255, 255, 255, 255 });
                NPPImage_8uC1 temp = new NPPImage_8uC1(_rawImage.DevicePointer, forColor.CroppedWidth, forColor.CroppedHeight, forColor.CroppedWidth);
                _rawImageFloat.Div(65535);
                _rawImageFloat.Mul(255);
                _rawImageFloat.Convert(temp, NppRoundMode.Near);

                temp.Copy(_imageToShow, 2);
                temp.Copy(_imageToShow, 1);
                temp.Copy(_imageToShow, 0);
                _decodedImage.ResetRoi();
                _rawImageFloat.ResetRoi();
            }
            else
            {
                _decodedImage.Set(new float[] { 0, 0, 0 });

                deBayerGreenKernel.BayerPattern = pef.BayerPattern; //same cumodule --> same for red/blue
                                                                    //we apply AsShot white balance for better demosaicing
                deBayerGreenKernel.RunSafe(_rawImageFloat, _decodedImage, blackPoint, scaling);
                deBayerRedBlueKernel.RunSafe(_rawImageFloat, _decodedImage, blackPoint, scaling);
                //remove again white balance as it is included in matrix CameraToXYZ
                _decodedImage.Div(new float[] { whitePoint.x / (float)_colorSpec.CameraWhite[0],
                                                whitePoint.y / (float)_colorSpec.CameraWhite[1], 
                                                whitePoint.z / (float)_colorSpec.CameraWhite[2] });

            }
        }

        public void DeBayerColorVisu(int index)
        {
            if (index < 0 || index >= _pefFiles.Count)
                return;
            DeBayerFullRes(index);


            if (_isTestMode)
            {
                OnPropertyChanged("Image");
            }
            else
            {
                ProcessColorsAndShow();
            }
        }

        public void DeBayerBWGaussWBVisu(int index)
        {
            if (index < 0 || index >= _pefFiles.Count)
                return;

            if (_preAlignment != null)
            {
                if (!_preAlignment.DimensionsFit(_rawImageFloat.Width, _rawImageFloat.Height))
                {
                    _preAlignment.FreeResources();
                    _preAlignment = null;
                }
            }

            if (_preAlignment == null)
            {
                _preAlignment = new PreAlignment(_rawImageFloat, _ctx);
                _preAlignment.AllocateDeviceMemory();
            }

            DeBayerFullRes(index);


            gammaCorrectionKernel.RunSafe(_decodedImage);

            float[] filter = gaussian_filter_1D(SigmaDebayerTracking);
            CudaDeviceVariable<float> filterDebayer = new CudaDeviceVariable<float>(filterBuffer.DevicePointer, false, filter.Length * sizeof(float));
            filterDebayer.CopyToDevice(filter);
            float[] colorToGray = new float[] { 0.2989f, 0.5870f, 0.1141f };
            if (GreenChannelOnly)
                colorToGray = new float[] { 0, 1, 0 };

            _decodedImage.ColorToGray(_rawImageFloat, colorToGray);
            NPPImage_32fC1 temp = new NPPImage_32fC1(_decodedImage.DevicePointer, _decodedImage.Width, _decodedImage.Height, _rawImageFloat.Pitch);

            if (_preAlignment != null)
            {
                if (_preAlignment.DimensionsFit(_rawImageFloat.Width, _rawImageFloat.Height))
                {
                    _preAlignment.FourierFilter(_rawImageFloat, ClearAxis, HighPass, HighPassSigma);
                }
            }

            _rawImageFloat.FilterGaussBorder(temp, filterDebayer, NppiBorderType.Replicate);

            temp.Mul(255 * (float)Math.Pow(2.0, Exposure));

            _imageToShow.Set(new byte[] { 255, 255, 255, 255 });
            NPPImage_8uC1 temp2 = new NPPImage_8uC1(_rawImageFloat.DevicePointer, _rawImageFloat.Width, _rawImageFloat.Height, _rawImageFloat.Width);
            temp.Convert(temp2, NppRoundMode.Near);
            temp2.SetRoi(_pefFiles[0].CropLeft, _pefFiles[0].CropTop, _pefFiles[0].CroppedWidth, _pefFiles[0].CroppedHeight);
            temp2.Copy(_imageToShow, 0);
            temp2.Copy(_imageToShow, 1);
            temp2.Copy(_imageToShow, 2);
            temp2.ResetRoi();

            OnPropertyChanged("Image");
        }

        public void TestCC()
        {
            if (!IsPreAlignmentReady)
                return;

            if (_preAlignment != null)
            {
                if (!_preAlignment.DimensionsFit(_rawImageFloat.Width, _rawImageFloat.Height))
                {
                    _preAlignment.FreeResources();
                    _preAlignment = null;
                }
            }

            if (_preAlignment == null)
            {
                _preAlignment = new PreAlignment(_rawImageFloat, _ctx);
                _preAlignment.AllocateDeviceMemory();
            }

            float[] filter = gaussian_filter_1D(SigmaDebayerTracking);
            CudaDeviceVariable<float> filterDebayer = new CudaDeviceVariable<float>(filterBuffer.DevicePointer, false, filter.Length * sizeof(float));
            filterDebayer.CopyToDevice(filter);

            DeBayerBWGaussWB(0, filterDebayer);
            DeBayerBWGaussWB(1, filterDebayer);

            int sx1, sy1, sx2, sy2;
            _rawImageFloat.CopyToDevice(_debayerdImagesBW[0]);
            _preAlignment.SetReferenceImage(_rawImageFloat);

            //shift second image by 5 pixels in X and Y
            _rawImageFloat.CopyToDevice(_debayerdImagesBW[1]);
            (sx1, sy1) = _preAlignment.TestCC(_rawImageFloat, 5, 5);

            //test again with original image: we should have a difference of 5 pixels...
            _rawImageFloat.CopyToDevice(_debayerdImagesBW[1]);
            (sx2, sy2) = _preAlignment.TestCC(_rawImageFloat, 0, 0);

            if (sx1 - sx2 != 5 && sy1 - sy2 != 5)
            {
                MessageBox.Show("Expected a shift of 5 pixels in X and Y.\n\nBut got: X = " + (sx1 - sx2) + " Y = " + (sy1 - sy2));
            }
            else
            {
                MessageBox.Show("Found the expected shift of 5 pixels in X and Y!\n\nSettings seem good.");
            }

            _rawImageFloat.Mul(255);
            _imageToShow.Set(new byte[] { 255, 255, 255, 255 });
            NPPImage_8uC1 temp2 = new NPPImage_8uC1(_decodedImage.DevicePointer, _rawImageFloat.Width, _rawImageFloat.Height, _rawImageFloat.Width);
            _rawImageFloat.Convert(temp2, NppRoundMode.Near);

            temp2.SetRoi(_pefFiles[0].CropLeft, _pefFiles[0].CropTop, _pefFiles[0].CroppedWidth, _pefFiles[0].CroppedHeight);
            temp2.Copy(_imageToShow, 0);
            temp2.Copy(_imageToShow, 1);
            temp2.Copy(_imageToShow, 2);
            temp2.ResetRoi();

            OnPropertyChanged("Image");
        }

        public void ComputePreAlignment()
        {
            if (!IsPreAlignmentReady)
                return;

            float[] filter = gaussian_filter_1D(SigmaDebayerTracking);
            CudaDeviceVariable<float> filterDebayer = new CudaDeviceVariable<float>(filterBuffer.DevicePointer, false, filter.Length * sizeof(float));
            filterDebayer.CopyToDevice(filter);

            float rollReference = -_pefFiles[0].RollAngle;

            if (_preAlignment != null)
            {
                if (!_preAlignment.DimensionsFit(_rawImageFloat.Width, _rawImageFloat.Height))
                {
                    _preAlignment.FreeResources();
                    _preAlignment = null;
                }
            }

            if (_preAlignment == null)
            {
                _preAlignment = new PreAlignment(_rawImageFloat, _ctx);
                _preAlignment.AllocateDeviceMemory();
            }

            for (int i = 0; i < _pefFiles.Count; i++)
            {
                DeBayerBWGaussWB(i, filterDebayer);
            }

            _rawImageFloat.CopyToDevice(_debayerdImagesBW[0]);

            _preAlignment.SetReferenceImage(_rawImageFloat);


            double4[] initialShifts = new double4[_pefFiles.Count];

            for (int toTrack = 1; toTrack < _pefFiles.Count; toTrack++)
            {
                float rollTrack = -_pefFiles[toTrack].RollAngle;

                _rawImageFloat.CopyToDevice(_debayerdImagesBW[toTrack]);

                if (_isTestMode)
                    initialShifts[toTrack] = _preAlignment.ScanAngles(_rawImageFloat, 0.2, 1, 0);
                else
                    initialShifts[toTrack] = _preAlignment.ScanAngles(_rawImageFloat, RotationSearchIncrement, RotationSearchRange, rollTrack - rollReference);

                Console.WriteLine("Shift X: " + initialShifts[toTrack].x + ", Y: " + initialShifts[toTrack].y + ", angle: " + initialShifts[toTrack].z.ToString("0.000°"));
            }

            if (_isTestMode)
            {
                for (int i = 0; i < _pefFiles.Count; i++)
                {
                    int idx = i;
                    if (i >= 5)
                    {
                        idx = 0;
                    }
                    initialShifts[i] = new double4(preShiftsX[idx], preShiftsY[idx], preRot[idx], 0);
                }
            }

            _preAlignmentStore = new PreAlignmentStore(initialShifts);
            if (_isTestMode)
            {
                _preAlignmentStore.ReferenceIndex = 0;
            }
            _preAlignment.FreeResources();
            
            OnPropertyChanged("IsReady");
            OnPropertyChanged("IsPreAlignmentReady");
            OnPropertyChanged("IsPatchAlignmentReady");
            OnPropertyChanged("IsAccumulationReady");
            OnPropertyChanged("ReferenceIndex");
        }

        public void SkipPreAlignment()
        {
            if (!IsPreAlignmentReady)
                return;

            float[] filter = gaussian_filter_1D(SigmaDebayerTracking);
            CudaDeviceVariable<float> filterDebayer = new CudaDeviceVariable<float>(filterBuffer.DevicePointer, false, filter.Length * sizeof(float));
            filterDebayer.CopyToDevice(filter);

            if (_preAlignment != null)
            {
                if (!_preAlignment.DimensionsFit(_rawImageFloat.Width, _rawImageFloat.Height))
                {
                    _preAlignment.FreeResources();
                    _preAlignment = null;
                }
            }

            if (_preAlignment == null)
            {
                _preAlignment = new PreAlignment(_rawImageFloat, _ctx);
                _preAlignment.AllocateDeviceMemory();
            }
            
            for (int i = 0; i < _pefFiles.Count; i++)
            {
                DeBayerBWGaussWB(i, filterDebayer);
            }

            _preAlignment.FreeResources();

            double4[] initialShifts = new double4[_pefFiles.Count];
            _preAlignmentStore = new PreAlignmentStore(initialShifts);
            
            OnPropertyChanged("IsReady");
            OnPropertyChanged("IsPreAlignmentReady");
            OnPropertyChanged("IsPatchAlignmentReady");
            OnPropertyChanged("IsAccumulationReady");
            OnPropertyChanged("ReferenceIndex");
        }

        public void AddPatchTrackingLevel()
        {
            if (PatchTrackingLevels == null)
                PatchTrackingLevels = new ObservableCollection<PatchTrackingLevel>();

            if (PatchTrackingLevels.Count == 0)
            {
                PatchTrackingLevels.Add(new PatchTrackingLevel());
                return;
            }

            PatchTrackingLevel oldLevel = PatchTrackingLevels[PatchTrackingLevels.Count - 1];
            PatchTrackingLevel newLevel = new PatchTrackingLevel(oldLevel.ResizeLevel * 2, 32, 4);
            PatchTrackingLevels.Add(newLevel);
            OnPropertyChanged("CanRemovePatchTrackingLevel");
        }

        public void RemovePatchTrackingLevel()
        {
            if (PatchTrackingLevels != null)
            {

                if (PatchTrackingLevels.Count > 1)
                {
                    PatchTrackingLevels.RemoveAt(PatchTrackingLevels.Count - 1);
                }
            }
            OnPropertyChanged("CanRemovePatchTrackingLevel");
        }

        public void TrackPatches()
        {
            if (!IsPatchAlignmentReady)
                return;

            if (_shiftCollection != null)
            {
                _shiftCollection.FreeResources();
            }

            List<int> resizeLevels = new List<int>();
            List<int> maxShifts = new List<int>();
            List<int> tileSizes = new List<int>();

            foreach (var item in PatchTrackingLevels)
            {
                resizeLevels.Add(item.ResizeLevel);
                maxShifts.Add(item.MaxShift);
                tileSizes.Add(item.TileSize);
            }

            PatchTracker patchTracker = new PatchTracker(_pefFiles[0].RawWidth, _pefFiles[0].RawHeight, tileSizes, maxShifts, resizeLevels, _ctx);
            _shiftCollection = new ShiftCollection(_pefFiles.Count, patchTracker.MaxBlockCountX, patchTracker.MaxBlockCountY, _preAlignmentStore.ReferenceIndex, TrackingStrategy, TrackingBlockSize, _ctx);

            patchTracker.AllocateDeviceMemory();
            NPPImage_32fC1 imgToTrack = new NPPImage_32fC1(_pefFiles[0].RawWidth, _pefFiles[0].RawHeight);
            NPPImage_32fC1 imgReference = new NPPImage_32fC1(_pefFiles[0].RawWidth, _pefFiles[0].RawHeight);
            NPPImage_32fC2 shifts_temp = new NPPImage_32fC2(_pefFiles[0].RawWidth, _pefFiles[0].RawHeight);
            //reuse previously allocated buffers for temp images:
            NPPImage_32fC1 img1Work = new NPPImage_32fC1(_rawImageFloat.DevicePointer, _pefFiles[0].RawWidth, _pefFiles[0].RawHeight, imgToTrack.Pitch);
            NPPImage_32fC1 img2Work = new NPPImage_32fC1(_decodedImage.DevicePointer, _pefFiles[0].RawWidth, _pefFiles[0].RawHeight, imgToTrack.Pitch);




            int oldPatchCountX = 0;
            int oldPatchCountY = 0;
            int oldLevel = 0;

            for (int i = resizeLevels.Count - 1; i >= 0; i--)
            {
                int level = resizeLevels[i];
                int maxShift = maxShifts[i];
                int tileSize = tileSizes[i];

                if (i == resizeLevels.Count - 1)
                {
                    patchTracker.InitForSize(_pefFiles[0].RawWidth / level, _pefFiles[0].RawHeight / level, tileSize, maxShift);
                    oldPatchCountX = patchTracker.CurrentBlockCountX;
                    oldPatchCountY = patchTracker.CurrentBlockCountY;
                    oldLevel = level;
                }
                else
                {
                    //scale shifts
                    patchTracker.InitForSize(_pefFiles[0].RawWidth / level, _pefFiles[0].RawHeight / level, tileSize, maxShift);

                    foreach (var item in _shiftCollection.GetShiftPairs())
                    {
                        NPPImage_32fC2 currentShift = _shiftCollection[item.reference, item.toTrack];
                        shifts_temp.SetRoi(0, 0, currentShift.WidthRoi, currentShift.HeightRoi);
                        upsampleShifts.RunSafe(currentShift, shifts_temp, oldLevel, level, oldPatchCountX, oldPatchCountY, patchTracker.CurrentBlockCountX, patchTracker.CurrentBlockCountY, tileSizes[i + 1], tileSizes[i]);

                        currentShift.CopyToDeviceRoi(shifts_temp);
                    }

                    oldPatchCountX = patchTracker.CurrentBlockCountX;
                    oldPatchCountY = patchTracker.CurrentBlockCountY;
                    oldLevel = level;
                }

                Console.WriteLine("Patch count X: " + patchTracker.CurrentBlockCountX + "; Y: " + patchTracker.CurrentBlockCountY);

                for (int f = 0; f < _pefFiles.Count; f++)
                {
                    Console.WriteLine(_preAlignmentStore.GetShift(f));
                }

                foreach (var item in _shiftCollection.GetShiftPairs())
                {
                    imgToTrack.CopyToDevice(_debayerdImagesBW[item.toTrack]);
                    imgReference.CopyToDevice(_debayerdImagesBW[item.reference]);

                    img1Work.ResetRoi();
                    img2Work.ResetRoi();

                    img1Work.Set(0);
                    img2Work.Set(0);

                    img1Work.SetRoi(0, 0, img1Work.WidthRoi / level, img1Work.HeightRoi / level);
                    img2Work.SetRoi(0, 0, img2Work.WidthRoi / level, img2Work.HeightRoi / level);

                    if (level > 1)
                    {
                        imgToTrack.ResizeSqrPixel(img1Work, 1.0f / level, 1.0f / level, 0, 0, InterpolationMode.SuperSampling);
                        imgReference.ResizeSqrPixel(img2Work, 1.0f / level, 1.0f / level, 0, 0, InterpolationMode.SuperSampling);
                    }
                    else
                    {
                        imgToTrack.Copy(img1Work);
                        imgReference.Copy(img2Work);
                    }

                    patchTracker.Track(img1Work, img2Work, _shiftCollection[item.reference, item.toTrack], i,
                        _preAlignmentStore.GetShift(item.reference) / level, _preAlignmentStore.GetRotation(item.reference),
                        _preAlignmentStore.GetShift(item.toTrack) / level, _preAlignmentStore.GetRotation(item.toTrack),
                        MatchingThreshold);
                }

                _shiftCollection.MinimizeCUBLAS(patchTracker.CurrentBlockCountX, patchTracker.CurrentBlockCountY);
            }

            if (_isTestMode)
            {
                for (int i = 0; i < _pefFiles.Count; i++)
                {
                    float2[] shifts = _shiftCollection.getOptimalShift(i).ToCudaPitchedDeviceVariable();

                    int tileIdxX = 1;
                    int tileIdxY = 1;

                    if (tileIdxX >= patchTracker.CurrentBlockCountX)
                        tileIdxX = patchTracker.CurrentBlockCountX - 1;
                    if (tileIdxY >= patchTracker.CurrentBlockCountY)
                        tileIdxY = patchTracker.CurrentBlockCountY - 1;

                    int tileIdx = tileIdxY * patchTracker.CurrentBlockCountX + tileIdxX;

                    Console.WriteLine("(" + i + ") Found shift: " + shifts[tileIdx].x.ToString("0.00") + ", " + shifts[tileIdx].y.ToString("0.00"));
                    Console.WriteLine("(" + i + ") Found total shift: " + (shifts[tileIdx].x + _preAlignmentStore.GetShift(i).x).ToString("0.00") + ", " + (shifts[tileIdx].y + _preAlignmentStore.GetShift(i).y).ToString("0.00"));
                }
            }

            Console.WriteLine((long)(_ctx.GetFreeDeviceMemorySize()) / 1024.0 / 1024.0 + " free of " + (long)(_ctx.GetTotalDeviceMemorySize()) / 1024.0 / 1024.0);
            patchTracker.FreeDeviceMemory();
            _shiftCollection.FreeUneededResources();
            Console.WriteLine((long)(_ctx.GetFreeDeviceMemorySize()) / 1024.0 / 1024.0 + " free of " + (long)(_ctx.GetTotalDeviceMemorySize()) / 1024.0 / 1024.0);
            imgToTrack.Dispose();
            shifts_temp.Dispose();
            OnPropertyChanged("IsReady");
            OnPropertyChanged("IsPreAlignmentReady");
            OnPropertyChanged("IsPatchAlignmentReady");
            OnPropertyChanged("IsPrepareAccumulationReady");
            OnPropertyChanged("IsAccumulationReady");
        }

        public float2[] GetTrackedPatchFlow(int index)
        {
            if (_shiftCollection == null)
                return null;

            NPPImage_32fC2 shifts = _shiftCollection.getOptimalShift(index);
            float2[] ret = new float2[shifts.WidthRoi * shifts.HeightRoi];
            shifts.CopyToHost(ret);


            if (ShowTilesWithPreAlignment)
            {
                float2 baseShift = _preAlignmentStore.GetShift(index);
                float baseRotation = _preAlignmentStore.GetRotation(index);

                for (int y = 0; y < shifts.HeightRoi; y++)
                {
                    for (int x = 0; x < shifts.WidthRoi; x++)
                    {
                        int i = y * shifts.WidthRoi + x;
                        //add base shift and rotation
                        ret[i].x -= (float)(Math.Cos(baseRotation) * baseShift.x - Math.Sin(baseRotation) * baseShift.y);
                        ret[i].y -= (float)(Math.Sin(baseRotation) * baseShift.x + Math.Cos(baseRotation) * baseShift.y);

                        float patchCenterX = x * TileSize + TileSize / 2 - _pefFiles[0].RawWidth / 2; //in pixels
                        float patchCenterY = y * TileSize + TileSize / 2 - _pefFiles[0].RawHeight / 2;

                        ret[i].x += (float)(Math.Cos(baseRotation) * patchCenterX - Math.Sin(baseRotation) * patchCenterY - patchCenterX);
                        ret[i].y += (float)(Math.Sin(baseRotation) * patchCenterX + Math.Cos(baseRotation) * patchCenterY - patchCenterY);
                    }
                }
            }
            return ret;
        }

        public float3 GetKernel(int pixelX, int pixelY)
        {
            if (_kernelParameters == null)
                return new float3(100, 100, 0);

            int index = pixelY * _pefFiles[0].RawWidth + pixelX;
            if (index >= _kernelParameters.Length)
                return new float3(100, 100, 0);

            return _kernelParameters[index];
        }

        public void PrepareAccumulation()
        {
            if (!IsPrepareAccumulationReady)
                return;

            //debayer images again but now with settings for accumulation
            float[] filter = gaussian_filter_1D(SigmaDebayerAccumulation);
            CudaDeviceVariable<float> filterDebayer = new CudaDeviceVariable<float>(filterBuffer.DevicePointer, false, filter.Length * sizeof(float));
            filterDebayer.CopyToDevice(filter);

            for (int i = 0; i < _pefFiles.Count; i++)
            {
                DeBayerBWGaussWB(i, filterDebayer, true);
            }


            filter = gaussian_filter_1D(SigmaStructureTensor);
            CudaDeviceVariable<float> filterStructureTensor = new CudaDeviceVariable<float>(filterBuffer.DevicePointer, false, filter.Length * sizeof(float));
            filterStructureTensor.CopyToDevice(filter);

            int refImage = _preAlignmentStore.ReferenceIndex;
            Console.WriteLine("Reference image is: " + refImage);
            
            debayerSubSample.BayerPattern = _pefFiles[refImage].BayerPattern;

            _finalImage?.Dispose();
            _totalWeight?.Dispose();
            _structureTensor?.Dispose();
            _debayerRefHalfRes?.Dispose();
            _debayerTrackHalfRes?.Dispose();
            _uncertaintyMask?.Dispose();
            _uncertaintyMaskEroded?.Dispose();

            _structureTensor4?.Dispose();
            


            _totalWeight = new NPPImage_32fC3(_pefFiles[refImage].RawWidth, _pefFiles[refImage].RawHeight);
            _finalImage = new NPPImage_32fC3(_pefFiles[refImage].RawWidth, _pefFiles[refImage].RawHeight);
            _structureTensor = new NPPImage_32fC3(_pefFiles[refImage].RawWidth, _pefFiles[refImage].RawHeight);

            if (SuperResolution)
            {
                //we need a float4 in order to use texture interpolation
                _structureTensor4 = new NPPImage_32fC4(_pefFiles[refImage].RawWidth, _pefFiles[refImage].RawHeight);
            }

            NPPImage_32fC1 d_dx = new NPPImage_32fC1(_pefFiles[refImage].RawWidth, _pefFiles[refImage].RawHeight);
            NPPImage_32fC1 d_dy = new NPPImage_32fC1(_pefFiles[refImage].RawWidth, _pefFiles[refImage].RawHeight);
            NPPImage_32fC2 zeroShift = new NPPImage_32fC2(_pefFiles[refImage].RawWidth, _pefFiles[refImage].RawHeight);

            _debayerRefHalfRes = new NPPImage_32fC3(_pefFiles[refImage].RawWidth / 2, _pefFiles[refImage].RawHeight / 2);
            _debayerTrackHalfRes = new NPPImage_32fC3(_pefFiles[refImage].RawWidth / 2, _pefFiles[refImage].RawHeight / 2);
            _uncertaintyMask = new NPPImage_32fC4(_pefFiles[refImage].RawWidth / 2, _pefFiles[refImage].RawHeight / 2);
            _uncertaintyMaskEroded = new NPPImage_32fC4(_pefFiles[refImage].RawWidth / 2, _pefFiles[refImage].RawHeight / 2);

            _rawImageFloat.CopyToDevice(_debayerdImagesBW[refImage]);
            computeDerivatives2Kernel.RunSafe(_rawImageFloat, d_dx, d_dy);

            computeStructureTensor.RunSafe(d_dx, d_dy, _totalWeight);
            _totalWeight.FilterGaussBorder(_structureTensor, filterStructureTensor, NppiBorderType.Replicate);


            computeKernelParam.RunSafe(_structureTensor, Dth, Dtr, kDetail, kDenoise, kStretch, kShrink);
            _totalWeight.Set(new float[] { 0, 0, 0 });
            _finalImage.Set(new float[] { 0, 0, 0 });

            _kernelParameters = new float3[_structureTensor.Width * _structureTensor.Height];
            _structureTensor.CopyToHost(_kernelParameters);

            if (SuperResolution)
            {
                _structureTensor4.Set(new float[] { 0, 0, 0, 0 });
                _structureTensor.Copy(d_dx, 0);
                d_dx.Copy(_structureTensor4, 0);
                _structureTensor.Copy(d_dx, 1);
                d_dx.Copy(_structureTensor4, 1);
                _structureTensor.Copy(d_dx, 2);
                d_dx.Copy(_structureTensor4, 2);

                _structureTensor.Dispose();
            }

            float[] blackLevel = _pefFiles[refImage].BlackLevel;
            float[] maxLevel = _pefFiles[refImage].WhiteLevel;
            maxLevel[0] += blackLevel[0];
            maxLevel[1] += blackLevel[1];
            maxLevel[2] += blackLevel[2];
            float maxValue = maxLevel.Max();

            _rawImageNoPitch.CopyToDevice(_pefFiles[refImage].RawImage);
            debayerSubSample.RunSafe(_rawImageNoPitch, _debayerRefHalfRes, maxValue);


            _uncertaintyMask.Set(new float[] { 1, 1, 1, 1 });
            zeroShift.Set(new float[] { 0, 0 });

            if (SuperResolution)
            {
                accumulateImagesSuperRes.RunSafe(_rawImageNoPitch, _finalImage, _totalWeight, _uncertaintyMask, _structureTensor4, zeroShift, maxValue);
                _ctx.Synchronize();
            }
            else
            {
                accumulateImages.RunSafe(_rawImageNoPitch, _finalImage, _totalWeight, _uncertaintyMask, _structureTensor, zeroShift, maxValue);
            }

            zeroShift.Dispose();
            d_dy.Dispose();
            d_dx.Dispose();

            if (SuperResolution)
            {
                applyWeightingKernel.RunSafe(_decodedImage, _finalImage, _totalWeight, 0);
                _ctx.Synchronize();
            }
            else
            {
                DeBayerFullRes(refImage);
                applyWeightingKernel.RunSafe(_decodedImage, _finalImage, _totalWeight, 1);
            }
            

            ProcessColorsAndShow();
            OnPropertyChanged("IsReady");
            OnPropertyChanged("IsPreAlignmentReady");
            OnPropertyChanged("IsPatchAlignmentReady");
            OnPropertyChanged("IsPrepareAccumulationReady");
            OnPropertyChanged("IsAccumulationReady");
        }

        public void Accumulate()
        {
            if (!IsAccumulationReady)
                return;

            if (SuperResolution && _structureTensor4 == null)
            {
                return;
            }

            OpticalFlow of = new OpticalFlow(_pefFiles[0].RawWidth, _pefFiles[0].RawHeight, _ctx);
            int refImage = _preAlignmentStore.ReferenceIndex;

            NPPImage_32fC1 imgToTrack = new NPPImage_32fC1(_pefFiles[0].RawWidth, _pefFiles[0].RawHeight);
            CudaDeviceVariable<byte> morph = new CudaDeviceVariable<byte>(ErodeSize * ErodeSize);
            morph.Set(255);

            _rawImageFloat.CopyToDevice(_debayerdImagesBW[refImage]);

            if (ClearResults)
            {
                _totalWeight.Set(new float[] { 0, 0, 0 });
                _finalImage.Set(new float[] { 0, 0, 0 });
            }
            float[] blackLevel = _pefFiles[refImage].BlackLevel;
            float[] maxLevel = _pefFiles[refImage].WhiteLevel;
            maxLevel[0] += blackLevel[0];
            maxLevel[1] += blackLevel[1];
            maxLevel[2] += blackLevel[2];
            float maxValue = maxLevel.Max();

            for (int i = 0; i < _pefFiles.Count; i++)
            {                
                if (i != refImage && _selected[i])
                {
                    _rawImageNoPitch.CopyToDevice(_pefFiles[i].RawImage);
                    debayerSubSample.RunSafe(_rawImageNoPitch, _debayerTrackHalfRes, maxValue);


                    imgToTrack.CopyToDevice(_debayerdImagesBW[i]);
                    
                    //imgToTrack contains the warped image after LucasKanade
                    of.LucasKanade(imgToTrack, _rawImageFloat, _shiftCollection.getOptimalShift(i), PatchTrackingLevels[0].TileSize, TileCountX, TileCountY, IterationsLK, _preAlignmentStore.GetShift(refImage, i), _preAlignmentStore.GetRotation(refImage, i), MinDetLK, WindowSizeLK);
                    
                    
                    computeMask.RunSafe(_debayerRefHalfRes, _debayerTrackHalfRes, _uncertaintyMask, of.LastFlow, (float)NoiseModelA, (float)NoiseModelB, ThresholdM);
                    _uncertaintyMask.ErodeBorder(_uncertaintyMaskEroded, morph, new NppiSize(ErodeSize, ErodeSize), new NppiPoint(ErodeSize / 2, ErodeSize / 2), NppiBorderType.Replicate);

                    if (SuperResolution)
                    {
                        accumulateImagesSuperRes.RunSafe(_rawImageNoPitch, _finalImage, _totalWeight, _uncertaintyMaskEroded, _structureTensor4, of.LastFlow, maxValue);
                    }
                    else
                    {
                        accumulateImages.RunSafe(_rawImageNoPitch, _finalImage, _totalWeight, _uncertaintyMaskEroded, _structureTensor, of.LastFlow, maxValue);
                    }
                }
            }

            NPPImage_8uC1 temp = new NPPImage_8uC1(_rawImage.DevicePointer, _rawImageFloat.Width, _rawImageFloat.Height, _rawImageFloat.Width);
            switch (ShowWhatResult)
            {
                case WhatToShow.FinalImage:
                    //prepare final image, but without touching it
                    if (SuperResolution)
                    {                        
                        applyWeightingKernel.RunSafe(_decodedImage, _finalImage, _totalWeight, 0);
                    }
                    else
                    {
                        DeBayerFullRes(refImage);
                        float3 whitePoint = new float3(_pefFiles[0].WhiteLevel[0], _pefFiles[0].WhiteLevel[1], _pefFiles[0].WhiteLevel[2]);
                        float3 blackPoint = new float3(_pefFiles[0].BlackLevel[0], _pefFiles[0].BlackLevel[1], _pefFiles[0].BlackLevel[2]);
                        
                        applyWeightingKernel.RunSafe(_decodedImage, _finalImage, _totalWeight, 1);
                    }
                    

                    ProcessColorsAndShow();
                    break;
                case WhatToShow.WeightingsRed:
                    {
                        CudaDeviceVariable<float> maxVals = new CudaDeviceVariable<float>(3);

                        _totalWeight.Max(maxVals);
                        float[] maxs = maxVals;
                        float max = maxs.Max();

                        _totalWeight.Div(new float[] { max, max, max }, _decodedImage);
                        _decodedImage.Mul(new float[] { 255, 255, 255 });

                        _imageToShow.Set(new byte[] { 255, 255, 255, 255 });

                        _decodedImage.Copy(_rawImageFloat, 0);
                        _rawImageFloat.Convert(temp, NppRoundMode.Near);
                        temp.SetRoi(_pefFiles[0].CropLeft, _pefFiles[0].CropTop, _pefFiles[0].CroppedWidth, _pefFiles[0].CroppedHeight);
                        temp.Copy(_imageToShow, 2);
                        temp.Copy(_imageToShow, 1);
                        temp.Copy(_imageToShow, 0);
                        maxVals.Dispose();
                        OnPropertyChanged("Image");
                    }
                    break;
                case WhatToShow.WeightingsGreen:
                    {
                        CudaDeviceVariable<float> maxVals = new CudaDeviceVariable<float>(3);

                        _totalWeight.Max(maxVals);
                        float[] maxs = maxVals;
                        float max = maxs.Max();

                        _totalWeight.Div(new float[] { max, max, max }, _decodedImage);
                        _decodedImage.Mul(new float[] { 255, 255, 255 });

                        _imageToShow.Set(new byte[] { 255, 255, 255, 255 });

                        _decodedImage.Copy(_rawImageFloat, 1);
                        _rawImageFloat.Convert(temp, NppRoundMode.Near);
                        temp.SetRoi(_pefFiles[0].CropLeft, _pefFiles[0].CropTop, _pefFiles[0].CroppedWidth, _pefFiles[0].CroppedHeight);
                        temp.Copy(_imageToShow, 2);
                        temp.Copy(_imageToShow, 1);
                        temp.Copy(_imageToShow, 0);
                        maxVals.Dispose();
                        OnPropertyChanged("Image");
                    }
                    break;
                case WhatToShow.WeightingsBlue:
                    {
                        CudaDeviceVariable<float> maxVals = new CudaDeviceVariable<float>(3);

                        _totalWeight.Max(maxVals);
                        float[] maxs = maxVals;
                        float max = maxs.Max();

                        _totalWeight.Div(new float[] { max, max, max }, _decodedImage);
                        _decodedImage.Mul(new float[] { 255, 255, 255 });

                        _imageToShow.Set(new byte[] { 255, 255, 255, 255 });

                        _decodedImage.Copy(_rawImageFloat, 2);
                        _rawImageFloat.Convert(temp, NppRoundMode.Near);
                        temp.SetRoi(_pefFiles[0].CropLeft, _pefFiles[0].CropTop, _pefFiles[0].CroppedWidth, _pefFiles[0].CroppedHeight);
                        temp.Copy(_imageToShow, 2);
                        temp.Copy(_imageToShow, 1);
                        temp.Copy(_imageToShow, 0);
                        maxVals.Dispose();
                        OnPropertyChanged("Image");
                    }
                    break;
                case WhatToShow.CertaintyMaskRed:
                    {
                        _uncertaintyMaskEroded.Mul(new float[] { 255, 255, 255, 255 });
                        _imageToShow.Set(new byte[] { 255, 255, 255, 255 });
                        _rawImageFloat.SetRoi(0, 0, _uncertaintyMask.Width, _uncertaintyMask.Height);
                        _uncertaintyMaskEroded.Copy(_rawImageFloat, 0);
                        _rawImageFloat.ResizeSqrPixel(imgToTrack, 2, 2, 0, 0, InterpolationMode.Linear);
                        _rawImageFloat.ResetRoi();
                        imgToTrack.Convert(temp, NppRoundMode.Near);
                        temp.SetRoi(_pefFiles[0].CropLeft, _pefFiles[0].CropTop, _pefFiles[0].CroppedWidth, _pefFiles[0].CroppedHeight);
                        temp.Copy(_imageToShow, 2);
                        temp.Copy(_imageToShow, 1);
                        temp.Copy(_imageToShow, 0);
                        OnPropertyChanged("Image");
                    }
                    break;
                case WhatToShow.CertaintyMaskGreen:
                    {
                        _uncertaintyMaskEroded.Mul(new float[] { 255, 255, 255, 255 });
                        _imageToShow.Set(new byte[] { 255, 255, 255, 255 });
                        _rawImageFloat.SetRoi(0, 0, _uncertaintyMask.Width, _uncertaintyMask.Height);
                        _uncertaintyMaskEroded.Copy(_rawImageFloat, 1);
                        _rawImageFloat.ResizeSqrPixel(imgToTrack, 2, 2, 0, 0, InterpolationMode.Linear);
                        _rawImageFloat.ResetRoi();
                        imgToTrack.Convert(temp, NppRoundMode.Near);
                        temp.SetRoi(_pefFiles[0].CropLeft, _pefFiles[0].CropTop, _pefFiles[0].CroppedWidth, _pefFiles[0].CroppedHeight);
                        temp.Copy(_imageToShow, 2);
                        temp.Copy(_imageToShow, 1);
                        temp.Copy(_imageToShow, 0);
                        OnPropertyChanged("Image");
                    }
                    break;
                case WhatToShow.CertaintyMaskBlue:
                    {
                        _uncertaintyMaskEroded.Mul(new float[] { 255, 255, 255, 255 });
                        _imageToShow.Set(new byte[] { 255, 255, 255, 255 });
                        _rawImageFloat.SetRoi(0, 0, _uncertaintyMask.Width, _uncertaintyMask.Height);
                        _uncertaintyMaskEroded.Copy(_rawImageFloat, 2);
                        _rawImageFloat.ResizeSqrPixel(imgToTrack, 2, 2, 0, 0, InterpolationMode.Linear);
                        _rawImageFloat.ResetRoi();
                        imgToTrack.Convert(temp, NppRoundMode.Near);
                        temp.SetRoi(_pefFiles[0].CropLeft, _pefFiles[0].CropTop, _pefFiles[0].CroppedWidth, _pefFiles[0].CroppedHeight);
                        temp.Copy(_imageToShow, 2);
                        temp.Copy(_imageToShow, 1);
                        temp.Copy(_imageToShow, 0);
                        OnPropertyChanged("Image");
                    }
                    break;
                case WhatToShow.WarpedImage:
                    {
                        if (_selectedItem == refImage)
                        {
                            _rawImageFloat.Copy(_decodedImage, 0);
                            _rawImageFloat.Copy(_decodedImage, 1);
                            _rawImageFloat.Copy(_decodedImage, 2);
                        }
                        else
                        {
                            imgToTrack.Copy(_decodedImage, 0);
                            imgToTrack.Copy(_decodedImage, 1);
                            imgToTrack.Copy(_decodedImage, 2);
                        }

                        float brightnessAmplifier = (float)Math.Pow(2.0, Exposure);
                        _decodedImage.Mul(new float[] { 255 * brightnessAmplifier, 255 * brightnessAmplifier, 255 * brightnessAmplifier });


                        _imageToShow.Set(new byte[] { 255, 255, 255, 255 });
                        _decodedImage.Copy(_rawImageFloat, 0);
                        _rawImageFloat.Convert(temp, NppRoundMode.Near);
                        temp.SetRoi(_pefFiles[0].CropLeft, _pefFiles[0].CropTop, _pefFiles[0].CroppedWidth, _pefFiles[0].CroppedHeight);
                        temp.Copy(_imageToShow, 2);
                        temp.ResetRoi();

                        _decodedImage.Copy(_rawImageFloat, 1);
                        _rawImageFloat.Convert(temp, NppRoundMode.Near);
                        temp.SetRoi(_pefFiles[0].CropLeft, _pefFiles[0].CropTop, _pefFiles[0].CroppedWidth, _pefFiles[0].CroppedHeight);
                        temp.Copy(_imageToShow, 1);
                        temp.ResetRoi();

                        _decodedImage.Copy(_rawImageFloat, 2);
                        _rawImageFloat.Convert(temp, NppRoundMode.Near);
                        temp.SetRoi(_pefFiles[0].CropLeft, _pefFiles[0].CropTop, _pefFiles[0].CroppedWidth, _pefFiles[0].CroppedHeight);
                        temp.Copy(_imageToShow, 0);
                        temp.ResetRoi();
                        OnPropertyChanged("Image");
                    }
                    break;
                default:
                    break;
            }

            of.FreeDeviceMemory();
            imgToTrack.Dispose();
            morph.Dispose();
            OnPropertyChanged("IsReady");
            OnPropertyChanged("IsPreAlignmentReady");
            OnPropertyChanged("IsPatchAlignmentReady");
            OnPropertyChanged("IsPrepareAccumulationReady");
            OnPropertyChanged("IsAccumulationReady");
            OnPropertyChanged("IsPostProcessingReady");

        }

        public void ShowFinalResult()
        {
            if (!IsPostProcessingReady)
                return;

            if (SuperResolution)
            {
                applyWeightingKernel.RunSafe(_decodedImage, _finalImage, _totalWeight, 0);
            }
            else
            {
                DeBayerFullRes(_preAlignmentStore.ReferenceIndex);
                applyWeightingKernel.RunSafe(_decodedImage, _finalImage, _totalWeight, 2);
            }

            ProcessColorsAndShow();
        }

        public void SaveAs16BitTiff(string aFilename)
        {
            if (!IsPostProcessingReady)
                return;

            if (SuperResolution)
            {
                applyWeightingKernel.RunSafe(_decodedImage, _finalImage, _totalWeight, 0);
            }
            else
            {
                DeBayerFullRes(_preAlignmentStore.ReferenceIndex);
                applyWeightingKernel.RunSafe(_decodedImage, _finalImage, _totalWeight, 1);
            }

            //apply color settings:
            RawFile forColor = _pefFiles[0];

            float[] whiteBalance = new float[3];
            for (int i = 0; i < 3; i++)
            {
                whiteBalance[i] = (float)_colorSpec.CameraWhite[(uint)i];
            }

            //limit to 0..1
            float[] zeros = new float[] { 0, 0, 0 };
            float[] ones = new float[] { 1, 1, 1 };
            _decodedImage.ThresholdLTGT(zeros, zeros, whiteBalance, whiteBalance);

            DNGMatrix cameraToXYZ = _colorSpec.CameraToPCS;
            DNGMatrix cameraToProPhoto = DNGColorSpace.ProPhoto.FromPCS * cameraToXYZ;

            _decodedImage.ColorTwist(cameraToProPhoto.GetAs3x4Array());

            _decodedImage.ThresholdLTGT(zeros, zeros, ones, ones);

            _LUTy.CopyToDevice(defaultLUT);
            CudaDeviceVariable<float>[] x = new CudaDeviceVariable<float>[] { _LUTx, _LUTx, _LUTx };
            CudaDeviceVariable<float>[] y = new CudaDeviceVariable<float>[] { _LUTy, _LUTy, _LUTy };
            _decodedImage.LUTCubic(y, x);

            DNGMatrix proPhotoTosRGB = DNGColorSpace.sRGB50.FromPCS * DNGColorSpace.ProPhoto.ToPCS;
            _decodedImage.ColorTwist(proPhotoTosRGB.GetAs3x4Array());

            float brightnessAmplifier = (float)Math.Pow(2.0, Exposure);
            _decodedImage.Mul(new float[] { brightnessAmplifier, brightnessAmplifier, brightnessAmplifier });
            gammaCorrectionKernel.RunSafe(_decodedImage);

            _decodedImage.Mul(new float[] { 65535, 65535, 65535 });

            _decodedImage.SetRoi(forColor.CropLeft, forColor.CropTop, forColor.CroppedWidth, forColor.CroppedHeight);

            NPPImage_16uC3 result = new NPPImage_16uC3(forColor.CroppedWidth, forColor.CroppedHeight);
            _decodedImage.Convert(result, NppRoundMode.Near);

            ushort[] result_host = new ushort[forColor.CroppedWidth * forColor.CroppedHeight * 3];

            result.CopyToHost(result_host);

            ImageFileDirectory ifd = new ImageFileDirectory((ushort)forColor.CroppedWidth, (ushort)forColor.CroppedHeight);
            ifd.SaveAsTiff(aFilename, result_host);
            result.Dispose();
        }
        #endregion

        #region private Methods
        public void ProcessColorsAndShow()
        {
            RawFile forColor = _pefFiles[0];

            float[] whiteBalance = new float[3];
            for (int i = 0; i < 3; i++)
            {
                whiteBalance[i] = (float)_colorSpec.CameraWhite[(uint)i];
            }
            float minWhiteScale = whiteBalance.Min();

            //limit to 0..1
            float[] zeros = new float[] { 0, 0, 0 };
            float[] ones = new float[] { 1, 1, 1 };
            _decodedImage.ThresholdLTGT(zeros, zeros, whiteBalance, whiteBalance);

            DNGMatrix cameraToXYZ = _colorSpec.CameraToPCS;

            DNGMatrix cameraToProPhoto = DNGColorSpace.ProPhoto.FromPCS * cameraToXYZ;

            _decodedImage.ColorTwist(cameraToProPhoto.GetAs3x4Array());

            _decodedImage.ThresholdLTGT(zeros, zeros, ones, ones);
            _LUTy.CopyToDevice(defaultLUT);
            CudaDeviceVariable<float>[] x = new CudaDeviceVariable<float>[] { _LUTx, _LUTx, _LUTx };
            CudaDeviceVariable<float>[] y = new CudaDeviceVariable<float>[] { _LUTy, _LUTy, _LUTy };
            _decodedImage.LUTCubic(y, x);

            DNGMatrix proPhotoTosRGB = DNGColorSpace.sRGB50.FromPCS * DNGColorSpace.ProPhoto.ToPCS;
            _decodedImage.ColorTwist(proPhotoTosRGB.GetAs3x4Array());

            float brightnessAmplifier = (float)Math.Pow(2.0, Exposure);
            _decodedImage.Mul(new float[] { brightnessAmplifier, brightnessAmplifier, brightnessAmplifier });
            gammaCorrectionKernel.RunSafe(_decodedImage);
            _decodedImage.Mul(new float[] { 255, 255, 255 });
            
            _decodedImage.SetRoi(forColor.CropLeft, forColor.CropTop, forColor.CroppedWidth, forColor.CroppedHeight);
            _rawImageFloat.SetRoi(forColor.CropLeft, forColor.CropTop, forColor.CroppedWidth, forColor.CroppedHeight);

            _imageToShow.Set(new byte[] { 255, 255, 255, 255 });
            NPPImage_8uC1 temp = new NPPImage_8uC1(_rawImage.DevicePointer, forColor.CroppedWidth, forColor.CroppedHeight, forColor.CroppedWidth);


            _decodedImage.Copy(_rawImageFloat, 0);

            _rawImageFloat.Convert(temp, NppRoundMode.Near);
            temp.Copy(_imageToShow, 2);

            _decodedImage.Copy(_rawImageFloat, 1);
            _rawImageFloat.Convert(temp, NppRoundMode.Near);
            temp.Copy(_imageToShow, 1);

            _decodedImage.Copy(_rawImageFloat, 2);
            _rawImageFloat.Convert(temp, NppRoundMode.Near);
            temp.Copy(_imageToShow, 0);
            _decodedImage.ResetRoi();
            _rawImageFloat.ResetRoi();
            OnPropertyChanged("Image");
        }
        private void AllocateDeviceMemory()
        {
            if (_ctx == null)
                return;

            if (_pefFiles == null)
                return;

            if (_pefFiles.Count < 1)
                return;

            int width = _pefFiles[0].RawWidth;
            int height = _pefFiles[0].RawHeight;

            _imageToShow?.Dispose();
            _rawImage?.Dispose();
            _rawImageFloat?.Dispose();
            _decodedImage?.Dispose();

            _imageToShow = new NPPImage_8uC4(_pefFiles[0].CroppedWidth, _pefFiles[0].CroppedHeight);
            _rawImage = new NPPImage_16uC1(width, height);
            _rawImageNoPitch = new CudaDeviceVariable<ushort>(_rawImage.DevicePointer, sizeof(ushort) * width * height);
            _rawImageFloat = new NPPImage_32fC1(width, height);
            _decodedImage = new NPPImage_32fC3(width, height);
            _LUTx = LUTx;
            _LUTy = defaultLUT;
        }

        private float[] gaussian_filter_1D(float sigma)
        {
            if (sigma <= 0)
            {
                return new float[] { 0, 0, 0, 0, 1, 0, 0, 0, 0 };
            }

            int size = (int)(sigma / 0.6f - 0.4f) * 2 + 1 + 2;
            size = Math.Min(size, 99); //100 seems to be NPP limit

            float[] ret = new float[size];
            int center = size / 2;

            for (int i = 0; i < size; i++)
            {
                int x = i - center;
                ret[i] = (float)(Math.Exp(-(x * x) / (2 * sigma * sigma)));
            }

            float sum = 0;
            for (int i = 0; i < size; i++)
            {
                sum += ret[i];
            }

            for (int i = 0; i < size; i++)
            {
                ret[i] /= sum;
            }

            return ret;
        }

        private void DeBayerBWGaussWB(int index, CudaDeviceVariable<float> gaussFilter, bool skipFourierFilter = false)
        {
            if (index < 0 || index >= _pefFiles.Count)
                return;

            if (_isTestMode)
            {
                _rawImage.CopyToDevice(_pefFiles[index].RawImage);
                _rawImage.Convert(_rawImageFloat);
                _rawImageFloat.Div(65535);
                _rawImageFloat.CopyToHost(_debayerdImagesBW[index]);
            }
            else
            {
                DeBayerFullRes(index);

                gammaCorrectionKernel.RunSafe(_decodedImage);
            
                NPPImage_32fC1 temp = new NPPImage_32fC1(_decodedImage.DevicePointer, _decodedImage.Width, _decodedImage.Height, _rawImageFloat.Pitch);

                float[] colorToGray = new float[] { 0.2989f, 0.5870f, 0.1141f };
                if (GreenChannelOnly)
                    colorToGray = new float[] { 0, 1, 0 };

                _decodedImage.ColorToGray(_rawImageFloat, colorToGray);
                if (!skipFourierFilter)
                {
                    _preAlignment.FourierFilter(_rawImageFloat, ClearAxis, HighPass, HighPassSigma);
                }

                _rawImageFloat.FilterGaussBorder(temp, gaussFilter, NppiBorderType.Replicate);

                temp.CopyToHost(_debayerdImagesBW[index]);
            }
        }
        #endregion
    }
}
