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
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using Microsoft.Win32;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using ManagedCuda.NPP;
using System.Drawing;

namespace PEFStudioDX
{
    /// <summary>
    /// Interaktionslogik für MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        ImageStackAlignatorController myController;
        WriteableBitmap kernelImage = new WriteableBitmap(5, 5, 1, 1, PixelFormats.Gray8, null);
        public MainWindow()
        {
            InitializeComponent();

        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            

            OpenFileDialog ofd = new OpenFileDialog();
            ofd.Filter = "Supported RAW Files (DNG and Pentax PEF)|*.dng;*.pef|Pentax RAW Files|*.pef|DNG RAW-Files|*.dng";
            ofd.Multiselect = true;
            
            if (ofd.ShowDialog() == true)
            {
                myController.FileNameList.Clear();

                if (ofd.FileNames.Length > 0)
                {
                    myController.BaseDirectory = System.IO.Path.GetDirectoryName(ofd.FileNames[0]);

                    foreach (var item in ofd.FileNames)
                    {
                        string filename = System.IO.Path.GetFileName(item);
                        myController.FileNameList.Add(filename);
                    }
                }

                Mouse.OverrideCursor = Cursors.Wait;
                try
                {
                    myController.ReadFiles();
                }
                finally
                {
                    Mouse.OverrideCursor = null;
                }
            }


        }

        private void Button_Click_1(object sender, RoutedEventArgs e)
        {
            imagePresenter.SetTiling(94, 62, 64, 1, 0);
        }

        private void ListView_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            ListView ls = sender as ListView;
            myController?.SelectedItemsChanged(ls.SelectedItems);   
        }

        private void TabControl_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            TabControl tc = sender as TabControl;
            ImageStackAlignatorController.WorkState state;
            if (tc.SelectedIndex < 0)
                state = ImageStackAlignatorController.WorkState.Init;
            else
                state = (ImageStackAlignatorController.WorkState)tc.SelectedIndex;

            if (state != ImageStackAlignatorController.WorkState.PatchAlign)
            {
                imagePresenter.ResetTiling();
            }
            myController?.WorkStateChanged(state);
        }

        private void Window_Loaded(object sender, RoutedEventArgs e)
        {

            myController = this.Resources["myController"] as ImageStackAlignatorController;
            System.ComponentModel.DependencyPropertyDescriptor pixelChanged = System.ComponentModel.DependencyPropertyDescriptor.FromProperty
            (ImagePresenterDX.PixelCoordinateProperty, typeof(ImagePresenterDX));
            pixelChanged.AddValueChanged(this.imagePresenter, new EventHandler(this.PixelChangedHandler));
        }

        private void ImageColorChanged(object sender, EventArgs e)
        {
            if (fileList.SelectedIndex < 0)
                return;

            myController.DeBayerColorVisu(fileList.SelectedIndex);
        }

        private void PixelChangedHandler(object sender, EventArgs e)
        {
            System.Windows.Point p = (sender as ImagePresenterDX).PixelCoordinate;
            if (double.IsInfinity(p.X) || double.IsInfinity(p.Y))
                return;

            byte[] pixels = new byte[25];
            int x = (int)p.X;
            int y = (int)p.Y;
            
            float3 kernel = myController.GetKernel(x, y);

            for (int py = -2; py <= 2; py++)
            {
                for (int px = -2; px <= 2; px++)
                {
                    double w = px * px * kernel.x + 2 * px * py * kernel.z + py * py * kernel.y;
                    w = Math.Exp(-0.5 * w);
                    if (double.IsNaN(w))
                        w = 0;

                    byte val = (byte)(w * 255);

                    int i = (py + 2) * 5 + px + 2;
                    pixels[i] = val;
                }
            }
            kernelImage.WritePixels(new Int32Rect(0, 0, 5, 5), pixels, 5, 0);
            KernelImageViewer.Source = kernelImage;
        }

        private void Window_ContentRendered(object sender, EventArgs e)
        {
            //at this state imagePresenter initialized a cuda context with directX binding, so we use that one.
            myController.InitCuda(imagePresenter.CudaContext);

            myController.PropertyChanged += MyController_PropertyChanged;
        }

        private void MyController_PropertyChanged(object sender, System.ComponentModel.PropertyChangedEventArgs e)
        {
            if (e.PropertyName == "Image")
            {
                float2 preShift = myController.PreAlignmentShift;
                float preRotation = myController.PreAlignmentRotation;

                imagePresenter.SetPreAlignment(preShift.x, preShift.y, preRotation);
                imagePresenter.SetImage(myController.Image, myController.Orientation);
            }
        }

        private void Window_SourceInitialized(object sender, EventArgs e)
        {

        }

        private void TestPreAlignment_Click(object sender, RoutedEventArgs e)
        {
            if (fileList.SelectedIndex < 0)
                return;

            myController.DeBayerBWGaussWBVisu(fileList.SelectedIndex);
        }

        private void ComputePreAlignment_Click(object sender, RoutedEventArgs e)
        {
            Mouse.OverrideCursor = Cursors.Wait;
            try
            {
                myController.ComputePreAlignment();
            }
            finally
            {
                Mouse.OverrideCursor = null;
            }
        }

        private void ResetPreAlignment_Click(object sender, RoutedEventArgs e)
        {
            Mouse.OverrideCursor = Cursors.Wait;
            try
            {
                myController.SkipPreAlignment();
            }
            finally
            {
                Mouse.OverrideCursor = null;
            }
        }

        private void AddPatchTrackingLevel_Click(object sender, RoutedEventArgs e)
        {
            myController.AddPatchTrackingLevel();
        }

        private void RemovePatchTrackingLevel_Click(object sender, RoutedEventArgs e)
        {
            myController.RemovePatchTrackingLevel();
        }

        private void ButtonTrackPatches_Click(object sender, RoutedEventArgs e)
        {
            Mouse.OverrideCursor = Cursors.Wait;
            try
            {
                myController.TrackPatches();
            }
            finally
            {
                Mouse.OverrideCursor = null;
            }
        }

        private void ShowTrackedPatches_Click(object sender, RoutedEventArgs e)
        {
            if (fileList.SelectedIndex < 0)
                return;
            float2[] flow = myController.GetTrackedPatchFlow(fileList.SelectedIndex);

            imagePresenter.SetTiling(myController.TileCountX, myController.TileCountY, myController.TileSize, myController.ResizeLevel, myController.MaxShift);
            imagePresenter.SetTileShifts(myController.TileCountX, myController.TileCountY, myController.TileSize, myController.ResizeLevel, myController.MaxShift, flow);
        }

        private void PrepareAccumulationBtn_Click(object sender, RoutedEventArgs e)
        {
            Mouse.OverrideCursor = Cursors.Wait;
            try
            {
                myController.PrepareAccumulation();
            }
            finally
            {
                Mouse.OverrideCursor = null;
            }
        }

        private void AccumulationBtn_Click(object sender, RoutedEventArgs e)
        {
            Mouse.OverrideCursor = Cursors.Wait;
            try
            {
                myController.Accumulate();
            }
            finally
            {
                Mouse.OverrideCursor = null;
            }
        }

        private void TestCrossCorrelation_Click(object sender, RoutedEventArgs e)
        {
            myController.TestCC();
        }

        private void myLUT_LUTChangedEvent()
        {
            myLUT.EvaluateLUT(myController.LUTx, myController.defaultLUT);
        }

        private void SaveResultBtn_Click(object sender, RoutedEventArgs e)
        {
            SaveFileDialog sfd = new SaveFileDialog();
            sfd.Filter = "TIF files|*.tif";

            if (sfd.ShowDialog() == true)
            {
                Mouse.OverrideCursor = Cursors.Wait;
                try
                {
                    myController.SaveAs16BitTiff(sfd.FileName);
                }
                finally
                {
                    Mouse.OverrideCursor = null;
                }
                
            }
        }

        private void ShowResultBtn_Click(object sender, RoutedEventArgs e)
        {
            myController.ShowFinalResult();
        }
    }
    public class MultiplyConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            double v = System.Convert.ToDouble(value, System.Globalization.CultureInfo.InvariantCulture);
            double p = System.Convert.ToDouble(parameter, System.Globalization.CultureInfo.InvariantCulture);
            return v * p;
        }

        public object ConvertBack(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {

            double v = System.Convert.ToDouble(value, System.Globalization.CultureInfo.InvariantCulture);
            double p = System.Convert.ToDouble(parameter, System.Globalization.CultureInfo.InvariantCulture);
            return v / p;
        }
    }
    public class UnevenSliderConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            double v = System.Convert.ToDouble(value, System.Globalization.CultureInfo.InvariantCulture);
            return Math.Floor(v / 2.0);
        }

        public object ConvertBack(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {

            double v = System.Convert.ToDouble(value, System.Globalization.CultureInfo.InvariantCulture);
            v = Math.Round(v);
            return v * 2 + 1;
        }
    }
    public class TintSliderConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            double v = System.Convert.ToDouble(value, System.Globalization.CultureInfo.InvariantCulture);
            v = Math.Min(Math.Max(v, -150.0), 150.0);
            return Math.Floor(v * 100.0);
        }

        public object ConvertBack(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {

            double v = System.Convert.ToDouble(value, System.Globalization.CultureInfo.InvariantCulture);
            v = v / 100.0;
            return v;
        }
    }
    public class ColorTempConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            double v = System.Convert.ToDouble(value, System.Globalization.CultureInfo.InvariantCulture);
            v = v - 1500;
            v = Math.Log(v) * 1000 - 6214;
            return Math.Round(v);
        }

        public object ConvertBack(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {

            double v = System.Convert.ToDouble(value, System.Globalization.CultureInfo.InvariantCulture);
            v += 6214;
            v = v / 1000.0;
            v = Math.Exp(v);
            v = v + 1500;
            v = Math.Round(v);
            v = Math.Min(Math.Max(v, 2000.0), 50000.0);
            return v;
        }
    }
    public class DoubleToTextConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            double v = System.Convert.ToDouble(value, System.Globalization.CultureInfo.InvariantCulture);
            return v.ToString(System.Globalization.CultureInfo.InvariantCulture);
        }

        public object ConvertBack(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {

            double v = System.Convert.ToDouble(value, System.Globalization.CultureInfo.InvariantCulture);
            return v;
        }
    }

}
