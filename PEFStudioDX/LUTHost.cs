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
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;

namespace PEFStudioDX
{
    public class LUTHost : FrameworkElement
    {
        private readonly VisualCollection _children;
        Nullable<Point> dragStart = null;

        public LUTHost()
        {
            _children = new VisualCollection(this);
            MouseLeftButtonUp += LUTHost_MouseLeftButtonUp;
            MouseLeftButtonDown += LUTHost_MouseLeftButtonDown;
            MouseDown += LUTHost_MouseDown;
            MouseLeave += LUTHost_MouseLeave;
            MouseMove += LUTHost_MouseMove;
        }

        private void LUTHost_MouseDown(object sender, MouseButtonEventArgs e)
        {
            if (e.ChangedButton == MouseButton.Middle && e.ButtonState == MouseButtonState.Pressed)
            {
                // Retreive the coordinates of the mouse button event.
                Point pt = e.GetPosition((UIElement)sender);

                PointCreateOrDeleteEvent?.Invoke(pt.X / ActualWidth, (ActualHeight - pt.Y) / ActualHeight);
            }
        }

        public delegate void PointSelectedEventHandler(double x, double y);
        public event PointSelectedEventHandler PointSelectedEvent;

        public delegate void PointMovedEventHandler(double x, double y);
        public event PointMovedEventHandler PointMovedEvent;

        public delegate void PointCreateOrDeleteEventHandler(double x, double y);
        public event PointCreateOrDeleteEventHandler PointCreateOrDeleteEvent;

        public delegate void PointReleasedEventHandler();
        public event PointReleasedEventHandler PointReleasedEvent;

        private void LUTHost_MouseMove(object sender, MouseEventArgs e)
        {
            if (dragStart != null && e.LeftButton == MouseButtonState.Pressed)
            {
                var element = (UIElement)sender;
                var p2 = e.GetPosition(this);
                PointMovedEvent?.Invoke(p2.X / ActualWidth, (ActualHeight - p2.Y) / ActualHeight);
            }
        }

        private void LUTHost_MouseLeave(object sender, MouseEventArgs e)
        {
            PointReleasedEvent?.Invoke();
            dragStart = null;
            this.ReleaseMouseCapture();
        }

        private void LUTHost_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            // Retreive the coordinates of the mouse button event.
            Point pt = e.GetPosition((UIElement)sender);

            dragStart = pt;
            PointSelectedEvent?.Invoke(pt.X / ActualWidth, (ActualHeight - pt.Y) / ActualHeight);
            this.CaptureMouse();
        }

        private void LUTHost_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            dragStart = null;
            PointReleasedEvent?.Invoke();
            this.ReleaseMouseCapture();
        }

        public HitTestResultBehavior MyCallback(HitTestResult result)
        {
            PointHitTestResult pointHit = result as PointHitTestResult;
            if (pointHit != null)
            {
                Console.WriteLine("Hit: " + pointHit.PointHit.X / ActualWidth + "; " + (ActualHeight - pointHit.PointHit.Y) / ActualHeight);
            }
            if (result.VisualHit.GetType() == typeof(System.Windows.Media.DrawingVisual))
            {
                if (pointHit != null)
                {
                    dragStart = pointHit.PointHit;                    
                }
                this.CaptureMouse();
            }

            return HitTestResultBehavior.Stop;
        }

        // Provide a required override for the VisualChildrenCount property.
        protected override int VisualChildrenCount => _children.Count;

        // Provide a required override for the GetVisualChild method.
        protected override Visual GetVisualChild(int index)
        {
            if (index < 0 || index >= _children.Count)
            {
                throw new ArgumentOutOfRangeException();
            }
            return _children[index];
        }

        public void DrawPoints(SortedList<double, double> points, Point[] line)
        {
            _children.Clear();
            
            double width = ActualWidth;
            double height = ActualHeight;


            foreach (var point in line)
            {
                System.Windows.Media.DrawingVisual drawingVisual = new System.Windows.Media.DrawingVisual();
                DrawingContext drawingContext = drawingVisual.RenderOpen();
                double x = point.X * width;
                double y = height - point.Y * height;

                drawingContext.DrawEllipse(Brushes.Blue, null, new Point(x, y), 1, 1);
                drawingContext.Close();
                _children.Add(drawingVisual);
            }
            

            foreach (var point in points)
            {
                System.Windows.Media.DrawingVisual drawingVisual = new System.Windows.Media.DrawingVisual();
                DrawingContext drawingContext = drawingVisual.RenderOpen();
                Pen pen = new Pen(Brushes.AntiqueWhite, 3);
                drawingContext.DrawEllipse(Brushes.Transparent, pen, new Point(point.Key * width, height - point.Value * height), 3, 3);
                drawingContext.Close();
                _children.Add(drawingVisual);
            }
        }
    }
}
