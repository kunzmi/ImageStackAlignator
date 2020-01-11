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
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace PEFStudioDX
{
    /// <summary>
    /// Interaktionslogik für LUTControl.xaml
    /// </summary>
    public partial class LUTControl : UserControl
    {
        SortedList<double, double> _controlPoints;
        double[] S;
        double[] X;
        double[] Y;
        bool _needsSolve = true;
        LUTHost _host;
        int _selectedPoint = -1;

        public delegate void LUTChangedEventHandler();
        public event LUTChangedEventHandler LUTChangedEvent;


        public LUTControl()
        {
            _controlPoints = new SortedList<double, double>();


            _controlPoints.Add(0, 0);
            _controlPoints.Add(0.012797142375453, 0.0105369636685493);
            _controlPoints.Add(0.0212267133102924, 0.0231813200708084);
            _controlPoints.Add(0.0707504425524739, 0.123282474922026);
            _controlPoints.Add(0.12975743909635, 0.271853662648571);
            _controlPoints.Add(0.237234468515552, 0.499452077889235);
            _controlPoints.Add(0.323637570597656, 0.629056731012391);
            _controlPoints.Add(0.452188527353958, 0.766037258703532);
            _controlPoints.Add(0.629209516985585, 0.887212340891849);
            _controlPoints.Add(0.816767470285762, 0.962024782938548);
            _controlPoints.Add(0.909492750568996, 0.985206103009357);
            _controlPoints.Add(1, 1);


            InitializeComponent();
        }

        private void MyGrid_SizeChanged(object sender, SizeChangedEventArgs e)
        {
            double minSize = Math.Min(e.NewSize.Width, e.NewSize.Height);
            
            myGrid.Width = minSize;
            myGrid.Height = minSize;            
        }

        private void UserControl_Loaded(object sender, RoutedEventArgs e)
        {
            myGrid.SizeChanged += MyGrid_SizeChanged;
            myGrid.Children.Clear();
            _host = new LUTHost();
            _host.PointSelectedEvent += _host_PointSelectedEvent;
            _host.PointMovedEvent += _host_PointMovedEvent;
            _host.PointReleasedEvent += _host_PointReleasedEvent;
            _host.PointCreateOrDeleteEvent += _host_PointCreateOrDeleteEvent;


            myGrid.Children.Add(_host);     
            
            //needed for some reasons...
            _host.InvalidateArrange();
            _host.UpdateLayout();

            DrawPoints();
            LUTChangedEvent?.Invoke();
        }

        private void _host_PointCreateOrDeleteEvent(double x, double y)
        {
            int hit = checkForPoint(x, y);

            if (hit > 0 && hit < _controlPoints.Count - 1)
            {
                _controlPoints.RemoveAt(hit);
                _selectedPoint = -1;
                _needsSolve = true;
                //Console.WriteLine("Removed point");
                DrawPoints();
                LUTChangedEvent?.Invoke();

            }
            //else
            //{
            //    if (x > 0 && x < 1 && y >= 0 && y <= 1)
            //    {
            //        _controlPoints.Add(x, y);
            //        Console.WriteLine("Added point");
            //        _host.DrawPoints(_controlPoints);
            //    }
            //}
        }

        private void _host_PointReleasedEvent()
        {
            //Console.WriteLine("RELEASE!");
            _selectedPoint = -1;
            LUTChangedEvent?.Invoke();
        }

        private void _host_PointMovedEvent(double x, double y)
        {
            //Console.WriteLine("moved " + _selectedPoint +  " to " + x + ", " + y);
            if (_selectedPoint >= 0)
            {
                if (_selectedPoint == 0 || _selectedPoint == _controlPoints.Count - 1)
                {
                    //can't move these...
                }
                else
                {
                    if (x > _controlPoints.ElementAt(_selectedPoint - 1).Key
                        && x < _controlPoints.ElementAt(_selectedPoint + 1).Key
                        && y >= 0 && y <= 1)
                    {
                        _controlPoints.RemoveAt(_selectedPoint);
                        _controlPoints.Add(x, y);
                        _needsSolve = true;
                    }
                }
                DrawPoints();
            }
        }

        private void _host_PointSelectedEvent(double x, double y)
        {
            _selectedPoint = checkForPoint(x, y);

            if (_selectedPoint >= 0)
            { 
                
            }
        }

        private int checkForPoint(double x, double y)
        {
            int closestIndex = -1;
            double minDist = double.MaxValue;
            for (int p = 0; p < _controlPoints.Count; p++)
            {
                KeyValuePair<double, double> point = _controlPoints.ElementAt(p);
                double dist = Math.Sqrt((x - point.Key) * (x - point.Key) + (y - point.Value) * (y - point.Value));
                if (dist < minDist)
                {
                    minDist = dist;
                    closestIndex = p;
                }
            }

            if (minDist < 5.0 / ActualWidth)
            {
                //Console.WriteLine("Selected point: " + _selectedPoint);
                return closestIndex;
            }
            return -1;
        }

        private void UserControl_SizeChanged(object sender, SizeChangedEventArgs e)
        {
            double minSize = Math.Min(Width, Height);
            myGrid.Width = minSize;
            myGrid.Height = minSize;
            DrawPoints();
        }

        private void UserControl_MouseLeave(object sender, MouseEventArgs e)
        {
            //_host.RaiseEvent(e);
        }

        private void myGrid_MouseDown(object sender, MouseButtonEventArgs e)
        {
            if (e.ChangedButton == MouseButton.Right && e.ButtonState == MouseButtonState.Pressed)
            {
                // Retreive the coordinates of the mouse button event.
                Point pt = e.GetPosition((UIElement)_host);
                double posX = pt.X / _host.ActualWidth;
                double posY = (_host.ActualHeight - pt.Y) / _host.ActualHeight;
                int hit = checkForPoint(posX, posY);

                if (posX > 0 && posX < 1 && posY >= 0 && posY <= 1)
                {
                    if (_controlPoints.ContainsKey(posX))
                    {
                        return;
                    }
                    _controlPoints.Add(posX, posY);
                    _needsSolve = true;
                    //Console.WriteLine("Added point");
                    DrawPoints();
                    LUTChangedEvent?.Invoke();
                }
            }
        }

        private void Solve()
        {
            int count = _controlPoints.Count;
            
            int start = 0;
            int end = count;


            S = new double[_controlPoints.Count];
            X = new double[count];
            Y = new double[count];

            for (int i = 0; i < count; i++)
            {
                X[i] = _controlPoints.ElementAt(i).Key;
                Y[i] = _controlPoints.ElementAt(i).Value;
            }

            double A = X[start + 1] - X[start];
            double B = (Y[start + 1] - Y[start]) / A;
            
            S[start] = B;
            int j;

            // Slopes here are a weighted average of the slopes
            // to each of the adjcent control points.
            for (j = start + 2; j < end; ++j)
            {
                double C = X[j] - X[j - 1];
                double D = (Y[j] - Y[j - 1]) / C;

                S[j - 1] = (B * C + D * A) / (A + C);
                A = C;
                B = D;
            }

            S[end - 1] = 2.0 * B - S[end - 2];
            S[start] = 2.0 * S[start] - S[start + 1];

            if ((end - start) > 2)
            {
                double[] E = new double[count];
                double[] F = new double[count];
                double[] G = new double[count];

                F[start] = 0.5;
                E[end - 1] = 0.5;
                G[start] = 0.75 * (S[start] + S[start + 1]);
                G[end - 1] = 0.75 * (S[end - 2] + S[end - 1]);

                for (j = start + 1; j < end - 1; ++j)
                {
                    A = (X[j + 1] - X[j - 1]) * 2.0;
                    E[j] = (X[j + 1] - X[j]) / A;
                    F[j] = (X[j] - X[j - 1]) / A;
                    G[j] = 1.5 * S[j];
                }

                for (j = start + 1; j < end; ++j)
                {
                    A = 1.0 - F[j - 1] * E[j];
                    if (j != end - 1) F[j] /= A;
                    G[j] = (G[j] - G[j - 1] * E[j]) / A;
                }

                for (j = end - 2; j >= start; --j)
                    G[j] = G[j] - F[j] * G[j + 1];

                for (j = start; j < end; ++j)
                    S[j] = G[j];

            }

            _needsSolve = false;
        }

        private double EvaluateSplineSegment(double x,
                                     double x0,
                                     double y0,
                                     double s0,
                                     double x1,
                                     double y1,
                                     double s1)
        {
            double A = x1 - x0;
            double B = (x - x0) / A;
            double C = (x1 - x) / A;
            double D = ((y0 * (2.0 - C + B) + (s0 * A * B)) * (C * C)) +
                       ((y1 * (2.0 - B + C) - (s1 * A * C)) * (B * B));
            return Math.Min(Math.Max(0, D), 1);
        }


        public double Evaluate(double x)
        {
            int count = X.Length;

            // Check for off each end of point list.
            if (x <= X[0])
                return Y[0];

            if (x >= X[count - 1])
                return Y[count - 1];

            // Binary search for the index.

            int lower = 1;
            int upper = count - 1;

            while (upper > lower)
            {
                int mid = (lower + upper) >> 1;

                if (x == X[mid])
                {
                    return Y[mid];
                }

                if (x > X[mid])
                    lower = mid + 1;
                else
                    upper = mid;
            }
            
            int j = lower;

            // X [j - 1] < x <= X [j]
            // A is the distance between the X [j] and X [j - 1]
            // B and C describe the fractional distance to either side. B + C = 1.

            // We compute a cubic spline between the two points with slopes
            // S[j-1] and S[j] at either end. Specifically, we compute the 1-D Bezier
            // with control values:
            //
            //		Y[j-1], Y[j-1] + S[j-1]*A, Y[j]-S[j]*A, Y[j]

            return EvaluateSplineSegment(x,
                                        X[j - 1],
                                        Y[j - 1],
                                        S[j - 1],
                                        X[j],
                                        Y[j],
                                        S[j]);
        }

        private void DrawPoints()
        {
            if (_needsSolve)
            {
                Solve();
            }

            int pointCount = (int)_host.ActualWidth;

            Point[] points = new Point[pointCount];

            for (int i = 0; i < pointCount; i++)
            {
                double x = (double)i / ((double)pointCount - 1);
                double y = Evaluate(x);
                points[i] = new Point(x, y);
            }

            _host.DrawPoints(_controlPoints, points);
        }

        public void EvaluateLUT(float[] x, float[] y)
        {
            if (_needsSolve)
            {
                Solve();
            }

            for (int i = 0; i < x.Length; i++)
            {
                y[i] = (float)Evaluate(x[i]);
            }
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            Console.WriteLine("");
            Console.WriteLine("dumped points: ");
            foreach (var item in _controlPoints)
            {
                Console.WriteLine(item.Key + ", " + item.Value);
            }
            Console.WriteLine("");
        }
    }
}
