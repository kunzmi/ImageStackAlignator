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
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Runtime.InteropServices;
using System.Windows.Interop;
using SlimDX;
using SlimDX.Direct3D9;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;

namespace PEFStudioDX
{
    /// <summary>
    /// Interaktionslogik für ImagePresenterDX.xaml
    /// </summary>
    public partial class ImagePresenterDX : UserControl
    {
        #region Members

        [StructLayout(LayoutKind.Sequential)]
        struct vertex
        {
            public float x;
            public float y;
            public float z;
            public float u;
            public float v;


            public vertex(float aX, float aY, float aZ, float aU, float aV)
            {
                x = aX;
                y = aY;
                z = aZ;
                u = aU;
                v = aV;
            }
        }

        [StructLayout(LayoutKind.Sequential)]
        struct linevertex
        {
            public float x, y, z;
            public uint color;

            public linevertex(float aX, float aY, float aZ, uint aColor)
            {
                x = aX;
                y = aY;
                z = aZ;
                color = aColor;
            }
        }

        public enum Rotation
        {
            _0,
            _90,
            _180,
            _270,
        }

        Direct3DEx _d3d = null;
        bool _deviceFound;
        bool _initOK = false;
        Rotation _rotation = Rotation._0;
        Device _device;
        SwapChain _swapChain;

        Texture _texture;
        VertexBuffer _vertexBufferTexture;
        VertexBuffer _vertexBufferTileGrid;
        VertexBuffer _vertexBufferFlow;
        int _tileCount;
        CudaContext _ctx;
        CudaGraphicsInteropResourceCollection _graphicsres;

        float _scaleFac = 1.0f;
        float _projFac = 1.0f;
        bool _sizeOnHeight = false;
        bool _clicked = false;
        int _lastX = 0, _lastY = 0; // Mouse positions
        float _viewShiftX = 0;
        float _viewShiftY = 0;

        int _imageWidth = 512;
        int _imageHeight = 512;
        int _realImageWidth = 512;
        int _realImageHeight = 512;
        
        Color4 _backgroundColor = new Color4(0.247f, 0.247f, 0.247f);
        #endregion

        #region Public Methods
        public ImagePresenterDX()
        {
            InitializeComponent();
            _graphicsres = new CudaGraphicsInteropResourceCollection();
        }


        #endregion

        #region Private Methods

        private bool InitializeD3D()
        {
            HwndSource hwnd = new HwndSource(0, 0, 0, 0, 0, "null", IntPtr.Zero);
            // Create the D3D object.
            _d3d = new Direct3DEx();


            PresentParameters pp = new PresentParameters();
            pp.BackBufferWidth = 1;
            pp.BackBufferHeight = 1;
            pp.BackBufferFormat = Format.Unknown;
            pp.BackBufferCount = 0;
            pp.Multisample = MultisampleType.None;
            pp.MultisampleQuality = 0;
            pp.SwapEffect = SwapEffect.Discard;
            pp.DeviceWindowHandle = (IntPtr)0;
            pp.Windowed = true;
            pp.EnableAutoDepthStencil = false;
            pp.AutoDepthStencilFormat = Format.Unknown;
            pp.PresentationInterval = PresentInterval.Default;

            PresentParameters pp2 = new PresentParameters();
            pp2.BackBufferWidth = (int)ActualWidthDpi;
            pp2.BackBufferHeight = (int)ActualHeightDpi;
            pp2.BackBufferFormat = Format.Unknown;
            pp2.BackBufferCount = 0;
            pp2.Multisample = MultisampleType.None;
            pp2.MultisampleQuality = 0;
            pp2.SwapEffect = SwapEffect.Discard;
            pp2.DeviceWindowHandle = (IntPtr)0;
            pp2.Windowed = true;
            pp2.EnableAutoDepthStencil = false;
            pp2.AutoDepthStencilFormat = Format.Unknown;
            pp2.PresentationInterval = PresentInterval.Default;

            _deviceFound = false;
            CUdevice[] cudaDevices = null;
            int g_iAdapter;
            for (g_iAdapter = 0; g_iAdapter < _d3d.AdapterCount; g_iAdapter++)
            {
                _device = new Device(_d3d, _d3d.Adapters[g_iAdapter].Adapter, DeviceType.Hardware, hwnd.Handle, CreateFlags.HardwareVertexProcessing | CreateFlags.Multithreaded, pp);
                try
                {
                    cudaDevices = CudaContext.GetDirectXDevices(_device.ComPointer, CUd3dXDeviceList.All, CudaContext.DirectXVersion.D3D9);
                    _deviceFound = cudaDevices.Length > 0;

                    break;
                }
                catch (CudaException ex)
                {
                    MessageBox.Show(ex.Message);
                    //No Cuda device found for this Direct3D9 device

                }
            }

            // we check to make sure we have found a cuda-compatible D3D device to work on  
            if (!_deviceFound)
            {
                if (_device != null)
                    _device.Dispose();
                return false;
            }

            _swapChain = new SwapChain(_device, pp2);

            _ctx = new CudaContext(cudaDevices[0], _device.ComPointer, CUCtxFlags.BlockingSync, CudaContext.DirectXVersion.D3D9);

            // Set projection matrix
            SlimDX.Matrix matProj = SlimDX.Matrix.OrthoOffCenterLH(-0.5f, 0.5f, 0.5f, -0.5f, 0, 1);
            _device.SetTransform(TransformState.Projection, matProj);

            // Turn off D3D lighting, since we are providing our own vertex colors
            _device.SetRenderState(RenderState.Lighting, false);
            _device.SetRenderState(RenderState.DiffuseMaterialSource, ColorSource.Color1);

            d3dimage.Lock();
            Surface surf = _swapChain.GetBackBuffer(0);
            _device.SetRenderTarget(0, surf);
            d3dimage.SetBackBuffer(D3DResourceType.IDirect3DSurface9, surf.ComPointer);
            d3dimage.Unlock();
            surf.Dispose();

            _device.SetSamplerState(0, SamplerState.MinFilter, TextureFilter.Linear);
            _device.SetSamplerState(0, SamplerState.MagFilter, TextureFilter.Point);
            _device.SetSamplerState(0, SamplerState.MipFilter, TextureFilter.Point);

            initTexture(_imageWidth, _imageHeight);

            return true;
        }

        private void initTexture(int width, int height)
        {
            if (_texture == null)
            {
                _texture = new Texture(_device, width, height, 0, Usage.AutoGenerateMipMap, Format.X8R8G8B8, Pool.Default);
            }
            else
            {
                _graphicsres[0].Unregister();
                _graphicsres.Clear();
                _texture.Dispose();
                _texture = new Texture(_device, width, height, 0, Usage.AutoGenerateMipMap, Format.X8R8G8B8, Pool.Default);
            }

            CudaDirectXInteropResource resource = new CudaDirectXInteropResource(_texture.ComPointer, CUGraphicsRegisterFlags.None, CudaContext.DirectXVersion.D3D9, CUGraphicsMapResourceFlags.None);
            _graphicsres.Add(resource);

            if (_vertexBufferTexture == null)
            {
                _vertexBufferTexture = new VertexBuffer(_device, 4 * Marshal.SizeOf(typeof(vertex)), Usage.None, VertexFormat.Position | VertexFormat.Texture1, Pool.Default);
            }

            DataStream str = _vertexBufferTexture.Lock(0, 4 * Marshal.SizeOf(typeof(vertex)), LockFlags.None);
            str.Position = 0;
            str.WriteRange<vertex>(GetVertices(width, height));
            _vertexBufferTexture.Unlock();

            //_device.SetTexture(0, _texture);
        }

        private void updateFrame()
        {
            if (_device == null)
                return;


            d3dimage.Lock();
            _device.Clear(ClearFlags.Target, _backgroundColor, 0.0f, 0);

            //if (_tileCount > 0)
            //{
            //    //reset pre-alignment in case of plotting displacements
            //    //_device.SetTransform(TransformState.World, SlimDX.Matrix.Identity);
            //}


            if (_device.BeginScene().IsSuccess)
            {
                Result res;
                //Draw particles
                _device.SetTexture(0, _texture);
                res = _device.SetStreamSource(0, _vertexBufferTexture, 0, Marshal.SizeOf(typeof(vertex)));
                _device.VertexFormat = VertexFormat.Position | VertexFormat.Texture1;
                res = _device.DrawPrimitives(PrimitiveType.TriangleStrip, 0, 2);

                SlimDX.Matrix transSave = _device.GetTransform(TransformState.World);

                if (_tileCount > 0)
                {
                    _device.SetTexture(0, null);
                    //draw grid on reference image without shifts
                    _device.SetTransform(TransformState.World, SlimDX.Matrix.Identity);

                    _device.SetStreamSource(0, _vertexBufferTileGrid, 0, 16);
                    _device.VertexFormat = VertexFormat.Position | VertexFormat.Diffuse;
                    _device.DrawPrimitives(PrimitiveType.LineList, 0, _tileCount + 2);
                    _device.SetTransform(TransformState.World, transSave);
                }

                if (_tileCount > 0 && _vertexBufferFlow != null)
                {
                    _device.SetTexture(0, null);
                    //draw grid on reference image without shifts
                    _device.SetTransform(TransformState.World, SlimDX.Matrix.Identity);

                    _device.SetStreamSource(0, _vertexBufferFlow, 0, 16);
                    _device.VertexFormat = VertexFormat.Position | VertexFormat.Diffuse;
                    _device.DrawPrimitives(PrimitiveType.LineList, 0, _tileCount);
                    _device.SetTransform(TransformState.World, transSave);
                }

                _device.EndScene();
            }
            //display
            _swapChain.Present(Present.None);
            d3dimage.AddDirtyRect(new Int32Rect(0, 0, d3dimage.PixelWidth, d3dimage.PixelHeight));
            d3dimage.Unlock();
        }


        private vertex[] GetVertices(int width, int height)
        {
            vertex[] v = new vertex[4];

            if (width < height)
            {
                float w = width / (float)height * 0.5f;
                v = new vertex[] {  new vertex(-w,-0.5f, 0.5f,0.0f,0.0f),
                                        new vertex( w,-0.5f, 0.5f,1.0f,0.0f),
                                        new vertex(-w, 0.5f, 0.5f,0.0f,1.0f),
                                        new vertex( w, 0.5f, 0.5f,1.0f,1.0f) };
            }
            else
            {
                float h = height / (float)width * 0.5f;
                v = new vertex[] {  new vertex(-0.5f,-h, 0.5f,0.0f,0.0f),
                                        new vertex( 0.5f,-h, 0.5f,1.0f,0.0f),
                                        new vertex(-0.5f, h, 0.5f,0.0f,1.0f),
                                        new vertex( 0.5f, h, 0.5f,1.0f,1.0f) };
            }

            return v;
        }

        private void SetProjectionTransform()
        {
            float w = (float)ActualWidthDpi;
            float h = (float)ActualHeightDpi;

            float imgW = _imageWidth;
            float imgH = _imageHeight;

            _sizeOnHeight = w > h;
            float ratioScreen = h / w;
            float ratioImg = imgH / imgW;

            _projFac = 1.0f;
            if (ratioScreen < ratioImg)
            {
                if (imgH > imgW)
                {
                    _sizeOnHeight = true;
                }
                if (imgH < imgW)
                {
                    _projFac = imgH / imgW;
                }
            }
            else
            {
                if (imgH < imgW)
                {
                    _sizeOnHeight = false;
                }
                if (imgH > imgW)
                {
                    _projFac = imgW / imgH;
                }
            }


            SlimDX.Matrix matProj;
            if (_sizeOnHeight)
            {
                float wf = w / (float)h * 0.5f;
                matProj = SlimDX.Matrix.OrthoOffCenterLH(-wf * _projFac, wf * _projFac, 0.5f * _projFac, -0.5f * _projFac, 0, 1);
            }
            else
            {
                float hf = h / (float)w * 0.5f;
                matProj = SlimDX.Matrix.OrthoOffCenterLH(-0.5f * _projFac, 0.5f * _projFac, hf * _projFac, -hf * _projFac, 0, 1);
            }


            // Set projection matrix
            _device.SetTransform(TransformState.Projection, matProj);

        }

        private void Image_SizeChanged(object sender, SizeChangedEventArgs e)
        {
            if (!_initOK) return;

            int w = (int)ActualWidthDpi;
            int h = (int)ActualHeightDpi;


            _swapChain.Dispose();
            PresentParameters pp = new PresentParameters();
            pp.BackBufferWidth = w;
            pp.BackBufferHeight = h;
            pp.BackBufferFormat = Format.Unknown;
            pp.BackBufferCount = 0;
            pp.Multisample = MultisampleType.None;
            pp.MultisampleQuality = 0;
            pp.SwapEffect = SwapEffect.Discard;
            pp.DeviceWindowHandle = (IntPtr)0;
            pp.Windowed = true;
            pp.EnableAutoDepthStencil = false;
            pp.AutoDepthStencilFormat = Format.Unknown;
            pp.PresentationInterval = PresentInterval.Default;

            SetProjectionTransform();

            _swapChain = new SwapChain(_device, pp);
            Surface surf = _swapChain.GetBackBuffer(0);
            _device.SetRenderTarget(0, surf);
            d3dimage.Lock();
            d3dimage.SetBackBuffer(D3DResourceType.IDirect3DSurface9, surf.ComPointer);
            d3dimage.Unlock();
            surf.Dispose();

            updateFrame();
            return;
        }

        private Point GetWorlCoordinateFromMouseCoordinate(Point mouseCoordinate)
        {
            Point ret = new Point();

            SlimDX.Matrix projection = _device.GetTransform(TransformState.Projection);
            SlimDX.Matrix view = _device.GetTransform(TransformState.View);

            SlimDX.Matrix transform = view * projection;
            transform.Invert();

            SlimDX.Vector3 vecMouse = new Vector3((float)((mouseCoordinate.X / ActualWidthDpi) - 0.5f) * 2.0f, (float)-((mouseCoordinate.Y / ActualHeightDpi) - 0.5f) * 2.0f, 0);
            SlimDX.Vector4 vecWorld = Vector3.Transform(vecMouse, transform);

            ret.X = vecWorld.X;
            ret.Y = vecWorld.Y;

            return ret;
        }

        private Point GetImagePixelFromMouseCoordinate(Point mouseCoordinate)
        {
            Point ret = GetWorlCoordinateFromMouseCoordinate(mouseCoordinate);

            if (_realImageHeight > _realImageWidth)
            {
                double wf = _imageWidth / (double)_realImageHeight;
                ret.X = Math.Floor((ret.X + 0.5) * _realImageHeight - (_realImageHeight - _realImageWidth) / 2.0);
                ret.Y = Math.Floor((ret.Y + 0.5) * _realImageHeight);
            }
            else
            {
                double hf = _realImageHeight / (double)_realImageWidth;
                ret.X = Math.Floor((ret.X + 0.5) * _realImageWidth);
                ret.Y = Math.Floor((ret.Y + 0.5) * _realImageWidth - (_realImageWidth - _realImageHeight) / 2.0);
            }

            if (ret.X < 0) ret.X = double.NegativeInfinity;
            if (ret.X >= _realImageWidth) ret.X = double.PositiveInfinity;
            if (ret.Y < 0) ret.Y = double.NegativeInfinity;
            if (ret.Y >= _realImageHeight) ret.Y = double.PositiveInfinity;

            return ret;
        }

        private void Image_MouseWheel(object sender, MouseWheelEventArgs e)
        {
            float factor = e.Delta / 1200.0f + 1.0f;
            //if (factor < 0) factor = -1.0f / factor;
            System.Windows.Point position = GetPositionWithDpi(e);

            float shiftScale = Math.Min((float)ActualWidthDpi, (float)ActualHeightDpi);
            Point ret = GetWorlCoordinateFromMouseCoordinate(position);
            //undo rotation from world coordinates to screen coordinates
            double temp;
            switch (_rotation)
            {
                case Rotation._0:
                    break;
                case Rotation._90:
                    temp = ret.X;
                    ret.X = -ret.Y;
                    ret.Y = temp;
                    break;
                case Rotation._180:
                    ret.Y = -ret.Y;
                    ret.X = -ret.X;
                    break;
                case Rotation._270:
                    temp = ret.X;
                    ret.X = ret.Y;
                    ret.Y = -temp;
                    break;
            }

            ret.X *= shiftScale;
            ret.Y *= shiftScale;

            float scaleFacOld = _scaleFac;
            _scaleFac *= factor;
            if (_scaleFac <= 0.1f)
                _scaleFac = 0.1f;

            _viewShiftX *= scaleFacOld;
            _viewShiftY *= scaleFacOld;
            _viewShiftX -= (float)((ret.X * _scaleFac) - ret.X * scaleFacOld);
            _viewShiftY -= (float)((ret.Y * _scaleFac) - ret.Y * scaleFacOld);
            _viewShiftX /= _scaleFac;
            _viewShiftY /= _scaleFac;

            SlimDX.Matrix matScale = SlimDX.Matrix.Scaling(_scaleFac, _scaleFac, 1);
            SlimDX.Matrix matTrans = SlimDX.Matrix.Translation(_viewShiftX / shiftScale * _scaleFac, _viewShiftY / shiftScale * _scaleFac, 0);

            SlimDX.Matrix mat = matScale * matTrans;
            SlimDX.Matrix rotMat = new SlimDX.Matrix();
            switch (_rotation)
            {
                case Rotation._0:
                    rotMat = SlimDX.Matrix.RotationZ(0);
                    break;
                case Rotation._90:
                    rotMat = SlimDX.Matrix.RotationZ((float)(90.0 / 180.0 * Math.PI));
                    break;
                case Rotation._180:
                    rotMat = SlimDX.Matrix.RotationZ((float)(180.0 / 180.0 * Math.PI));
                    break;
                case Rotation._270:
                    rotMat = SlimDX.Matrix.RotationZ((float)(270.0 / 180.0 * Math.PI));
                    break;
            }

            _device.SetTransform(TransformState.View, rotMat * mat);

            updateFrame();
        }

        private void Image_MouseMove(object sender, MouseEventArgs e)
        {
            System.Windows.Point position = GetPositionWithDpi(e);
            int pX = (int)position.X;
            int pY = (int)position.Y;

            Point pixel = GetImagePixelFromMouseCoordinate(position);

            uchar4[] p = new uchar4[1];

            if (!double.IsInfinity(pixel.X) && !double.IsInfinity(pixel.Y))
            {
                d3dimage.Lock();

                _graphicsres.MapAllResources();
                CudaArray2D arr = _graphicsres[0].GetMappedArray2D(0, 0);

                CUDAMemCpy2D copy = new CUDAMemCpy2D();
                GCHandle handle = GCHandle.Alloc(p, GCHandleType.Pinned);
                copy.dstHost = handle.AddrOfPinnedObject();
                copy.srcArray = arr.CUArray;
                copy.srcMemoryType = CUMemoryType.Array;
                copy.dstMemoryType = CUMemoryType.Host;
                copy.Height = 1;
                copy.WidthInBytes = 4;
                copy.srcXInBytes = (int)pixel.X * 4;
                copy.srcY = (int)pixel.Y;

                arr.CopyData(copy);

                _graphicsres.UnmapAllResources();
                arr.Dispose();

                handle.Free();
                d3dimage.Unlock();
            }
            SetValue(ColorOfPixelProperty, Color.FromArgb(p[0].w, p[0].z, p[0].y, p[0].x));
            SetValue(PixelCoordinateProperty, pixel);

            if (_clicked)
            {
                _viewShiftX += (-_lastX + pX) / _scaleFac * _projFac;
                _viewShiftY += (-_lastY + pY) / _scaleFac * _projFac;
                _lastX = pX;
                _lastY = pY;

                SlimDX.Matrix matScale = SlimDX.Matrix.Scaling(_scaleFac, _scaleFac, 1);
                float shiftScale = Math.Min((float)ActualWidthDpi, (float)ActualHeightDpi);
                SlimDX.Matrix matTrans = SlimDX.Matrix.Translation(_viewShiftX / shiftScale * _scaleFac, _viewShiftY / shiftScale * _scaleFac, 0);

                SlimDX.Matrix mat = matScale * matTrans;

                SlimDX.Matrix rotMat = new SlimDX.Matrix();
                switch (_rotation)
                {
                    case Rotation._0:
                        rotMat = SlimDX.Matrix.RotationZ(0);
                        break;
                    case Rotation._90:
                        rotMat = SlimDX.Matrix.RotationZ((float)(90.0 / 180.0 * Math.PI));
                        break;
                    case Rotation._180:
                        rotMat = SlimDX.Matrix.RotationZ((float)(180.0 / 180.0 * Math.PI));
                        break;
                    case Rotation._270:
                        rotMat = SlimDX.Matrix.RotationZ((float)(270.0 / 180.0 * Math.PI));
                        break;
                }

                _device.SetTransform(TransformState.View, rotMat * mat);

                updateFrame();
            }
        }

        private void Image_MouseDown(object sender, MouseButtonEventArgs e)
        {
            if (e.ChangedButton == MouseButton.Left && e.ClickCount == 2) //double click
            {
                _viewShiftX *= _scaleFac;
                _viewShiftY *= _scaleFac;

                float w = (float)ActualWidthDpi;
                float h = (float)ActualHeightDpi;

                //if (w > h)
                if (_sizeOnHeight)
                {
                    //height is 1
                    _scaleFac = _imageHeight / h;
                }
                else
                {
                    //width is 1
                    _scaleFac = _imageWidth / w;
                }

                _viewShiftX /= _scaleFac;
                _viewShiftY /= _scaleFac;

                SlimDX.Matrix matScale = SlimDX.Matrix.Scaling(_scaleFac, _scaleFac, 1);
                float shiftScale = Math.Min((float)ActualWidthDpi, (float)ActualHeightDpi);
                SlimDX.Matrix matTrans = SlimDX.Matrix.Translation(_viewShiftX / shiftScale * _scaleFac, _viewShiftY / shiftScale * _scaleFac, 0);

                SlimDX.Matrix mat = matScale * matTrans;
                SlimDX.Matrix rotMat = new SlimDX.Matrix();
                switch (_rotation)
                {
                    case Rotation._0:
                        rotMat = SlimDX.Matrix.RotationZ(0);
                        break;
                    case Rotation._90:
                        rotMat = SlimDX.Matrix.RotationZ((float)(90.0 / 180.0 * Math.PI));
                        break;
                    case Rotation._180:
                        rotMat = SlimDX.Matrix.RotationZ((float)(180.0 / 180.0 * Math.PI));
                        break;
                    case Rotation._270:
                        rotMat = SlimDX.Matrix.RotationZ((float)(270.0 / 180.0 * Math.PI));
                        break;
                }

                _device.SetTransform(TransformState.View, rotMat * mat);
                updateFrame();
                return;
            }

            if (e.ChangedButton == MouseButton.Left) //simple click
            {
                System.Windows.Point position = GetPositionWithDpi(e);
                double pX = position.X;
                double pY = position.Y;

                _lastX = (int)pX;
                _lastY = (int)pY;

                _clicked = true;
            }
            if (e.ChangedButton == MouseButton.Right && e.ClickCount == 2)
            {
                _viewShiftX = 0;
                _viewShiftY = 0;


                float w = (float)ActualWidthDpi;
                float h = (float)ActualHeightDpi;
                _scaleFac = 1;

                SlimDX.Matrix matScale = SlimDX.Matrix.Scaling(_scaleFac, _scaleFac, 1);
                float shiftScale = Math.Min((float)ActualWidthDpi, (float)ActualHeightDpi);
                SlimDX.Matrix matTrans = SlimDX.Matrix.Translation(_viewShiftX / shiftScale * _scaleFac, _viewShiftY / shiftScale * _scaleFac, 0);

                SlimDX.Matrix mat = matScale * matTrans;
                SlimDX.Matrix rotMat = new SlimDX.Matrix();
                switch (_rotation)
                {
                    case Rotation._0:
                        rotMat = SlimDX.Matrix.RotationZ(0);
                        break;
                    case Rotation._90:
                        rotMat = SlimDX.Matrix.RotationZ((float)(90.0 / 180.0 * Math.PI));
                        break;
                    case Rotation._180:
                        rotMat = SlimDX.Matrix.RotationZ((float)(180.0 / 180.0 * Math.PI));
                        break;
                    case Rotation._270:
                        rotMat = SlimDX.Matrix.RotationZ((float)(270.0 / 180.0 * Math.PI));
                        break;
                }

                _device.SetTransform(TransformState.View, rotMat * mat);
                updateFrame();
            }
        }

        private void Image_MouseUp(object sender, MouseButtonEventArgs e)
        {
            _lastX = 0;
            _lastY = 0;
            _clicked = false;
        }

        private void Image_MouseLeave(object sender, MouseEventArgs e)
        {
            _lastX = 0;
            _lastY = 0;
            _clicked = false;
        }
        #endregion

        #region Public Interactive Methods
        public void SetImage(CudaPitchedDeviceVariable<uchar4> image, Rotation orientation)
        {
            bool sizeChanged = false;
            if (_realImageWidth != image.Width || _realImageHeight != image.Height || _rotation != orientation)
            {
                sizeChanged = true;
            }

            //Always create a new texture to recreate the mipmaps...
            //if (_realImageWidth != image.Width || _realImageHeight != image.Height || _rotation != orientation)
            {
                _rotation = orientation;
                _realImageWidth = image.Width;
                _realImageHeight = image.Height;
                _imageWidth = image.Width;
                _imageHeight = image.Height;

                if (_rotation == Rotation._90 || _rotation == Rotation._270)
                {
                    _imageWidth = _realImageHeight;
                    _imageHeight = _realImageWidth;
                }
                initTexture(image.Width, image.Height);
            }

            d3dimage.Lock();
            
            _graphicsres.MapAllResources();
            CudaArray2D arr = _graphicsres[0].GetMappedArray2D(0, 0);

            arr.CopyFromDeviceToThis<uchar4>(image);
            _graphicsres.UnmapAllResources();
            arr.Dispose();


            d3dimage.Unlock();

            //if size didn't change, don't change geometry
            if (sizeChanged)
            {
                SetProjectionTransform();
                SlimDX.Matrix rotMat = new SlimDX.Matrix();
                switch (_rotation)
                {
                    case Rotation._0:
                        rotMat = SlimDX.Matrix.RotationZ(0);
                        break;
                    case Rotation._90:
                        rotMat = SlimDX.Matrix.RotationZ((float)(90.0 / 180.0 * Math.PI));
                        break;
                    case Rotation._180:
                        rotMat = SlimDX.Matrix.RotationZ((float)(180.0 / 180.0 * Math.PI));
                        break;
                    case Rotation._270:
                        rotMat = SlimDX.Matrix.RotationZ((float)(270.0 / 180.0 * Math.PI));
                        break;
                }

                _device.SetTransform(TransformState.View, rotMat);
            }

            updateFrame();
        }


        public void SetTiling(int tileCountX, int tileCountY, int tileSize, int levelSize, int maxShift )
        {
            _tileCount = tileCountX * tileCountY;
            float w, h;
            float increment;
            float shiftBorder = (float)(maxShift * levelSize);

            if (_realImageWidth < _realImageHeight)
            {
                float factor = _realImageWidth / (float)_realImageHeight;
                w = factor * 0.5f;
                h = 0.5f;
                increment = (float)tileSize * (float)levelSize / (float)_realImageHeight;
                shiftBorder /= _realImageHeight;
            }
            else
            {
                float factor = _realImageHeight / (float)_realImageWidth;
                w = 0.5f;
                h = factor * 0.5f;
                increment = (float)tileSize * (float)levelSize / (float)_realImageWidth;
                shiftBorder /= _realImageWidth;
            }

            uint color = 0xffff0000; //red

            linevertex[] linevertices = new linevertex[(_tileCount + 2) * 2];
            int index = 0;

            for (int i = 0; i < tileCountX + 1; i++)
            {
                linevertices[index] = new linevertex(-w + shiftBorder + i * increment, -h + shiftBorder, 0.1f, color);
                linevertices[index+1] = new linevertex(-w + shiftBorder + i * increment, -h + shiftBorder + tileCountY * increment, 0.1f, color);
                index += 2;
            }
            for (int i = 0; i < tileCountY + 1; i++)
            {
                linevertices[index] = new linevertex(-w + shiftBorder, -h + shiftBorder + i * increment, 0.1f, color);
                linevertices[index + 1] = new linevertex(-w + shiftBorder + tileCountX * increment, -h + shiftBorder + i * increment, 0.1f, color);
                index += 2;
            }

            if (_vertexBufferTileGrid != null)
            {
                _vertexBufferTileGrid.Dispose();
                _vertexBufferFlow?.Dispose();
                _vertexBufferFlow = null;
            }
            
            _vertexBufferTileGrid = new VertexBuffer(_device, 2*(_tileCount + 2) * Marshal.SizeOf(typeof(linevertex)), Usage.None, VertexFormat.Position, Pool.Default);
            

            DataStream str = _vertexBufferTileGrid.Lock(0, 2*(_tileCount + 2) * Marshal.SizeOf(typeof(linevertex)), LockFlags.None);
            str.Position = 0;
            str.WriteRange<linevertex>(linevertices);
            _vertexBufferTileGrid.Unlock();

            //_device.SetStreamSource(0, _vertexBufferTileGrid, 0, Marshal.SizeOf(typeof(linevertex)));
            //_device.VertexFormat = VertexFormat.Position | VertexFormat.Diffuse;
            //_device.DrawPrimitives(PrimitiveType.LineList, 0, _tileCount + 2);

            updateFrame();
        }

        public void SetTileShifts(int tileCountX, int tileCountY, int tileSize, int levelSize, int maxShift, float2[] flow)
        {
            _tileCount = tileCountX * tileCountY;
            float w, h;
            float increment;
            float maxSize;
            float shiftBorder = (float)(maxShift * levelSize);

            if (_realImageWidth < _realImageHeight)
            {
                float factor = _realImageWidth / (float)_realImageHeight;
                w = factor * 0.5f;
                h = 0.5f;
                increment = (float)tileSize * (float)levelSize / (float)_realImageHeight;
                maxSize = 1.0f / _realImageHeight;
                shiftBorder /= _realImageHeight;
            }
            else
            {
                float factor = _realImageHeight / (float)_realImageWidth;
                w = 0.5f;
                h = factor * 0.5f;
                increment = (float)tileSize * (float)levelSize / (float)_realImageWidth;
                maxSize = 1.0f / _realImageWidth;
                shiftBorder /= _realImageWidth;
            }

            uint color = 0xff00ff00; //green

            linevertex[] linevertices = new linevertex[(_tileCount) * 2];

            for (int y = 0; y < tileCountY; y++)
            {
                for (int x = 0; x < tileCountX; x++)
                {
                    int index = y * tileCountX + x;

                    float2 shift = flow[index] * levelSize * maxSize;

                    float posx = -w + shiftBorder + (x + 0.5f) * increment;
                    float posy = -h + shiftBorder + (y + 0.5f) * increment;
                    linevertices[2 * index] = new linevertex(posx, posy, 0.1f, color);
                    linevertices[2 * index + 1] = new linevertex(posx + shift.x, posy + shift.y, 0.1f, color);
                }
            }

            if (_vertexBufferFlow != null)
            {
                _vertexBufferFlow.Dispose();
            }

            _vertexBufferFlow = new VertexBuffer(_device, 2 * (_tileCount) * Marshal.SizeOf(typeof(linevertex)), Usage.None, VertexFormat.Position, Pool.Default);


            DataStream str = _vertexBufferFlow.Lock(0, 2 * (_tileCount) * Marshal.SizeOf(typeof(linevertex)), LockFlags.None);
            str.Position = 0;
            str.WriteRange<linevertex>(linevertices);
            _vertexBufferFlow.Unlock();
            
            updateFrame();
            
        }

        public void ResetTiling()
        {
            _tileCount = 0;
            updateFrame();
        }


        public void SetPreAlignment(float preShiftX, float preShiftY, float preRotationRad)
        {
            float maxDim = Math.Max(_realImageWidth, _realImageHeight);
            SlimDX.Matrix shift = SlimDX.Matrix.Translation(preShiftX / maxDim, preShiftY / maxDim, 0);
            SlimDX.Matrix rotation = SlimDX.Matrix.RotationZ(-preRotationRad);
                       
            SlimDX.Matrix world = rotation * shift;

            _device.SetTransform(TransformState.World, world);
        }
        #endregion

        private void UserControl_Loaded(object sender, RoutedEventArgs e)
        {
            _initOK = InitializeD3D();
            updateFrame();
        }

        private Point GetPositionWithDpi(MouseEventArgs e)
        {
            System.Windows.Point position = e.GetPosition(this);
            PresentationSource source = PresentationSource.FromVisual(this);
            if (source != null)
            {
                position.X *= source.CompositionTarget.TransformToDevice.M11;
                position.Y *= source.CompositionTarget.TransformToDevice.M22;
            }
            return position;
        }

        #region Properties
        public CudaContext CudaContext
        {
            get { return _ctx; }
        }

        public double ActualWidthDpi
        {
            get
            {
                PresentationSource source = PresentationSource.FromVisual(this);
                if (source != null)
                {
                    return ActualWidth * source.CompositionTarget.TransformToDevice.M11;
                }
                return ActualWidth;
            }
        }

        public double ActualHeightDpi
        {
            get
            {
                PresentationSource source = PresentationSource.FromVisual(this);
                if (source != null)
                {
                    return ActualHeight * source.CompositionTarget.TransformToDevice.M22;
                }
                return ActualWidth;
            }
        }
        #endregion

        #region Dependency Properties


        public Color ColorOfPixel
        {
            get { return (Color)GetValue(ColorOfPixelProperty); }
            set { SetValue(ColorOfPixelProperty, value); }
        }

        public static readonly DependencyProperty ColorOfPixelProperty =
            DependencyProperty.Register("ColorOfPixel", typeof(Color), typeof(ImagePresenterDX), new PropertyMetadata(Color.FromArgb(0, 0, 0, 0)));


        public Point PixelCoordinate
        {
            get { return (Point)GetValue(PixelCoordinateProperty); }
            set { SetValue(PixelCoordinateProperty, value); }
        }

        public static readonly DependencyProperty PixelCoordinateProperty =
            DependencyProperty.Register("PixelCoordinate", typeof(Point), typeof(ImagePresenterDX), new PropertyMetadata(new Point(0, 0)));


        #endregion

    }
}
