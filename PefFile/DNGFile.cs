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
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Drawing;
using System.Drawing.Imaging;
using System.Threading.Tasks;


namespace PentaxPefFile
{
    //Most DNG related features are heavily "inspired" by the DNG SDK
    //https://www.adobe.com/support/downloads/dng/dng_sdk.html

    public class DNGFile : RawFile
    {
        private IFDDNGLinearizationTable mLinearizationTable;

        private bool isDNGVersionLarge = false;

        #region "Inspired" by dcraw
        private unsafe class jhead
        {
            public int algo, bits, high, wide, clrs, sraw, psv, restart;
            public int[] vpred;
            public ushort[] quant;
            public ushort[] idct;
            public ushort[][] huff;
            public ushort[][] free;
            public ushort[] row;
            public ushort* rowPtr;

            public jhead()
            {
                algo = 0;
                bits = 0;
                high = 0;
                wide = 0;
                clrs = 0;
                sraw = 0;
                psv = 0;
                restart = 0;

                vpred = new int[6];
                quant = new ushort[64];
                idct = new ushort[64];
                huff = new ushort[20][];
                free = new ushort[20][];
                row = null;
            }
        }

        private unsafe
        ushort[] make_decoder_ref(byte** source)
        {

            int max, len, h, i, j;
            byte* count;
            ushort[] huff;

            count = (*source += 16) - 17;
	        for (max = 16; max > 0 && (count[max] == 0); max--);
            huff = new ushort[1 + (1 << max)];// (ushort*)calloc(1 + (1 << max), sizeof *huff);
	        //merror(huff, "make_decoder()");
	        huff[0] = (ushort)max;
	        for (h = len = 1; len <= max; len++)
		        for (i = 0; i<count[len]; i++, ++*source)
			        for (j = 0; j< 1 << (max - len); j++)
				        if (h <= 1 << max)
					        huff[h++] = (ushort)(len << 8 | **source);
	        return huff;
        }
        
        unsafe int ljpeg_start(jhead jh, int info_only, Stream s)
        {

            int c, tag, len;
            byte[] data = new byte[0x10000];
            byte* dp;

            //jh = new jhead();
            
            jh.restart = int.MaxValue;

            s.ReadByte();
	        if ((s.ReadByte()) != 0xd8) return 0;
	        do {
                //s.Read(data, 0, 4);
		        if (4 != s.Read(data, 0, 4))
                    return 0;
		        tag = data[0] << 8 | data[1];
		        len = (data[2] << 8 | data[3]) - 2;
		        if (tag <= 0xff00)
                    return 0;

                s.Read(data, 0, len);
                //fread(data, 1, len, ifp);
		        switch (tag) {
		        case 0xffc3:
			        jh.sraw = ((data[7] >> 4) * (data[7] & 15) - 1) & 3;
                    jh.algo = tag & 0xff;
                    jh.bits = data[0];
                    jh.high = data[1] << 8 | data[2];
                    jh.wide = data[3] << 8 | data[4];
                    jh.clrs = data[5] + jh.sraw;
                        //if (len == 9 && !dng_version) getc(ifp);
                        break;
		        case 0xffc1:
		        case 0xffc0:
			        jh.algo = tag & 0xff;
			        jh.bits = data[0];
			        jh.high = data[1] << 8 | data[2];
			        jh.wide = data[3] << 8 | data[4];
			        jh.clrs = data[5] + jh.sraw;
			        //if (len == 9 && !dng_version) getc(ifp);
			        break;
		        case 0xffc4:
                    //if (info_only) break;
                    fixed (byte* ptr = data)
                    {
                        for (dp = ptr; dp < ptr + len && ((c = *dp++) & -20) == 0;)
                            jh.free[c] = jh.huff[c] = make_decoder_ref(&dp);
                    }
			        break;
		        case 0xffda:
			        jh.psv = data[1 + data[0] * 2];
			        jh.bits -= data[3 + data[0] * 2] & 15;
			        break;
		        case 0xffdb:
                    for(c = 0; c < 64; c++)
                        jh.quant[c] = (ushort)(data[c * 2 + 1] << 8 | data[c * 2 + 2]);
			        break;
		        case 0xffdd:
			        jh.restart = data[0] << 8 | data[1];
                    break;
		        }
        } while (tag != 0xffda);

            for (c = 0; c < 19; c++)
                if (jh.huff[c + 1] == null) jh.huff[c + 1] = jh.huff[c];

            //if (jh.sraw) {

            //       FORC(4)        jh->huff[2 + c] = jh->huff[1];

            //       FORC(jh->sraw) jh->huff[1 + c] = jh->huff[0];
            //}
            jh.row = new ushort[jh.wide * jh.clrs * 2]; //(ushort*)calloc(jh->wide* jh->clrs, 4);
	        //merror(jh->row, "ljpeg_start()");
	        return 1;
        }

        private unsafe ushort* ljpeg_row(int jrow, jhead jh, Stream s)
        {
	        int col, c, diff, pred, spred = 0;
            ushort mark = 0;
            ushort*[] row = new ushort*[3];

	        if (jrow * jh.wide % jh.restart == 0)
            {
                for (c = 0; c < 6; c++)
                    jh.vpred[c] = 1 << (jh.bits - 1);

                if (jrow > 0)
                {
                    s.Seek(-2, SeekOrigin.Current);
                    //fseek(ifp, -2, SEEK_CUR);
			        do
                        mark = (ushort)((mark << 8) + (c = s.ReadByte()));
			        while (s.Position < s.Length - 1 && mark >> 4 != 0xffd);
		        }
                getbithuff(-1, null, s);
            }

            for (c = 0; c < 3; c++)
                row[c] = jh.rowPtr + jh.wide * jh.clrs * ((jrow + c) & 1);

	        for (col = 0; col<jh.wide; col++)
                for (c = 0; c < jh.clrs; c++)
                {
                    fixed (ushort* ptr = jh.huff[c])
                    {
                        diff = ljpeg_diff(ptr, s);
                        if (jh.sraw > 0 && c <= jh.sraw && (col | c) > 0)
                            pred = spred;
                        else if (col > 0) pred = row[0][-jh.clrs];
                        else pred = (jh.vpred[c] += diff) - diff;
                        if (jrow > 0 && col > 0)
                            switch (jh.psv)
                            {
                                case 1: break;
                                case 2: pred = row[1][0]; break;
                                case 3: pred = row[1][-jh.clrs]; break;
                                case 4: pred = pred + row[1][0] - row[1][-jh.clrs]; break;
                                case 5: pred = pred + ((row[1][0] - row[1][-jh.clrs]) >> 1); break;
                                case 6: pred = row[1][0] + ((pred - row[1][-jh.clrs]) >> 1); break;
                                case 7: pred = (pred + row[1][0]) >> 1; break;
                                default: pred = 0;
                                    break;
                            }
                        if ((*row[0] = (ushort)(pred + diff)) >> jh.bits != 0)
                            return null;
                        if (c <= jh.sraw) spred = *row[0];
                        row[0]++; row[1]++;
                    }
                }
	            return row[2];
            }


        //these functions are a bit different than in PEF!
        uint bitbuf = 0;
        int vbits = 0, reset = 0;
        private unsafe uint getbithuff(int nbits, ushort* huff, Stream s)
        {
            uint c;

            if (nbits > 25) return 0;
            if (nbits < 0)
            {
                reset = 0;
                vbits = 0;
                bitbuf = 0;
                return 0;
            }

            if (nbits == 0 || vbits < 0) return 0;

            while (!(reset != 0) && vbits < nbits /*&& (c = ReadUI1() fgetc(ifp)) != EOF && !(reset = 0 && c == 0xff && fgetc(ifp))*/)
            {
                c = (uint)s.ReadByte();// ReadUI1();
                if (c == 255)
                {
                    int t = s.ReadByte();
                }
                bitbuf = (bitbuf << 8) + (byte)c;
                vbits += 8;
            }
            c = bitbuf << (32 - vbits) >> (32 - nbits);
            if (huff != null)
            {
                vbits -= huff[c] >> 8;
                c = (byte)huff[c];
            }
            else
                vbits -= nbits;
            return c;
        }

        private unsafe int ljpeg_diff(ushort* huff, Stream s)
        {
            int len, diff;
            len = (int)getbithuff(*huff, huff + 1, s);
            if (len == 16 && isDNGVersionLarge)
                return -32768;

            diff = (int)getbithuff(len, null, s);
            if ((diff & (1 << (len - 1))) == 0)
                diff -= (1 << len) - 1;
            return diff;
        }
        #endregion

        public DNGFile(string aFileName)
            : base(aFileName)
        {

            IFDDNGVersion version = mIfds[0].GetEntry<IFDDNGVersion>();
            isDNGVersionLarge = version == null;
            if (version != null)
                isDNGVersionLarge = version.Value[1] > 1;


            ImageFileDirectory rawIFD = SetMembers();

            //try to read additional noise models
            try
            {
                ExtraCameraProfiles profiles = ExtraCameraProfiles.Load("ExtraCameraProfiles.xml");
                string make = mIfds[0].GetEntry<IFDMake>()?.Value;
                string uniqueModel = mIfds[0].GetEntry<IFDDNGUniqueCameraModel>()?.Value;

                ExtraCameraProfile profile = profiles.GetProfile(make, uniqueModel);

                double a, b;
                (a, b) = profile.NoiseModel.GetValue(mISO);

                if (a != 0.0)
                {
                    mNoiseModelAlpha = (float)a;
                    mNoiseModelBeta = (float)b;
                }
            }
            catch (Exception)
            {

            }

            //get non-global blacklevels
            ushort[] blackLevelV = new ushort[mHeight];
            ushort[] blackLevelH = new ushort[mWidth];

            int maxBlackLevel = 0;
            IFDDNGBlackLevelDeltaV deltaV = rawIFD.GetEntry<IFDDNGBlackLevelDeltaV>();
            IFDDNGActiveArea activeArea = rawIFD.GetEntry<IFDDNGActiveArea>();

            int blackOffsetH = 0;
            int blackOffsetV = 0;
            int borderBottom = 0;
            int borderRight = 0;

            if (activeArea != null)
            {
                blackOffsetV = (int)activeArea.Value[0];
                blackOffsetH = (int)activeArea.Value[1];
                borderBottom = mHeight - (int)activeArea.Value[2];
                borderRight = mWidth - (int)activeArea.Value[3];
            }


            if (deltaV != null)
            {

                if (deltaV.Value.Length + blackOffsetV  + borderBottom != mHeight)
                {
                    throw new Exception("Count in IFDDNGBlackLevelDeltaV doesn't fit image height");
                }

                for (int i = 0; i < deltaV.Value.Length; i++)
                {
                    blackLevelV[blackOffsetV + i] = (ushort)deltaV.Value[i].Value;
                }
            }

            IFDDNGBlackLevelDeltaH deltaH = rawIFD.GetEntry<IFDDNGBlackLevelDeltaH>();
            if (deltaH != null)
            {
                if (deltaH.Value.Length + blackOffsetH + borderRight != mWidth)
                {
                    throw new Exception("Count in IFDDNGBlackLevelDeltaH doesn't fit image width");
                }

                for (int i = 0; i < deltaH.Value.Length; i++)
                {
                    blackLevelH[blackOffsetH + i] = (ushort)deltaH.Value[i].Value;
                }
            }

            //read actual image data

            //data stored in tiles:
            if (rawIFD.GetEntry<IFDTileWidth>() != null)
            {
                int tileWidth = (int)rawIFD.GetEntry<IFDTileWidth>().Value;
                int tileHeight = (int)rawIFD.GetEntry<IFDTileLength>().Value;

                uint[] offsets = rawIFD.GetEntry<IFDTileOffsets>().Value;
                uint[] byteCounts = rawIFD.GetEntry<IFDTileByteCounts>().Value;

                mRawImage = new ushort[mHeight, mWidth];

                int row = 0;
                int col = 0;



                if (rawIFD.GetEntry<IFDCompression>().Value == IFDCompression.Compression.LosslessJPEG)
                {
                    for (int tile = 0; tile < offsets.Length; tile++)
                    {
                        byte[] data;
                        Seek(offsets[tile], SeekOrigin.Begin);
                        data = mFileReader.ReadBytes((int)byteCounts[tile]);
                        MemoryStream ms = new MemoryStream(data);
                        ms.Seek(0, SeekOrigin.Begin);
                        jhead jh = new jhead();
                        int ret = ljpeg_start(jh, 0, ms);
                        int jrow, jcol;
                        if (ret > 0 && jh != null)
                        {
                            int jwide = jh.wide;
                            jwide *= jh.clrs;


                            unsafe
                            {
                                if (jh.algo == 0xc3) //lossless JPEG
                                {
                                    for (jrow = 0; jrow < jh.high; jrow++)
                                    {
                                        fixed (ushort* ptr = jh.row)
                                        {
                                            jh.rowPtr = ptr;
                                            ushort* rp = ljpeg_row(jrow, jh, ms);
                                            for (jcol = 0; jcol < jwide; jcol++)
                                            {
                                                if (jcol + col < mWidth && jrow + row < mHeight)
                                                {
                                                    int black = blackLevelH[jcol + col];
                                                    black += blackLevelV[jrow + row];
                                                    maxBlackLevel = Math.Max(maxBlackLevel, black);
                                                    if (mLinearizationTable != null)
                                                    {
                                                        mRawImage[row + jrow, col + jcol] = (ushort)Math.Max(0, (int)(mLinearizationTable.Value[rp[jcol] < mLinearizationTable.Value.Length ? rp[jcol] : mLinearizationTable.Value.Length - 1]) - black);
                                                    }
                                                    else
                                                    {
                                                        mRawImage[row + jrow, col + jcol] = (ushort)Math.Max(0, (int)(rp[jcol] - black));
                                                    }
                                                }
                                            }
                                            jh.rowPtr = null;
                                        }
                                    }
                                }
                            }
                        }
                        col += tileWidth;
                        if (col >= mWidth)
                        {
                            col = 0;
                            row += tileHeight;
                        }
                    }
                }
                else
                {
                    throw new ArgumentException("I don't know how to read that file :(");
                }
            }
            //data stored in strips:
            else if (rawIFD.GetEntry<IFDStripOffsets>() != null)
            {
                uint[] offsets = rawIFD.GetEntry<IFDStripOffsets>().Value;
                uint[] byteCounts = rawIFD.GetEntry<IFDStripByteCounts>().Value;
                uint rowsPerStrip = rawIFD.GetEntry<IFDRowsPerStrip>().Value;

                if (rawIFD.GetEntry<IFDCompression>().Value == IFDCompression.Compression.LosslessJPEG)
                {
                    mRawImage = new ushort[mHeight, mWidth];
                    for (int strip = 0; strip < offsets.Length; strip++)
                    {
                        byte[] data;
                        Seek(offsets[strip], SeekOrigin.Begin);
                        data = mFileReader.ReadBytes((int)byteCounts[strip]);
                        MemoryStream ms = new MemoryStream(data);
                        ms.Seek(0, SeekOrigin.Begin);
                        jhead jh = new jhead();
                        int ret = ljpeg_start(jh, 0, ms);
                        int jrow, jcol;
                        if (ret > 0 && jh != null)
                        {
                            int jwide = jh.wide;
                            jwide *= jh.clrs;


                            unsafe
                            {
                                if (jh.algo == 0xc3) //lossless JPEG
                                {
                                    for (jrow = 0; jrow < jh.high; jrow++)
                                    {
                                        fixed (ushort* ptr = jh.row)
                                        {
                                            jh.rowPtr = ptr;
                                            ushort* rp = ljpeg_row(jrow, jh, ms);
                                            for (jcol = 0; jcol < jwide; jcol++)
                                            {
                                                if (jcol < mWidth && jrow + (strip * rowsPerStrip) < mHeight)
                                                {
                                                    int black = blackLevelH[jcol];
                                                    black += blackLevelV[(strip * rowsPerStrip) + jrow];
                                                    maxBlackLevel = Math.Max(maxBlackLevel, black);
                                                    if (mLinearizationTable != null)
                                                        mRawImage[(strip * rowsPerStrip) + jrow, jcol] = (ushort)Math.Max(0, (int)((mLinearizationTable.Value[rp[jcol] < mLinearizationTable.Value.Length ? rp[jcol] : mLinearizationTable.Value.Length - 1]) - black));
                                                    else
                                                        mRawImage[(strip * rowsPerStrip) + jrow, jcol] = (ushort)Math.Max(0, (int)(rp[jcol] - black));
                                                }
                                            }
                                            jh.rowPtr = null;
                                        }
                                    }
                                }
                            }
                        }
                        
                    }
                }
                else if (rawIFD.GetEntry<IFDCompression>().Value == IFDCompression.Compression.NoCompression)
                {

                    mRawImage = new ushort[mHeight, mWidth];

                    if (mBitDepth == 16)
                    {
                        for (int strip = 0; strip < offsets.Length; strip++)
                        {
                            byte[] data;
                            Seek(offsets[strip], SeekOrigin.Begin);
                            data = mFileReader.ReadBytes((int)byteCounts[strip]);

                            unsafe
                            {
                                fixed (byte* ptr = data)
                                {
                                    ushort* usptr = (ushort*)ptr;
                                    for (int pixel = 0; pixel < data.Length / 2; pixel++)
                                    {
                                        int row = strip * (int)rowsPerStrip;
                                        int col = pixel;
                                        if (col >= mWidth)
                                        {
                                            row += pixel / mWidth;
                                            col = pixel % mWidth;
                                        }

                                        int black = blackLevelH[col];
                                        black += blackLevelV[row];
                                        maxBlackLevel = Math.Max(maxBlackLevel, black);
                                        if (mLinearizationTable != null)
                                            mRawImage[row, col] = (ushort)Math.Max(0, (int)(mLinearizationTable.Value[usptr[pixel] < mLinearizationTable.Value.Length ? usptr[pixel] : mLinearizationTable.Value.Length - 1])-black);
                                        else
                                            mRawImage[row, col] = (ushort)Math.Max(0, (int)(usptr[pixel])-black);
                                    }
                                }
                            }
                        }
                    }
                    else if (mBitDepth < 16)
                    {
                        for (int strip = 0; strip < offsets.Length; strip++)
                        {
                            byte[] data;
                            Seek(offsets[strip], SeekOrigin.Begin);
                            data = mFileReader.ReadBytes((int)byteCounts[strip]);

                            uint pos = 0;
                            int offset = 0;

                            for (int line = 0; line < rowsPerStrip; line++)
                            {
                                for (int pixel = 0; pixel < mWidth; pixel++)
                                {
                                    int row = strip * (int)rowsPerStrip + line;
                                    int col = pixel;

                                    uint val = data[pos];

                                    while (offset < mBitDepth)
                                    {
                                        val = data[pos];
                                        bitbuf = (bitbuf << 8) + val;
                                        offset += 8;
                                        pos++;
                                    }

                                    val = bitbuf << (32 - offset) >> (32 - mBitDepth);
                                    offset -= mBitDepth;

                                    int black = blackLevelH[col];
                                    black += blackLevelV[row];
                                    maxBlackLevel = Math.Max(maxBlackLevel, black);

                                    if (mLinearizationTable != null)
                                        mRawImage[row, col] = (ushort)Math.Max(0, (int)(mLinearizationTable.Value[val < mLinearizationTable.Value.Length ? (ushort)(val) : mLinearizationTable.Value.Length - 1]) - black);
                                    else
                                        mRawImage[row, col] = (ushort)Math.Max(0, (int)(val) - black);
                                }
                            }


                        }
                    }
                }
                else 
                {
                    throw new ArgumentException("I don't know how to read that file :(");
                }

            }
            //close the file
            Close();

            WhiteLevel[0] -= maxBlackLevel;
            WhiteLevel[1] -= maxBlackLevel;
            WhiteLevel[2] -= maxBlackLevel;
        }

        public DNGFile(string aFileName, bool HeaderOnly)
            : base(aFileName)
        {
            SetMembers();
            //close the file
            Close();
        }


        private ImageFileDirectory SetMembers()
        {
            ImageFileDirectory rawIFD = null;
            //find the IFD with the RAW Bayer image
            for (int i = 0; i < mIfds.Count; i++)
            {
                //well, it should actually be somewhere in IFD0...
                if (mIfds[i].GetEntry<IFDPhotometricInterpretation>() != null)
                {
                    if (mIfds[i].GetEntry<IFDPhotometricInterpretation>().Value ==
                        IFDPhotometricInterpretation.PhotometricInterpretation.CFA)
                    {
                        rawIFD = mIfds[i];
                        break;
                    }
                }
            }

            //no root IFD seems to contain RAW bayer, search for Sub-IFDs
            if (rawIFD == null)
            {
                //find the IFD with the RAW Bayer image
                for (int i = 0; i < mIfds.Count; i++)
                {
                    IFDSubIFDs subIFD = mIfds[i].GetEntry<IFDSubIFDs>();
                    if (subIFD == null)
                    {
                        continue;
                    }
                    for (int j = 0; j < subIFD.Value.Count; j++)
                    {
                        if (subIFD.Value[j].GetEntry<IFDPhotometricInterpretation>().Value ==
                        IFDPhotometricInterpretation.PhotometricInterpretation.CFA)
                        {
                            rawIFD = subIFD.Value[j];
                            break;
                        }
                    }
                }
            }

            if (rawIFD == null)
            {
                throw new ArgumentException("Can't find IFD with Bayer RAW image.");
            }

            mLinearizationTable = rawIFD.GetEntry<IFDDNGLinearizationTable>();
            mWidth = (int)rawIFD.GetEntry<IFDImageWidth>().Value;
            mHeight = (int)rawIFD.GetEntry<IFDImageLength>().Value;
            if (mIfds[0].GetEntry<IFDDateTime>() != null)
            {
                mRecordingDate = mIfds[0].GetEntry<IFDDateTime>().Value;
            }

            //in case of Pentax this will have succes:
            IFDDNGPrivateData privateData = mIfds[0].GetEntry<IFDDNGPrivateData>();
            if (privateData != null)
            {
                MNLevelInfo levelInfo = privateData.PentaxMakerNotes.GetEntry<MNLevelInfo>();
                if (levelInfo != null)
                {
                    mRollAngle = levelInfo.Value.RollAngle;
                    mRollAnglePresent = true;
                }
            }

            IFDExif exif = mIfds[0].GetEntry<IFDExif>();
            if (exif != null)
            {
                mISO = exif.GetEntry<ExifISOSpeedRatings>().Value;
                mExposureTime = exif.GetEntry<ExifExposureTime>().Value;
                mRecordingDate = (exif.GetEntry<ExifDateTimeDigitized>()?.Value).GetValueOrDefault(mRecordingDate);
            }
            else if (rawIFD.GetEntry<IFDISOSpeedRatings>() != null)
            {
                mISO = rawIFD.GetEntry<IFDISOSpeedRatings>().Value;
                //very likely that exposure time is also present
                mExposureTime = rawIFD.GetEntry<IFDExposureTime>().Value;
            }
            else if (mIfds[0].GetEntry<IFDISOSpeedRatings>() != null)
            {
                mISO = mIfds[0].GetEntry<IFDISOSpeedRatings>().Value;
                //very likely that exposure time is also present
                mExposureTime = mIfds[0].GetEntry<IFDExposureTime>().Value;
            }

            mBitDepth = rawIFD.GetEntry<IFDBitsPerSample>().Value[0];

            int bayerWidth = rawIFD.GetEntry<IFDCFARepeatPatternDim>().Value[0];
            int bayerHeight = rawIFD.GetEntry<IFDCFARepeatPatternDim>().Value[1];
            if (bayerHeight != 2 || bayerWidth != 2)
            {
                throw new ArgumentException("This file has a bayer pattern size different than 2x2. Can't decode that.");
            }

            ExifCFAPattern.BayerColor[] bayer = rawIFD.GetEntry<IFDCFAPattern>().Value;
            mBayerPattern = new BayerColor[bayer.Length];
            for (int i = 0; i < bayer.Length; i++)
            {
                mBayerPattern[i] = (BayerColor)(int)bayer[i];
            }

            IFDDNGCFAPlaneColor planeColor = rawIFD.GetEntry<IFDDNGCFAPlaneColor>();
            int[] planeColorHelper = new int[] { 0, 1, 2 };
            if (planeColor != null) //did it ever differ from 0,1,2?
            {
                //0 = red, 1 = gree, 2 = blue.
                //The debayer algo creates images with plane order red/green/blue.
                //If this order differs, we need to re-order the planes in order to 
                //have the color matrices correct.

                //reset colorTwist matrix:
                mColorTwist = new float[3, 4];
                if (planeColor.Value.Length > 3 || planeColor.Value.Length < 3)
                {
                    throw new ArgumentException("This image doesn't contain three color planes.");
                }

                for (int i = 0; i < planeColor.Value.Length; i++)
                {
                    int color = planeColor.Value[i];
                    planeColorHelper[i] = color;
                    if (color > 2)
                    {
                        throw new ArgumentException("This image contains colors different than red/green/blue.");
                    }
                    mColorTwist[color, i] = 1;

                    if (color != i)
                    {
                        mColorTwistIsIdentity = false;
                    }
                }
            }

            mColorSpec = new DNGColorSpec(3, mIfds[0], rawIFD);
            
            IFDDNGWhiteLevel whiteLevel = rawIFD.GetEntry<IFDDNGWhiteLevel>();
            IFDDNGBlackLevel blackLevel = rawIFD.GetEntry<IFDDNGBlackLevel>();
            
            mBlackLevel = new float[3];
            if (blackLevel != null)
            {
                //only one value for all colors
                if (blackLevel.Value.Length == 1)
                {
                    mBlackLevel[0] = (float)blackLevel.Value[0].Value;
                    mBlackLevel[1] = (float)blackLevel.Value[0].Value;
                    mBlackLevel[2] = (float)blackLevel.Value[0].Value;
                }

                //values per color channel
                if (blackLevel.Value.Length == 3)
                {
                    mBlackLevel[planeColorHelper[0]] = (float)blackLevel.Value[0].Value;
                    mBlackLevel[planeColorHelper[1]] = (float)blackLevel.Value[1].Value;
                    mBlackLevel[planeColorHelper[2]] = (float)blackLevel.Value[2].Value;
                }

                //values per color bayer pattern
                if (blackLevel.Value.Length == 4)
                {
                    //red
                    int indexR = -1;
                    for (int i = 0; i < mBayerPattern.Length; i++)
                    {
                        if (mBayerPattern[i] == BayerColor.Red)
                        {
                            indexR = i;
                            break;
                        }
                    }
                    mBlackLevel[0] = (float)blackLevel.Value[indexR].Value;

                    //blue
                    int indexB = -1;
                    for (int i = 0; i < mBayerPattern.Length; i++)
                    {
                        if (mBayerPattern[i] == BayerColor.Blue)
                        {
                            indexB = i;
                            break;
                        }
                    }
                    mBlackLevel[2] = (float)blackLevel.Value[indexB].Value;

                    //green, the two remaining indices
                    int indexG1 = -1, indexG2 = -1;
                    for (int i = 0; i < mBayerPattern.Length; i++)
                    {
                        if (mBayerPattern[i] == BayerColor.Green && indexG1 == -1)
                        {
                            indexG1 = i;
                        }
                        if (mBayerPattern[i] == BayerColor.Green && indexG1 != -1)
                        {
                            indexG2 = i;
                        }
                    }
                    float g1 = (float)blackLevel.Value[indexG1].Value;
                    float g2 = (float)blackLevel.Value[indexG2].Value;

                    mBlackLevel[1] = Math.Max(g1, g2); //well, one could distinguish the two greens, but what for?
                }
            }
            mWhiteLevel = new float[3];
            mWhiteLevel[0] = (float)whiteLevel.Value[0];
            mWhiteLevel[1] = (float)whiteLevel.Value[0];
            mWhiteLevel[2] = (float)whiteLevel.Value[0];

            //subtract black level from white level
            mWhiteLevel[0] -= mBlackLevel[0];
            mWhiteLevel[1] -= mBlackLevel[1];
            mWhiteLevel[2] -= mBlackLevel[2];

            //get white balance from color spec
            DNGVector wb = mColorSpec.CameraWhite;
            mWhiteBalance = new float[3];
            mWhiteBalance[planeColorHelper[0]] = 1.0f / (float)wb[0];
            mWhiteBalance[planeColorHelper[1]] = 1.0f / (float)wb[1];
            mWhiteBalance[planeColorHelper[2]] = 1.0f / (float)wb[2];


            //look for orientation tag. If RAW ifd has the tag, choose that one
            if (rawIFD.GetEntry<IFDOrientation>() != null)
            {
                mOrientation = new DNGOrientation(rawIFD.GetEntry<IFDOrientation>().Value);
            }
            else if (mIfds[0].GetEntry<IFDOrientation>() != null)
            {
                mOrientation = new DNGOrientation(mIfds[0].GetEntry<IFDOrientation>().Value);
            }
            else
            {
                //no tag found, use default
                mOrientation = new DNGOrientation(DNGOrientation.Orientation.Normal);
            }

            //default Values:
            int cropLeft = 0;
            int cropTop = 0;
            int croppedWidth = mWidth;
            int croppedHeight = mHeight;

            IFDDNGActiveArea activeArea = rawIFD.GetEntry<IFDDNGActiveArea>();
            //if active area is defined:
            if (activeArea != null)
            {
                int top, left, bottom, right;
                top = (int)activeArea.Value[0];
                left = (int)activeArea.Value[1];
                bottom = (int)activeArea.Value[2];
                right = (int)activeArea.Value[3];

                cropLeft += left;
                cropTop += top;
                croppedWidth = right - left;
                croppedHeight = bottom - top;

                //CFA pattern is defined on active area. If top/left is uneven we need to shift the CFA pattern accordingly
                if (top % 2 != 0)
                {
                    BayerColor bayer0 = BayerPattern[0];
                    BayerColor bayer1 = BayerPattern[1];
                    BayerPattern[0] = BayerPattern[2];
                    BayerPattern[1] = BayerPattern[3];
                    BayerPattern[2] = bayer0;
                    BayerPattern[3] = bayer1;
                }

                if (left % 2 != 0)
                {
                    BayerColor bayer0 = BayerPattern[0];
                    BayerColor bayer2 = BayerPattern[2];
                    BayerPattern[0] = BayerPattern[1];
                    BayerPattern[2] = BayerPattern[3];
                    BayerPattern[1] = bayer0;
                    BayerPattern[3] = bayer2;
                }
            }

            IFDDNGDefaultCropOrigin cropOrigin = rawIFD.GetEntry<IFDDNGDefaultCropOrigin>();
            IFDDNGDefaultCropSize cropSize = rawIFD.GetEntry<IFDDNGDefaultCropSize>();

            if (cropOrigin != null && cropSize != null)
            {
                int top, left, width, height;
                left = (int)(cropOrigin.Value[0].Value);
                top = (int)(cropOrigin.Value[0].Value);
                width = (int)(cropSize.Value[0].Value);
                height = (int)(cropSize.Value[0].Value);

                cropLeft += left;
                cropTop += top;

                croppedWidth = width;
                croppedHeight = height;
            }

            //we always crop at least two pixels because of our algos...
            mCropLeft = Math.Max(2, cropLeft);
            mCropTop = Math.Max(2, cropTop);

            mCroppedWidth = croppedWidth - Math.Max(0, (cropLeft + croppedWidth) - (mWidth - 2));
            mCroppedHeight = croppedHeight - Math.Max(0, (cropTop + croppedHeight) - (mHeight - 2));

            IFDDNGNoiseProfile noise = rawIFD.GetEntry<IFDDNGNoiseProfile>();
            if (noise == null)
            { 
                noise = mIfds[0].GetEntry<IFDDNGNoiseProfile>();
            }
            if (noise != null)
            {
                //if noise level is given for all channels,
                //take the green one as it is usually scalled to one
                if (noise.Value.Length > 2)
                {
                    mNoiseModelAlpha = (float)noise.Value[planeColorHelper[1] * 2];
                    mNoiseModelBeta = (float)noise.Value[planeColorHelper[1] * 2 + 1];
                }
                else
                {
                    mNoiseModelAlpha = (float)noise.Value[0];
                    mNoiseModelBeta = (float)noise.Value[1];
                }
            }

            mMake = mIfds[0].GetEntry<IFDMake>().Value;
            mUniqueModelName = mIfds[0].GetEntry<IFDDNGUniqueCameraModel>().Value;
            
            return rawIFD;
        }

        public void SaveAsDNG(string aFilename)
        {
            FileStream fs = new FileStream(aFilename, FileMode.Create, FileAccess.Write);
            fs.Write(new byte[] { 0x49, 0x49, 0x2A, 00 }, 0, 4); //Tiff header
            fs.Write(new byte[] { 8, 0, 0, 0 }, 0, 4); //offset to first IFD

            foreach (var item in mIfds)
            {
                item.SavePass1(fs);
            }
            fs.Write(new byte[] { 0, 0, 0, 0 }, 0, 4);
            foreach (var item in mIfds)
            {
                item.SavePass2(fs);
            }
            fs.Seek(0, SeekOrigin.End);
            //uint finalImageOffset = (uint)fs.Position;
            //GetEntry<IFDStripOffsets>().SaveFinalOffsets(fs, new uint[] { finalImageOffset });
            //fs.Seek(finalImageOffset, SeekOrigin.Begin);
            //BinaryWriter br = new BinaryWriter(fs);
            //for (int i = 0; i < data.Length; i++)
            //{
            //    br.Write(data[i]);
            //}
            //br.Close();
            //br.Dispose();
            fs.Close();
        }
    }
}
