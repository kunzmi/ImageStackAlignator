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
	public class PentaxMakerNotesEntry
	{
		#region Const Data
		protected static Dictionary<uint, string> DictPentaxModelIDs = new Dictionary<uint, string>()
		{	{0xd, "Optio 330/430"},
			{0x12926, "Optio 230"},
			{0x12958, "Optio 330GS"},
			{0x12962, "Optio 450/550"},
			{0x1296c, "Optio S"},
			{0x12971, "Optio S V1.01"},
			{0x12994, "*ist D"},
			{0x129b2, "Optio 33L"},
			{0x129bc, "Optio 33LF"},
			{0x129c6, "Optio 33WR/43WR/555"},
			{0x129d5, "Optio S4"},
			{0x12a02, "Optio MX"},
			{0x12a0c, "Optio S40"},
			{0x12a16, "Optio S4i"},
			{0x12a34, "Optio 30"},
			{0x12a52, "Optio S30"},
			{0x12a66, "Optio 750Z"},
			{0x12a70, "Optio SV"},
			{0x12a75, "Optio SVi"},
			{0x12a7a, "Optio x"},
			{0x12a8e, "Optio S5i"},
			{0x12a98, "Optio S50"},
			{0x12aa2, "*ist DS"},
			{0x12ab6, "Optio MX4"},
			{0x12ac0, "Optio S5n"},
			{0x12aca, "Optio WP"},
			{0x12afc, "Optio S55"},
			{0x12b10, "Optio S5z"},
			{0x12b1a, "*ist DL"},
			{0x12b24, "Optio S60"},
			{0x12b2e, "Optio S45"},
			{0x12b38, "Optio S6"},
			{0x12b4c, "Optio WPi"},
			{0x12b56, "BenQ DC X600"},
			{0x12b60, "*ist DS2"},
			{0x12b62, "Samsung GX-1S"},
			{0x12b6a, "Optio A10"},
			{0x12b7e, "*ist DL2"},
			{0x12b80, "Samsung GX-1L"},
			{0x12b9c, "K100D"},
			{0x12b9d, "K110D"},
			{0x12ba2, "K100D Super"},
			{0x12bb0, "Optio T10/T20"},
			{0x12be2, "Optio W10"},
			{0x12bf6, "Optio M10"},
			{0x12c1e, "K10D"},
			{0x12c20, "Samsung GX10"},
			{0x12c28, "Optio S7"},
			{0x12c2d, "Optio L20"},
			{0x12c32, "Optio M20"},
			{0x12c3c, "Optio W20"},
			{0x12c46, "Optio A20"},
			{0x12c78, "Optio E30"},
			{0x12c7d, "Optio E35"},
			{0x12c82, "Optio T30"},
			{0x12c8c, "Optio M30"},
			{0x12c91, "Optio L30"},
			{0x12c96, "Optio W30"},
			{0x12ca0, "Optio A30"},
			{0x12cb4, "Optio E40"},
			{0x12cbe, "Optio M40"},
			{0x12cc3, "Optio L40"},
			{0x12cc5, "Optio L36"},
			{0x12cc8, "Optio Z10"},
			{0x12cd2, "K20D"},
			{0x12cd4, "Samsung GX20"},
			{0x12cdc, "Optio S10"},
			{0x12ce6, "Optio A40"},
			{0x12cf0, "Optio V10"},
			{0x12cfa, "K200D"},
			{0x12d04, "Optio S12"},
			{0x12d0e, "Optio E50"},
			{0x12d18, "Optio M50"},
			{0x12d22, "Optio L50"},
			{0x12d2c, "Optio V20"},
			{0x12d40, "Optio W60"},
			{0x12d4a, "Optio M60"},
			{0x12d68, "Optio E60/M90"},
			{0x12d72, "K2000"},
			{0x12d73, "K-m"},
			{0x12d86, "Optio P70"},
			{0x12d90, "Optio L70"},
			{0x12d9a, "Optio E70"},
			{0x12dae, "X70"},
			{0x12db8, "K-7"},
			{0x12dcc, "Optio W80"},
			{0x12dea, "Optio P80"},
			{0x12df4, "Optio WS80"},
			{0x12dfe, "K-x"},
			{0x12e08, "645D"},
			{0x12e12, "Optio E80"},
			{0x12e30, "Optio W90"},
			{0x12e3a, "Optio I-10"},
			{0x12e44, "Optio H90"},
			{0x12e4e, "Optio E90"},
			{0x12e58, "X90"},
			{0x12e6c, "K-r"},
			{0x12e76, "K-5"},
			{0x12e8a, "Optio RS1000/RS1500"},
			{0x12e94, "Optio RZ10"},
			{0x12e9e, "Optio LS1000"},
			{0x12ebc, "Optio WG-1 GPS"},
			{0x12ed0, "Optio S1"},
			{0x12ee4, "Q"},
			{0x12ef8, "K-1"},
			{0x12f0c, "Optio RZ18"},
			{0x12f16, "Optio VS20"},
			{0x12f2a, "Optio WG-2 GPS"},
			{0x12f48, "Optio LS465"},
			{0x12f52, "K-30"},
			{0x12f5c, "x-5"},
			{0x12f66, "Q10"},
			{0x12f70, "K-5 II"},
			{0x12f71, "K-5 II s"},
			{0x12f7a, "Q7"},
			{0x12f84, "MX-1"},
			{0x12f8e, "WG-3 GPS"},
			{0x12f98, "WG-3"},
			{0x12fa2, "WG-10"},
			{0x12fb6, "K-50"},
			{0x12fc0, "K-3"},
			{0x12fca, "K-500"},
			{0x12fde, "WG-4 GPS"},
			{0x12fe8, "WG-4"},
			{0x13006, "WG-20"},
			{0x13010, "645Z"},
			{0x1301a, "K-S1"},
            {0x13024, "K-S2"},
			{0x1302e, "Q-S1"},
			{0x13056, "WG-30"},
			{0x1307e, "WG-30W"},
			{0x13088, "WG-5 GPS"},
			{0x13092, "K-1"}, 
			{0x1309c, "K-3 II"},
			{0x131f0, "WG-M2"},
			{0x1320e, "GR III"},
			{0x13222, "K-70"},
			{0x1322c, "KP"},
			{0x13240, "K-1 Mark II"}
        };


		protected static Dictionary<uint, string> DictFlashModesA = new Dictionary<uint, string>()
		{	{0x0 ,"Auto, Did not fire"},
			{0x1 ,"Off, Did not fire"}, 
			{0x2 ,"On, Did not fire"}, 
			{0x3 ,"Auto, Did not fire, Red-eye reduction"}, 
			{0x5 ,"On, Did not fire, Wireless (Master)"}, 
			{0x100 ,"Auto, Fired"}, 
			{0x102 ,"On, Fired"}, 
			{0x103 ,"Auto, Fired, Red-eye reduction"},
			{0x104 ,"On, Red-eye reduction"}, 
			{0x105 ,"On, Wireless (Master)"}, 
			{0x106 ,"On, Wireless (Control)"}, 
			{0x108 ,"On, Soft"}, 
			{0x109 ,"On, Slow-sync"}, 
			{0x10a ,"On, Slow-sync, Red-eye reduction"}, 
			{0x10b ,"On, Trailing-curtain Sync"}};
		protected static Dictionary<uint, string> DictFlashModesB = new Dictionary<uint, string>()
		{	{0x0 ,"n/a - Off-Auto-Aperture"}, 
			{0x3f ,"Internal"}, 
			{0x100 ,"External, Auto"}, 
			{0x23f ,"External, Flash Problem"}, 
			{0x300 ,"External, Manual"}, 
			{0x304 ,"External, P-TTL Auto"}, 
			{0x305 ,"External, Contrast-control Sync"}, 
			{0x306 ,"External, High-speed Sync"}, 
			{0x30c ,"External, Wireless"}, 
			{0x30d ,"External, Wireless, High-speed Sync"}};

		protected static Dictionary<uint, string> DictFocusModes = new Dictionary<uint, string>()
		{	{0 ,"Normal"}, 
			{1 ,"Macro"},  
			{2 ,"Infinity"},  
			{3 ,"Manual"},  
			{4 ,"Super Macro"},  
			{5 ,"Pan Focus"},  
			{16 ,"AF-S (Focus-priority)"},  
			{17 ,"AF-C (Focus-priority)"},     
			{18 ,"AF-A (Focus-priority)"},  
			{32 ,"Contrast-detect (Focus-priority)"},  
			{33 ,"Tracking Contrast-detect (Focus-priority)"},  
			{272 ,"AF-S (Release-priority)"},  
			{273 ,"AF-C (Release-priority)"},  
			{274 ,"AF-A (Release-priority)"}, 
			{288 ,"Contrast-detect (Release-priority)"}};


		protected static Dictionary<uint, string> DictAFPointSelectedNonK3 = new Dictionary<uint, string>()
		{	{0 ,"None"},  
			{1 ,"Upper-left"},  
			{2 ,"Top"},  
			{3 ,"Upper-right"},  
			{4 ,"Left"},  
			{5 ,"Mid-left"},  
			{6 ,"Center"},  
			{7 ,"Mid-right"},  
			{8 ,"Right"}, 
			{9 ,"Lower-left"},  
			{10 ,"Bottom"},  
			{11 ,"Lower-right"},  
			{65531 ,"AF Select"},  
			{65532 ,"Face Detect AF"},  
			{65533 ,"Automatic Tracking AF"},  
			{65534 ,"Fixed Center"},  
			{65535 ,"Auto"}};


		protected static Dictionary<uint, string> DictAFPointSelectedK3 = new Dictionary<uint, string>()
		{	{0 ,"None"}, 
			{1 ,"Top-left"}, 
			{2 ,"Top Near-left"}, 
			{3 ,"Top"}, 
			{4 ,"Top Near-right"}, 
			{5 ,"Top-right"}, 
			{6 ,"Upper-left"}, 
			{7 ,"Upper Near-left"},
			{8 ,"Upper-middle"}, 
			{9 ,"Upper Near-right"},
			{10 ,"Upper-right"}, 
			{11 ,"Far Left"},
			{12 ,"Left"}, 
			{13 ,"Near-left"},
			{14 ,"Center"}, 
			{15 ,"Near-right"},
			{16 ,"Right"},
			{17 ,"Far Right"},
			{18 ,"Lower-left"},
			{19 ,"Lower Near-left"}, 
			{20 ,"Lower-middle"},
			{21 ,"Lower Near-right"}, 
			{22 ,"Lower-right"},
			{23 ,"Bottom-left"}, 
			{24 ,"Bottom Near-left"}, 
			{25 ,"Bottom"}, 
			{26 ,"Bottom Near-right"}, 
			{27 ,"Bottom-right"}, 
			{257 ,"Zone Select Top-left"}, 
			{258 ,"Zone Select Top Near-left"},
			{259 ,"Zone Select Top"}, 
			{260 ,"Zone Select Top Near-right"}, 
			{261 ,"Zone Select Top-right"}, 
			{262 ,"Zone Select Upper-left"}, 
			{263 ,"Zone Select Upper Near-left"}, 
			{264 ,"Zone Select Upper-middle"}, 
			{265 ,"Zone Select Upper Near-right"}, 
			{266 ,"Zone Select Upper-right"}, 
			{267 ,"Zone Select Far Left"}, 
			{268 ,"Zone Select Left"}, 
			{269 ,"Zone Select Near-left"}, 
			{270 ,"Zone Select Center"}, 
			{271 ,"Zone Select Near-right"}, 
			{272 ,"Zone Select Right"}, 
			{273 ,"Zone Select Far Right"}, 
			{274 ,"Zone Select Lower-left"}, 
			{275 ,"Zone Select Lower Near-left"}, 
			{276 ,"Zone Select Lower-middle"}, 
			{277 ,"Zone Select Lower Near-right"}, 
			{278 ,"Zone Select Lower-right"}, 
			{279 ,"Zone Select Bottom-left"}, 
			{280 ,"Zone Select Bottom Near-left"}, 
			{281 ,"Zone Select Bottom"}, 
			{282 ,"Zone Select Bottom Near-right"}, 
			{283 ,"Zone Select Bottom-right"}, 
			{65531 ,"AF Select"}, 
			{65532 ,"Face Detect AF"}, 
			{65533 ,"Automatic Tracking AF"}, 
			{65534 ,"Fixed Center"}, 
			{65535 ,"Auto"}};

		protected static Dictionary<uint, string> DictAFPointSelectedValue2 = new Dictionary<uint, string>()
		{	{0 ,"Single Point"}, 
			{1 ,"Expanded Area 9-point (S)"},    
			{3 ,"Expanded Area 25-point (M)"}, 
			{5 ,"Expanded Area 27-point (L)"} };

		protected static Dictionary<ushort, uint> DictISO = new Dictionary<ushort, uint>()
		{	{3, 50}, 
			{4, 64}, 
			{5, 80}, 
			{6, 100}, 
			{7, 125}, 
			{8, 160}, 
			{9, 200}, 
			{10, 250}, 
			{11, 320}, 
			{12, 400}, 
			{13, 500}, 
			{14, 640}, 
			{15, 800}, 
			{16, 1000}, 
			{17, 1250}, 
			{18, 1600}, 
			{19, 2000},    
			{20, 2500}, 
			{21, 3200}, 
			{22, 4000}, 
			{23, 5000}, 
			{24, 6400}, 
			{25, 8000}, 
			{26, 10000}, 
			{27, 12800}, 
			{28, 16000}, 
			{29, 20000}, 
			{30, 25600}, 
			{31, 32000}, 
			{32, 40000}, 
			{33, 51200}, 
			{34, 64000}, 
			{35, 80000}, 
			{36, 102400},    
			{37, 128000}, 
			{38, 160000}, 
			{39, 204800}, 
			{50, 50}, 
			{100, 100}, 
			{200, 200}, 
			{258, 50}, 
			{259, 70}, 
			{260, 100}, 
			{261, 140}, 
			{262, 200}, 
			{263, 280}, 
			{264, 400}, 
			{265, 560}, 
			{266, 800}, 
			{267, 1100}, 
			{268, 1600},    
			{269, 2200}, 
			{270, 3200},
			{271, 4500}, 
			{272, 6400}, 
			{273, 9000}, 
			{274, 12800}, 
			{275, 18000}, 
			{276, 25600}, 
			{277, 36000}, 
			{278, 51200}, 
			{400, 400}, 
			{800, 800}, 
			{1600, 1600}, 
			{3200, 3200}   
		};

		protected static Dictionary<ushort, string> DictCities = new Dictionary<ushort, string>()
		{	{0, "Pago Pago"},
			{1, "Honolulu"},
			{2, "Anchorage"},
			{3, "Vancouver"},
			{4, "San Francisco"},
			{5, "Los Angeles"},
			{6, "Calgary"},
			{7, "Denver"},
			{8, "Mexico City"},
			{9, "Chicago"},
			{10, "Miami"},
			{11, "Toronto"},
			{12, "New York"},
			{13, "Santiago"},
			{14, "Caracus"},
			{15, "Halifax"},
			{16, "Buenos Aires"},
			{17, "Sao Paulo"},
			{18, "Rio de Janeiro"},
			{19, "Madrid"},
			{20, "London"},
			{21, "Paris"},
			{22, "Milan"},
			{23, "Rome"},
			{24, "Berlin"},
			{25, "Johannesburg"},
			{26, "Istanbul"},
			{27, "Cairo"},
			{28, "Jerusalem"},
			{29, "Moscow"},
			{30, "Jeddah"},
			{31, "Tehran"},
			{32, "Dubai"},
			{33, "Karachi"},
			{34, "Kabul"},
			{35, "Male"},
			{36, "Delhi"},
			{37, "Colombo"},
			{38, "Kathmandu"},
			{39, "Dacca"},
			{40, "Yangon"},
			{41, "Bangkok"},
			{42, "Kuala Lumpur"},
			{43, "Vientiane"},
			{44, "Singapore"},
			{45, "Phnom Penh"},
			{46, "Ho Chi Minh"},
			{47, "Jakarta"},
			{48, "Hong Kong"},
			{49, "Perth"},
			{50, "Beijing"},
			{51, "Shanghai"},
			{52, "Manila"},
			{53, "Taipei"},
			{54, "Seoul"},
			{55, "Adelaide"},
			{56, "Tokyo"},
			{57, "Guam"},
			{58, "Sydney"},
			{59, "Noumea"},
			{60, "Wellington"},
			{61, "Auckland"},
			{62, "Lima"},
			{63, "Dakar"},
			{64, "Algiers"},
			{65, "Helsinki"},
			{66, "Athens"},
			{67, "Nairobi"},
			{68, "Amsterdam"},
			{69, "Stockholm"},
			{70, "Lisbon"},
			{71, "Copenhagen"},
			{72, "Warsaw"},
			{73, "Prague"},
			{74, "Budapest"} };


		protected static Dictionary<int, string> DictLenses = new Dictionary<int, string>()
		{	{0, "M-42 or No Lens "},
			{256, "K or M Lens "},
			{512, "A Series Lens "},
			{768, "Sigma "},
			{785, "smc PENTAX-FA SOFT 85mm F2.8 "},
			{786, "smc PENTAX-F 1.7X AF ADAPTER "},
			{787, "smc PENTAX-F 24-50mm F4 "},
			{788, "smc PENTAX-F 35-80mm F4-5.6 "},
			{789, "smc PENTAX-F 80-200mm F4.7-5.6 "},
			{790, "smc PENTAX-F FISH-EYE 17-28mm F3.5-4.5 "},
			{791, "smc PENTAX-F 100-300mm F4.5-5.6 or Sigma Lens "},
			{792, "smc PENTAX-F 35-135mm F3.5-4.5 "},
			{793, "smc PENTAX-F 35-105mm F4-5.6 or Sigma or Tokina Lens "},
			{794, "smc PENTAX-F* 250-600mm F5.6 ED[IF] "},
			{795, "smc PENTAX-F 28-80mm F3.5-4.5 or Tokina Lens "},
			{796, "smc PENTAX-F 35-70mm F3.5-4.5 or Tokina Lens "},
			{797, "PENTAX-F 28-80mm F3.5-4.5 or Sigma or Tokina Lens "},
			{798, "PENTAX-F 70-200mm F4-5.6 "},
			{799, "smc PENTAX-F 70-210mm F4-5.6 or Tokina or Takumar Lens "},
			{800, "smc PENTAX-F 50mm F1.4 "},
			{801, "smc PENTAX-F 50mm F1.7 "},
			{802, "smc PENTAX-F 135mm F2.8 [IF] "},
			{803, "smc PENTAX-F 28mm F2.8 "},
			{804, "Sigma 20mm F1.8 EX DG Aspherical RF "},
			{806, "smc PENTAX-F* 300mm F4.5 ED[IF] "},
			{807, "smc PENTAX-F* 600mm F4 ED[IF] "},
			{808, "smc PENTAX-F Macro 100mm F2.8 "},
			{809, "smc PENTAX-F Macro 50mm F2.8 or Sigma Lens "},
			{812, "Sigma or Tamron Lens (3 44) "},
			{814, "Sigma or Samsung Lens (3 46) "},
			{818, "smc PENTAX-FA 28-70mm F4 AL "},
			{819, "Sigma 28mm F1.8 EX DG Aspherical Macro "},
			{820, "smc PENTAX-FA 28-200mm F3.8-5.6 AL[IF] or Tamron Lens "},
			{821, "smc PENTAX-FA 28-80mm F3.5-5.6 AL "},
			{1015, "smc PENTAX-DA FISH-EYE 10-17mm F3.5-4.5 ED[IF] "},
			{1016, "smc PENTAX-DA 12-24mm F4 ED AL[IF] "},
			{1018, "smc PENTAX-DA 50-200mm F4-5.6 ED "},
			{1019, "smc PENTAX-DA 40mm F2.8 Limited "},
			{1020, "smc PENTAX-DA 18-55mm F3.5-5.6 AL "},
			{1021, "smc PENTAX-DA 14mm F2.8 ED[IF] "},
			{1022, "smc PENTAX-DA 16-45mm F4 ED AL "},
			{1023, "Sigma Lens (3 255) "},
			{1025, "smc PENTAX-FA SOFT 28mm F2.8 "},
			{1026, "smc PENTAX-FA 80-320mm F4.5-5.6 "},
			{1027, "smc PENTAX-FA 43mm F1.9 Limited "},
			{1030, "smc PENTAX-FA 35-80mm F4-5.6 "},
			{1036, "smc PENTAX-FA 50mm F1.4 "},
			{1039, "smc PENTAX-FA 28-105mm F4-5.6 [IF] "},
			{1040, "Tamron AF 80-210mm F4-5.6 (178D) "},
			{1043, "Tamron SP AF 90mm F2.8 (172E) "},
			{1044, "smc PENTAX-FA 28-80mm F3.5-5.6 "},
			{1045, "Cosina AF 100-300mm F5.6-6.7 "},
			{1046, "Tokina 28-80mm F3.5-5.6 "},
			{1047, "smc PENTAX-FA 20-35mm F4 AL "},
			{1048, "smc PENTAX-FA 77mm F1.8 Limited "},
			{1049, "Tamron SP AF 14mm F2.8 "},
			{1050, "smc PENTAX-FA Macro 100mm F3.5 or Cosina Lens "},
			{1051, "Tamron AF 28-300mm F3.5-6.3 LD Aspherical[IF] Macro (185D/285D) "},
			{1052, "smc PENTAX-FA 35mm F2 AL "},
			{1053, "Tamron AF 28-200mm F3.8-5.6 LD Super II Macro (371D) "},
			{1058, "smc PENTAX-FA 24-90mm F3.5-4.5 AL[IF] "},
			{1059, "smc PENTAX-FA 100-300mm F4.7-5.8 "},
			{1060, "Tamron AF 70-300mm F4-5.6 LD Macro 1:2 "},
			{1061, "Tamron SP AF 24-135mm F3.5-5.6 AD AL (190D) "},
			{1062, "smc PENTAX-FA 28-105mm F3.2-4.5 AL[IF] "},
			{1063, "smc PENTAX-FA 31mm F1.8 AL Limited "},
			{1065, "Tamron AF 28-200mm Super Zoom F3.8-5.6 Aspherical XR [IF] Macro (A03) "},
			{1067, "smc PENTAX-FA 28-90mm F3.5-5.6 "},
			{1068, "smc PENTAX-FA J 75-300mm F4.5-5.8 AL "},
			{1069, "Tamron Lens (4 45) "},
			{1070, "smc PENTAX-FA J 28-80mm F3.5-5.6 AL "},
			{1071, "smc PENTAX-FA J 18-35mm F4-5.6 AL "},
			{1073, "Tamron SP AF 28-75mm F2.8 XR Di LD Aspherical [IF] Macro "},
			{1075, "smc PENTAX-D FA 50mm F2.8 Macro "},
			{1076, "smc PENTAX-D FA 100mm F2.8 Macro "},
			{1079, "Samsung/Schneider D-XENOGON 35mm F2 "},
			{1080, "Samsung/Schneider D-XENON 100mm F2.8 Macro "},
			{1099, "Tamron SP AF 70-200mm F2.8 Di LD [IF] Macro (A001) "},
			{1238, "smc PENTAX-DA 35mm F2.4 AL "},
			{1253, "smc PENTAX-DA 18-55mm F3.5-5.6 AL II "},
			{1254, "Tamron SP AF 17-50mm F2.8 XR Di II "},
			{1255, "smc PENTAX-DA 18-250mm F3.5-6.3 ED AL [IF] "},
			{1261, "Samsung/Schneider D-XENOGON 10-17mm F3.5-4.5 "},
			{1263, "Samsung/Schneider D-XENON 12-24mm F4 ED AL [IF] "},
			{1266, "smc PENTAX-DA* 16-50mm F2.8 ED AL [IF] SDM (SDM unused) "},
			{1267, "smc PENTAX-DA 70mm F2.4 Limited "},
			{1268, "smc PENTAX-DA 21mm F3.2 AL Limited "},
			{1269, "Samsung/Schneider D-XENON 50-200mm F4-5.6 "},
			{1270, "Samsung/Schneider D-XENON 18-55mm F3.5-5.6 "},
			{1271, "smc PENTAX-DA FISH-EYE 10-17mm F3.5-4.5 ED[IF] "},
			{1272, "smc PENTAX-DA 12-24mm F4 ED AL [IF] "},
			{1273, "Tamron XR DiII 18-200mm F3.5-6.3 (A14) "},
			{1274, "smc PENTAX-DA 50-200mm F4-5.6 ED "},
			{1275, "smc PENTAX-DA 40mm F2.8 Limited "},
			{1276, "smc PENTAX-DA 18-55mm F3.5-5.6 AL "},
			{1277, "smc PENTAX-DA 14mm F2.8 ED[IF] "},
			{1278, "smc PENTAX-DA 16-45mm F4 ED AL "},
			{1281, "smc PENTAX-FA* 24mm F2 AL[IF] "},
			{1282, "smc PENTAX-FA 28mm F2.8 AL "},
			{1283, "smc PENTAX-FA 50mm F1.7 "},
			{1284, "smc PENTAX-FA 50mm F1.4 "},
			{1285, "smc PENTAX-FA* 600mm F4 ED[IF] "},
			{1286, "smc PENTAX-FA* 300mm F4.5 ED[IF] "},
			{1287, "smc PENTAX-FA 135mm F2.8 [IF] "},
			{1288, "smc PENTAX-FA Macro 50mm F2.8 "},
			{1289, "smc PENTAX-FA Macro 100mm F2.8 "},
			{1290, "smc PENTAX-FA* 85mm F1.4 [IF] "},
			{1291, "smc PENTAX-FA* 200mm F2.8 ED[IF] "},
			{1292, "smc PENTAX-FA 28-80mm F3.5-4.7 "},
			{1293, "smc PENTAX-FA 70-200mm F4-5.6 "},
			{1294, "smc PENTAX-FA* 250-600mm F5.6 ED[IF] "},
			{1295, "smc PENTAX-FA 28-105mm F4-5.6 "},
			{1296, "smc PENTAX-FA 100-300mm F4.5-5.6 "},
			{1378, "smc PENTAX-FA 100-300mm F4.5-5.6 "},
			{1537, "smc PENTAX-FA* 85mm F1.4 [IF] "},
			{1538, "smc PENTAX-FA* 200mm F2.8 ED[IF] "},
			{1539, "smc PENTAX-FA* 300mm F2.8 ED[IF] "},
			{1540, "smc PENTAX-FA* 28-70mm F2.8 AL "},
			{1541, "smc PENTAX-FA* 80-200mm F2.8 ED[IF] "},
			{1542, "smc PENTAX-FA* 28-70mm F2.8 AL "},
			{1543, "smc PENTAX-FA* 80-200mm F2.8 ED[IF] "},
			{1544, "smc PENTAX-FA 28-70mm F4AL "},
			{1545, "smc PENTAX-FA 20mm F2.8 "},
			{1546, "smc PENTAX-FA* 400mm F5.6 ED[IF] "},
			{1549, "smc PENTAX-FA* 400mm F5.6 ED[IF] "},
			{1550, "smc PENTAX-FA* Macro 200mm F4 ED[IF] "},
			{1792, "smc PENTAX-DA 21mm F3.2 AL Limited "},
			{1850, "smc PENTAX-D FA Macro 100mm F2.8 WR "},
			{1867, "Tamron SP AF 70-200mm F2.8 Di LD [IF] Macro (A001) "},
			{1993, "smc Pentax-DA L 50-200mm F4-5.6 ED WR "},
			{1994, "smc PENTAX-DA L 18-55mm F3.5-5.6 AL WR "},
			{1995, "HD PENTAX-DA 55-300mm F4-5.8 ED WR "},
			{1996, "HD PENTAX-DA 15mm F4 ED AL Limited "},
			{1997, "HD PENTAX-DA 35mm F2.8 Macro Limited "},
			{1998, "HD PENTAX-DA 70mm F2.4 Limited "},
			{1999, "HD PENTAX-DA 21mm F3.2 ED AL Limited "},
			{2000, "HD PENTAX-DA 40mm F2.8 Limited "},
			{2004, "smc PENTAX-DA 50mm F1.8 "},
			{2005, "smc PENTAX-DA 40mm F2.8 XS "},
			{2006, "smc PENTAX-DA 35mm F2.4 AL "},
			{2008, "smc PENTAX-DA L 55-300mm F4-5.8 ED "},
			{2009, "smc PENTAX-DA 50-200mm F4-5.6 ED WR "},
			{2010, "smc PENTAX-DA 18-55mm F3.5-5.6 AL WR "},
			{2012, "Tamron SP AF 10-24mm F3.5-4.5 Di II LD Aspherical [IF] "},
			{2013, "smc PENTAX-DA L 50-200mm F4-5.6 ED "},
			{2014, "smc PENTAX-DA L 18-55mm F3.5-5.6 "},
			{2015, "Samsung/Schneider D-XENON 18-55mm F3.5-5.6 II "},
			{2016, "smc PENTAX-DA 15mm F4 ED AL Limited "},
			{2017, "Samsung/Schneider D-XENON 18-250mm F3.5-6.3 "},
			{2018, "smc PENTAX-DA* 55mm F1.4 SDM (SDM unused) "},
			{2019, "smc PENTAX-DA* 60-250mm F4 [IF] SDM (SDM unused) "},
			{2020, "Samsung 16-45mm F4 ED "},
			{2021, "smc PENTAX-DA 18-55mm F3.5-5.6 AL II "},
			{2022, "Tamron AF 17-50mm F2.8 XR Di-II LD (Model A16) "},
			{2023, "smc PENTAX-DA 18-250mm F3.5-6.3 ED AL [IF] "},
			{2025, "smc PENTAX-DA 35mm F2.8 Macro Limited "},
			{2026, "smc PENTAX-DA* 300mm F4 ED [IF] SDM (SDM unused) "},
			{2027, "smc PENTAX-DA* 200mm F2.8 ED [IF] SDM (SDM unused) "},
			{2028, "smc PENTAX-DA 55-300mm F4-5.8 ED "},
			{2030, "Tamron AF 18-250mm F3.5-6.3 Di II LD Aspherical [IF] Macro "},
			{2033, "smc PENTAX-DA* 50-135mm F2.8 ED [IF] SDM (SDM unused) "},
			{2034, "smc PENTAX-DA* 16-50mm F2.8 ED AL [IF] SDM (SDM unused) "},
			{2035, "smc PENTAX-DA 70mm F2.4 Limited "},
			{2036, "smc PENTAX-DA 21mm F3.2 AL Limited "},
			{2048, "Sigma 50-150mm F2.8 II APO EX DC HSM "},
			{2051, "Sigma AF 18-125mm F3.5-5.6 DC "},
			{2052, "Sigma 50mm F1.4 EX DG HSM "},
			{2055, "Sigma 24-70mm F2.8 IF EX DG HSM "},
			{2056, "Sigma 18-250mm F3.5-6.3 DC OS HSM "},
			{2059, "Sigma 10-20mm F3.5 EX DC HSM "},
			{2060, "Sigma 70-300mm F4-5.6 DG OS "},
			{2061, "Sigma 120-400mm F4.5-5.6 APO DG OS HSM "},
			{2062, "Sigma 17-70mm F2.8-4.0 DC Macro OS HSM "},
			{2063, "Sigma 150-500mm F5-6.3 APO DG OS HSM "},
			{2064, "Sigma 70-200mm F2.8 EX DG Macro HSM II "},
			{2065, "Sigma 50-500mm F4.5-6.3 DG OS HSM "},
			{2066, "Sigma 8-16mm F4.5-5.6 DC HSM "},
			{2069, "Sigma 17-50mm F2.8 EX DC OS HSM "},
			{2070, "Sigma 85mm F1.4 EX DG HSM "},
			{2071, "Sigma 70-200mm F2.8 APO EX DG OS HSM "},
			{2073, "Sigma 17-50mm F2.8 EX DC HSM "},
			{2075, "Sigma 18-200mm F3.5-6.3 II DC HSM "},
			{2076, "Sigma 18-250mm F3.5-6.3 DC Macro HSM "},
			{2077, "Sigma 35mm F1.4 DG HSM "},
			{2078, "Sigma 17-70mm F2.8-4 DC Macro HSM Contemporary "},
			{2079, "Sigma 18-35mm F1.8 DC HSM "},
			{2080, "Sigma 30mm F1.4 DC HSM | A "},
			{2257, "HD PENTAX-DA 20-40mm F2.8-4 ED Limited DC WR "},
			{2258, "smc PENTAX-DA 18-270mm F3.5-6.3 ED SDM "},
			{2259, "HD PENTAX-DA 560mm F5.6 ED AW "},
			{2263, "smc PENTAX-DA 18-135mm F3.5-5.6 ED AL [IF] DC WR "},
			{2274, "smc PENTAX-DA* 55mm F1.4 SDM "},
			{2275, "smc PENTAX-DA* 60-250mm F4 [IF] SDM "},
			{2280, "smc PENTAX-DA 17-70mm F4 AL [IF] SDM "},
			{2282, "smc PENTAX-DA* 300mm F4 ED [IF] SDM "},
			{2283, "smc PENTAX-DA* 200mm F2.8 ED [IF] SDM "},
			{2289, "smc PENTAX-DA* 50-135mm F2.8 ED [IF] SDM "},
			{2290, "smc PENTAX-DA* 16-50mm F2.8 ED AL [IF] SDM "},
			{2303, "Sigma Lens (8 255) "},
			{2304, "645 Manual Lens "},
			{2560, "645 A Series Lens "},
			{2817, "smc PENTAX-FA 645 75mm F2.8 "},
			{2818, "smc PENTAX-FA 645 45mm F2.8 "},
			{2819, "smc PENTAX-FA* 645 300mm F4 ED [IF] "},
			{2820, "smc PENTAX-FA 645 45-85mm F4.5 "},
			{2821, "smc PENTAX-FA 645 400mm F5.6 ED [IF] "},
			{2823, "smc PENTAX-FA 645 Macro 120mm F4 "},
			{2824, "smc PENTAX-FA 645 80-160mm F4.5 "},
			{2825, "smc PENTAX-FA 645 200mm F4 [IF] "},
			{2826, "smc PENTAX-FA 645 150mm F2.8 [IF] "},
			{2827, "smc PENTAX-FA 645 35mm F3.5 AL [IF] "},
			{2828, "smc PENTAX-FA 645 300mm F5.6 ED [IF] "},
			{2830, "smc PENTAX-FA 645 55-110mm F5.6 "},
			{2832, "smc PENTAX-FA 645 33-55mm F4.5 AL "},
			{2833, "smc PENTAX-FA 645 150-300mm F5.6 ED [IF] "},
			{3346, "smc PENTAX-D FA 645 55mm F2.8 AL [IF] SDM AW "},
			{3347, "smc PENTAX-D FA 645 25mm F4 AL [IF] SDM AW "},
			{3348, "HD PENTAX-D FA 645 90mm F2.8 ED AW SR "},
			{3581, "HD PENTAX-DA 645 28-45mm F4.5 ED AW SR "},
			{5376, "Pentax Q Manual Lens "},
			{5377, "01 Standard Prime 8.5mm F1.9 "},
			{5378, "02 Standard Zoom 5-15mm F2.8-4.5 "},
			{5382, "06 Telephoto Zoom 15-45mm F2.8 "},
			{5383, "07 Mount Shield 11.5mm F9 "},
			{5384, "08 Wide Zoom 3.8-5.9mm F3.7-4 "},
			{5635, "03 Fish-eye 3.2mm F5.6 "},
			{5636, "04 Toy Lens Wide 6.3mm F7.1 "},
			{5637, "05 Toy Lens Telephoto 18mm F8 "}
		};
		#endregion

		protected PentaxMakerNotes mMakerNotes;
		protected ushort mTagID;
		protected TIFFValueType mFieldType;
		protected uint mValueCount;
		protected uint mOffset;

		public PentaxMakerNotesEntry(PentaxMakerNotes aMakerNotes)
		{
			mMakerNotes = aMakerNotes;
			mTagID = mMakerNotes.ReadUI2();
			mFieldType = new TIFFValueType((TIFFValueTypes)mMakerNotes.ReadUI2());
			mValueCount = mMakerNotes.ReadUI4();
			mOffset = mMakerNotes.ReadUI4();
		}

		protected PentaxMakerNotesEntry(PentaxMakerNotes aMakerNotes, ushort aTagID)
		{
			mMakerNotes = aMakerNotes;
			mTagID = aTagID;
            mFieldType = new TIFFValueType((TIFFValueTypes)mMakerNotes.ReadUI2());
			mValueCount = mMakerNotes.ReadUI4();
			mOffset = mMakerNotes.ReadUI4();
		}

		public static PentaxMakerNotesEntry CreatePentaxMakerNotesEntry(PentaxMakerNotes aMakerNotes)
		{
			ushort tagID = aMakerNotes.ReadUI2();

			switch (tagID)
			{
				case MNPentaxVersion.TagID:
					return new MNPentaxVersion(aMakerNotes, tagID);
				case MNPentaxModelType.TagID:
					return new MNPentaxModelType(aMakerNotes, tagID);
				case MNPreviewImageSize.TagID:
					return new MNPreviewImageSize(aMakerNotes, tagID);
				case MNPreviewImageLength.TagID:
					return new MNPreviewImageLength(aMakerNotes, tagID);
				case MNPreviewImageStart.TagID:
					return new MNPreviewImageStart(aMakerNotes, tagID);
				case MNPentaxModelID.TagID:
					return new MNPentaxModelID(aMakerNotes, tagID);
				case MNDate.TagID:
					return new MNDate(aMakerNotes, tagID);
				case MNTime.TagID:
					return new MNTime(aMakerNotes, tagID);
				case MNQuality.TagID:
					return new MNQuality(aMakerNotes, tagID);
				case MNPentaxImageSize.TagID:
					return new MNPentaxImageSize(aMakerNotes, tagID);
				case MNFlashMode.TagID:
					return new MNFlashMode(aMakerNotes, tagID);
				case MNFocusMode.TagID:
					return new MNFocusMode(aMakerNotes, tagID);
				case MNAFPointSelected.TagID:
					return new MNAFPointSelected(aMakerNotes, tagID);
				case MNAFPointsInFocus.TagID:
					return new MNAFPointsInFocus(aMakerNotes, tagID);
				case MNFocusPosition.TagID:
					return new MNFocusPosition(aMakerNotes, tagID);
				case MNExposureTime.TagID:
					return new MNExposureTime(aMakerNotes, tagID);
				case MNFNumber.TagID:
					return new MNFNumber(aMakerNotes, tagID);
				case MNISO.TagID:
					return new MNISO(aMakerNotes, tagID);
				case MNLightReading.TagID:
					return new MNLightReading(aMakerNotes, tagID);
				case MNExposureCompensation.TagID:
					return new MNExposureCompensation(aMakerNotes, tagID);
				case MNMeteringMode.TagID:
					return new MNMeteringMode(aMakerNotes, tagID);
				case MNAutoBracketing.TagID:
					return new MNAutoBracketing(aMakerNotes, tagID);
				case MNWhiteBalance.TagID:
					return new MNWhiteBalance(aMakerNotes, tagID);
				case MNWhiteBalanceMode.TagID:
					return new MNWhiteBalanceMode(aMakerNotes, tagID);
				case MNBlueBalance.TagID:
					return new MNBlueBalance(aMakerNotes, tagID);
				case MNRedBalance.TagID:
					return new MNRedBalance(aMakerNotes, tagID);
				case MNFocalLength.TagID:
					return new MNFocalLength(aMakerNotes, tagID);
				case MNDigitalZoom.TagID:
					return new MNDigitalZoom(aMakerNotes, tagID);
				case MNSaturation.TagID:
					return new MNSaturation(aMakerNotes, tagID);
				case MNContrast.TagID:
					return new MNContrast(aMakerNotes, tagID);
				case MNSharpness.TagID:
					return new MNSharpness(aMakerNotes, tagID);
				case MNWorldTimeLocation.TagID:
					return new MNWorldTimeLocation(aMakerNotes, tagID);
				case MNHometownCity.TagID:
					return new MNHometownCity(aMakerNotes, tagID);
				case MNDestinationCity.TagID:
					return new MNDestinationCity(aMakerNotes, tagID);
				case MNHometownDST.TagID:
					return new MNHometownDST(aMakerNotes, tagID);
				case MNDestinationDST.TagID:
					return new MNDestinationDST(aMakerNotes, tagID);
				case MNDSPFirmwareVersion.TagID:
					return new MNDSPFirmwareVersion(aMakerNotes, tagID);
				case MNCPUFirmwareVersion.TagID:
					return new MNCPUFirmwareVersion(aMakerNotes, tagID);
				case MNFrameNumber.TagID:
					return new MNFrameNumber(aMakerNotes, tagID);
				case MNEffectiveLV.TagID:
					return new MNEffectiveLV(aMakerNotes, tagID);
				case MNImageEditing.TagID:
					return new MNImageEditing(aMakerNotes, tagID);
				case MNPictureMode.TagID:
					return new MNPictureMode(aMakerNotes, tagID);
				case MNDriveMode.TagID:
					return new MNDriveMode(aMakerNotes, tagID);
				case MNSensorSize.TagID:
					return new MNSensorSize(aMakerNotes, tagID);
				case MNColorSpace.TagID:
					return new MNColorSpace(aMakerNotes, tagID);
				case MNImageAreaOffset.TagID:
					return new MNImageAreaOffset(aMakerNotes, tagID);
				case MNRawImageSize.TagID:
					return new MNRawImageSize(aMakerNotes, tagID);
				case MNDataScaling.TagID:
					return new MNDataScaling(aMakerNotes, tagID);
				case MNPreviewImageBorders.TagID:
					return new MNPreviewImageBorders(aMakerNotes, tagID);
				case MNLensRec.TagID:
					return new MNLensRec(aMakerNotes, tagID);
				case MNSensitivityAdjust.TagID:
					return new MNSensitivityAdjust(aMakerNotes, tagID);
				case MNImageEditCount.TagID:
					return new MNImageEditCount(aMakerNotes, tagID);
				case MNCameraTemperature.TagID:
					return new MNCameraTemperature(aMakerNotes, tagID);
				case MNAELock.TagID:
					return new MNAELock(aMakerNotes, tagID);
				case MNNoiseReduction.TagID:
					return new MNNoiseReduction(aMakerNotes, tagID);
				case MNFlashExposureComp.TagID:
					return new MNFlashExposureComp(aMakerNotes, tagID);
				case MNImageTone.TagID:
					return new MNImageTone(aMakerNotes, tagID);
				case MNColorTemperature.TagID:
					return new MNColorTemperature(aMakerNotes, tagID);
				case MNColorTempDaylight.TagID:
					return new MNColorTempDaylight(aMakerNotes, tagID);
				case MNColorTempShade.TagID:
					return new MNColorTempShade(aMakerNotes, tagID);
				case MNColorTempCloudy.TagID:
					return new MNColorTempCloudy(aMakerNotes, tagID);
				case MNColorTempTungsten.TagID:
					return new MNColorTempTungsten(aMakerNotes, tagID);
				case MNColorTempFluorescentD.TagID:
					return new MNColorTempFluorescentD(aMakerNotes, tagID);
				case MNColorTempFluorescentN.TagID:
					return new MNColorTempFluorescentN(aMakerNotes, tagID);
				case MNColorTempFluorescentW.TagID:
					return new MNColorTempFluorescentW(aMakerNotes, tagID);
				case MNColorTempFlash.TagID:
					return new MNColorTempFlash(aMakerNotes, tagID);
				case MNShakeReductionInfo.TagID:
					return new MNShakeReductionInfo(aMakerNotes, tagID);
				case MNShutterCount.TagID:
					return new MNShutterCount(aMakerNotes, tagID);
				case MNFaceInfo.TagID:
					return new MNFaceInfo(aMakerNotes, tagID);
				case MNRawDevelopmentProcess.TagID:
					return new MNRawDevelopmentProcess(aMakerNotes, tagID);
				case MNHue.TagID:
					return new MNHue(aMakerNotes, tagID);
				case MNAWBInfo.TagID:
					return new MNAWBInfo(aMakerNotes, tagID);
				case MNDynamicRangeExpansion.TagID:
					return new MNDynamicRangeExpansion(aMakerNotes, tagID);
				case MNTimeInfo.TagID:
					return new MNTimeInfo(aMakerNotes, tagID);
				case MNHighLowKeyAdj.TagID:
					return new MNHighLowKeyAdj(aMakerNotes, tagID);
				case MNContrastHighlight.TagID:
					return new MNContrastHighlight(aMakerNotes, tagID);
				case MNContrastShadow.TagID:
					return new MNContrastShadow(aMakerNotes, tagID);
				case MNContrastHighlightShadowAdj.TagID:
					return new MNContrastHighlightShadowAdj(aMakerNotes, tagID);
				case MNFineSharpness.TagID:
					return new MNFineSharpness(aMakerNotes, tagID);
				case MNHighISONoiseReduction.TagID:
					return new MNHighISONoiseReduction(aMakerNotes, tagID);
				case MNAFAdjustment.TagID:
					return new MNAFAdjustment(aMakerNotes, tagID);
				case MNMonochromeFilterEffect.TagID:
					return new MNMonochromeFilterEffect(aMakerNotes, tagID);
				case MNMonochromeToning.TagID:
					return new MNMonochromeToning(aMakerNotes, tagID);
				case MNFaceDetect.TagID:
					return new MNFaceDetect(aMakerNotes, tagID);
				case MNFaceDetectFrameSize.TagID:
					return new MNFaceDetectFrameSize(aMakerNotes, tagID);
				case MNShadowCorrection.TagID:
					return new MNShadowCorrection(aMakerNotes, tagID);
				case MNISOAutoParameters.TagID:
					return new MNISOAutoParameters(aMakerNotes, tagID);
				case MNCrossProcess.TagID:
					return new MNCrossProcess(aMakerNotes, tagID);
				case MNLensCorr.TagID:
					return new MNLensCorr(aMakerNotes, tagID);
				case MNWhiteLevel.TagID:
					return new MNWhiteLevel(aMakerNotes, tagID);
				case MNBleachBypassToning.TagID:
					return new MNBleachBypassToning(aMakerNotes, tagID);
				case MNAspectRatio.TagID:
					return new MNAspectRatio(aMakerNotes, tagID);
				case MNBlurControl.TagID:
					return new MNBlurControl(aMakerNotes, tagID);
				case MNHDR.TagID:
					return new MNHDR(aMakerNotes, tagID);
				case MNNeutralDensityFilter.TagID:
					return new MNNeutralDensityFilter(aMakerNotes, tagID);
				case MNISO2.TagID:
					return new MNISO2(aMakerNotes, tagID);
				case MNBlackPoint.TagID:
					return new MNBlackPoint(aMakerNotes, tagID);
				case MNWhitePoint.TagID:
					return new MNWhitePoint(aMakerNotes, tagID);
				case MNColorMatrixA.TagID:
					return new MNColorMatrixA(aMakerNotes, tagID);
				case MNColorMatrixB.TagID:
					return new MNColorMatrixB(aMakerNotes, tagID);
				case MNCameraSettings.TagID:
					return new MNCameraSettings(aMakerNotes, tagID);
				case MNAEInfo.TagID:
					return new MNAEInfo(aMakerNotes, tagID);
				case MNLensInfo.TagID:
					return new MNLensInfo(aMakerNotes, tagID);
				case MNFlashInfo.TagID:
					return new MNFlashInfo(aMakerNotes, tagID);
				case MNAEMeteringSegments.TagID:
					return new MNAEMeteringSegments(aMakerNotes, tagID);
				case MNFlashMeteringSegments.TagID:
					return new MNFlashMeteringSegments(aMakerNotes, tagID);
				case MNSlaveFlashMeteringSegments.TagID:
					return new MNSlaveFlashMeteringSegments(aMakerNotes, tagID);
				case MNWB_RGGBLevelsDaylight.TagID:
					return new MNWB_RGGBLevelsDaylight(aMakerNotes, tagID);
				case MNWB_RGGBLevelsShade.TagID:
					return new MNWB_RGGBLevelsShade(aMakerNotes, tagID);
				case MNWB_RGGBLevelsCloudy.TagID:
					return new MNWB_RGGBLevelsCloudy(aMakerNotes, tagID);
				case MNWB_RGGBLevelsTungsten.TagID:
					return new MNWB_RGGBLevelsTungsten(aMakerNotes, tagID);
				case MNWB_RGGBLevelsFluorescentD.TagID:
					return new MNWB_RGGBLevelsFluorescentD(aMakerNotes, tagID);
				case MNWB_RGGBLevelsFluorescentN.TagID:
					return new MNWB_RGGBLevelsFluorescentN(aMakerNotes, tagID);
				case MNWB_RGGBLevelsFluorescentW.TagID:
					return new MNWB_RGGBLevelsFluorescentW(aMakerNotes, tagID);
				case MNWB_RGGBLevelsFlash.TagID:
					return new MNWB_RGGBLevelsFlash(aMakerNotes, tagID);
				case MNCameraInfo.TagID:
					return new MNCameraInfo(aMakerNotes, tagID);
				case MNBatteryInfo.TagID:
					return new MNBatteryInfo(aMakerNotes, tagID);
				case MNHuffmanTable.TagID:
					return new MNHuffmanTable(aMakerNotes, tagID);
				case MNTempInfo.TagID:
					return new MNTempInfo(aMakerNotes, tagID);
				case MNLevelInfo.TagID:
					return new MNLevelInfo(aMakerNotes, tagID);
				default:
					return new PentaxMakerNotesEntry(aMakerNotes, tagID);
			}
		}
	}
	#region Typed base classes
	public class StringPentaxMakerNotesEntry : PentaxMakerNotesEntry
	{
		protected string mValue;

		public StringPentaxMakerNotesEntry(PentaxMakerNotes aMakerNotes, ushort aTagID)
			: base(aMakerNotes, aTagID)
		{
			if (mFieldType.SizeInBytes * mValueCount <= 4)
			{
				unsafe
				{
					fixed (uint* ptr = &mOffset)
					{
						byte* bptr = (byte*)ptr;
						byte[] text = new byte[4];

						for (int i = 0; i < mValueCount; i++)
						{
							if (aMakerNotes.EndianSwap)
								text[i] = bptr[4 / mFieldType.SizeInBytes - i - 1];
							else
								text[i] = bptr[i];
						}

						mValue = ASCIIEncoding.ASCII.GetString(text).Replace("\0", string.Empty);
					}
				}
			}
			else
			{
				uint currentOffset = mMakerNotes.Position();
				mMakerNotes.SeekWithOffset(mOffset, System.IO.SeekOrigin.Begin);

				mValue = mMakerNotes.ReadStr((int)mFieldType.SizeInBytes * (int)mValueCount).Replace("\0", string.Empty);

				mMakerNotes.Seek(currentOffset, System.IO.SeekOrigin.Begin);
			}
			mValue = mValue.TrimEnd(' ');
		}
	}

	public class BytePentaxMakerNotesEntry : PentaxMakerNotesEntry
	{
		protected byte[] mValue;

		public BytePentaxMakerNotesEntry(PentaxMakerNotes aMakerNotes, ushort aTagID)
			: base(aMakerNotes, aTagID)
		{
			if (mFieldType.SizeInBytes * mValueCount <= 4)
			{
				unsafe
				{
					fixed (uint* ptr = &mOffset)
					{
						byte* ptrUS = (byte*)ptr;
						mValue = new byte[mValueCount];
						for (int i = 0; i < mValueCount; i++)
						{
							if (aMakerNotes.EndianSwap)
								mValue[i] = ptrUS[4 / mFieldType.SizeInBytes - i - 1];
							else
								mValue[i] = ptrUS[i];
						}
					}
				}
			}
			else
			{
				uint currentOffset = mMakerNotes.Position();
				mMakerNotes.SeekWithOffset(mOffset, System.IO.SeekOrigin.Begin);

				mValue = new byte[mValueCount];
				for (int i = 0; i < mValueCount; i++)
				{
					mValue[i] = mMakerNotes.ReadUI1();
				}
				mMakerNotes.Seek(currentOffset, System.IO.SeekOrigin.Begin);
			}
		}
	}

	public class SBytePentaxMakerNotesEntry : PentaxMakerNotesEntry
	{
		protected sbyte[] mValue;

		public SBytePentaxMakerNotesEntry(PentaxMakerNotes aMakerNotes, ushort aTagID)
			: base(aMakerNotes, aTagID)
		{
			if (mFieldType.SizeInBytes * mValueCount <= 4)
			{
				unsafe
				{
					fixed (uint* ptr = &mOffset)
					{
						sbyte* ptrUS = (sbyte*)ptr;
						mValue = new sbyte[mValueCount];
						for (int i = 0; i < mValueCount; i++)
						{
							if (aMakerNotes.EndianSwap)
								mValue[i] = ptrUS[4 / mFieldType.SizeInBytes - i - 1];
							else
								mValue[i] = ptrUS[i];
						}
					}
				}
			}
			else
			{
				uint currentOffset = mMakerNotes.Position();
				mMakerNotes.SeekWithOffset(mOffset, System.IO.SeekOrigin.Begin);

				mValue = new sbyte[mValueCount];
				for (int i = 0; i < mValueCount; i++)
				{
					mValue[i] = mMakerNotes.ReadI1();
				}
				mMakerNotes.Seek(currentOffset, System.IO.SeekOrigin.Begin);
			}
		}
	}

	public class UShortPentaxMakerNotesEntry : PentaxMakerNotesEntry
	{
		protected ushort[] mValue;

		public UShortPentaxMakerNotesEntry(PentaxMakerNotes aMakerNotes, ushort aTagID)
			: base(aMakerNotes, aTagID)
		{
			if (mFieldType.SizeInBytes * mValueCount <= 4)
			{
				unsafe
				{
					fixed (uint* ptr = &mOffset)
					{
						ushort* ptrUS = (ushort*)ptr;
						mValue = new ushort[mValueCount];
						for (int i = 0; i < mValueCount; i++)
						{
							if (aMakerNotes.EndianSwap)
								mValue[i] = ptrUS[4 / mFieldType.SizeInBytes - i - 1];
							else
								mValue[i] = ptrUS[i];
						}
					}
				}
			}
			else
			{
				uint currentOffset = mMakerNotes.Position();
				mMakerNotes.SeekWithOffset(mOffset, System.IO.SeekOrigin.Begin);

				mValue = new ushort[mValueCount];
				for (int i = 0; i < mValueCount; i++)
				{
					mValue[i] = mMakerNotes.ReadUI2();
				}
				mMakerNotes.Seek(currentOffset, System.IO.SeekOrigin.Begin);
			}
		}
	}

	public class ShortPentaxMakerNotesEntry : PentaxMakerNotesEntry
	{
		protected short[] mValue;

		public ShortPentaxMakerNotesEntry(PentaxMakerNotes aMakerNotes, ushort aTagID)
			: base(aMakerNotes, aTagID)
		{
			if (mFieldType.SizeInBytes * mValueCount <= 4)
			{
				unsafe
				{
					fixed (uint* ptr = &mOffset)
					{
						short* ptrUS = (short*)ptr;
						mValue = new short[mValueCount];
						for (int i = 0; i < mValueCount; i++)
						{
							if (aMakerNotes.EndianSwap)
								mValue[i] = ptrUS[4 / mFieldType.SizeInBytes - i - 1];
							else
								mValue[i] = ptrUS[i];
						}
					}
				}
			}
			else
			{
				uint currentOffset = mMakerNotes.Position();
				mMakerNotes.SeekWithOffset(mOffset, System.IO.SeekOrigin.Begin);

				mValue = new short[mValueCount];
				for (int i = 0; i < mValueCount; i++)
				{
					mValue[i] = mMakerNotes.ReadI2();
				}
				mMakerNotes.Seek(currentOffset, System.IO.SeekOrigin.Begin);
			}
		}
	}

	public class IntPentaxMakerNotesEntry : PentaxMakerNotesEntry
	{
		protected int[] mValue;

		public IntPentaxMakerNotesEntry(PentaxMakerNotes aMakerNotes, ushort aTagID)
			: base(aMakerNotes, aTagID)
		{
			if (mFieldType.SizeInBytes * mValueCount <= 4)
			{
				unsafe
				{
					fixed (uint* ptr = &mOffset)
					{
						int* ptrUS = (int*)ptr;
						mValue = new int[mValueCount];
						for (int i = 0; i < mValueCount; i++)
						{
							if (aMakerNotes.EndianSwap)
								mValue[i] = ptrUS[4 / mFieldType.SizeInBytes - i - 1];
							else
								mValue[i] = ptrUS[i];
						}
					}
				}
			}
			else
			{
				uint currentOffset = mMakerNotes.Position();
				mMakerNotes.SeekWithOffset(mOffset, System.IO.SeekOrigin.Begin);

				mValue = new int[mValueCount];
				for (int i = 0; i < mValueCount; i++)
				{
					mValue[i] = mMakerNotes.ReadI4();
				}
				mMakerNotes.Seek(currentOffset, System.IO.SeekOrigin.Begin);
			}
		}
	}

	public class UIntPentaxMakerNotesEntry : PentaxMakerNotesEntry
	{
		protected uint[] mValue;

		public UIntPentaxMakerNotesEntry(PentaxMakerNotes aMakerNotes, ushort aTagID)
			: base(aMakerNotes, aTagID)
		{
			if (mFieldType.SizeInBytes * mValueCount <= 4)
			{
				mValue = new uint[mValueCount];
				mValue[0] = mOffset;
			}
			else
			{
				uint currentOffset = mMakerNotes.Position();
				mMakerNotes.SeekWithOffset(mOffset, System.IO.SeekOrigin.Begin);

				mValue = new uint[mValueCount];
				for (int i = 0; i < mValueCount; i++)
				{
					mValue[i] = mMakerNotes.ReadUI4();
				}
				mMakerNotes.Seek(currentOffset, System.IO.SeekOrigin.Begin);
			}
		}
	}

	public class RationalPentaxMakerNotesEntry : PentaxMakerNotesEntry
	{
		protected Rational[] mValue;

		public RationalPentaxMakerNotesEntry(PentaxMakerNotes aMakerNotes, ushort aTagID)
			: base(aMakerNotes, aTagID)
		{
			uint currentOffset = mMakerNotes.Position();
			mMakerNotes.SeekWithOffset(mOffset, System.IO.SeekOrigin.Begin);

			mValue = new Rational[mValueCount];
			for (int i = 0; i < mValueCount; i++)
			{
				uint tempNom = mMakerNotes.ReadUI4();
				uint tempDenom = mMakerNotes.ReadUI4();
				mValue[i] = new Rational(tempNom, tempDenom);
			}
			mMakerNotes.Seek(currentOffset, System.IO.SeekOrigin.Begin);
		}
	}

	public class SRationalPentaxMakerNotesEntry : PentaxMakerNotesEntry
	{
		protected SRational[] mValue;

		public SRationalPentaxMakerNotesEntry(PentaxMakerNotes aMakerNotes, ushort aTagID)
			: base(aMakerNotes, aTagID)
		{
			uint currentOffset = mMakerNotes.Position();
			mMakerNotes.SeekWithOffset(mOffset, System.IO.SeekOrigin.Begin);

			mValue = new SRational[mValueCount];
			for (int i = 0; i < mValueCount; i++)
			{
				int tempNom = mMakerNotes.ReadI4();
				int tempDenom = mMakerNotes.ReadI4();
				mValue[i] = new SRational(tempNom, tempDenom);
			}
			mMakerNotes.Seek(currentOffset, System.IO.SeekOrigin.Begin);
		}
	}

	public class FloatPentaxMakerNotesEntry : PentaxMakerNotesEntry
	{
		protected float[] mValue;

		public FloatPentaxMakerNotesEntry(PentaxMakerNotes aMakerNotes, ushort aTagID)
			: base(aMakerNotes, aTagID)
		{
			if (mFieldType.SizeInBytes * mValueCount <= 4)
			{
				unsafe
				{
					fixed (uint* ptr = &mOffset)
					{
						float* ptrUS = (float*)ptr;
						mValue = new float[mValueCount];
						for (int i = 0; i < mValueCount; i++)
						{
							if (aMakerNotes.EndianSwap)
								mValue[i] = ptrUS[4 / mFieldType.SizeInBytes - i - 1];
							else
								mValue[i] = ptrUS[i];
						}
					}
				}
			}
			else
			{
				uint currentOffset = mMakerNotes.Position();
				mMakerNotes.SeekWithOffset(mOffset, System.IO.SeekOrigin.Begin);

				mValue = new float[mValueCount];
				for (int i = 0; i < mValueCount; i++)
				{
					mValue[i] = mMakerNotes.ReadF4();
				}
				mMakerNotes.Seek(currentOffset, System.IO.SeekOrigin.Begin);
			}
		}
	}

	public class DoublePentaxMakerNotesEntry : PentaxMakerNotesEntry
	{
		protected double[] mValue;

		public DoublePentaxMakerNotesEntry(PentaxMakerNotes aMakerNotes, ushort aTagID)
			: base(aMakerNotes, aTagID)
		{
			uint currentOffset = mMakerNotes.Position();
			mMakerNotes.SeekWithOffset(mOffset, System.IO.SeekOrigin.Begin);

			mValue = new double[mValueCount];
			for (int i = 0; i < mValueCount; i++)
			{
				mValue[i] = mMakerNotes.ReadF8();
			}
			mMakerNotes.Seek(currentOffset, System.IO.SeekOrigin.Begin);
		}
	}
	#endregion

	public class MNPentaxVersion : BytePentaxMakerNotesEntry
	{
		Version v_Value;

		public MNPentaxVersion(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			v_Value = new Version(mValue[0], mValue[1], mValue[2], mValue[3]);
		}

		public const ushort TagID = 0x0000;

		public const string TagName = "Pentax version";

		public Version Value
		{
			get { return v_Value; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNPentaxModelType : UShortPentaxMakerNotesEntry
	{
		public MNPentaxModelType(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0001;

		public const string TagName = "Pentax model type";

		public ushort Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNPreviewImageSize : UShortPentaxMakerNotesEntry
	{
		public MNPreviewImageSize(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0002;

		public const string TagName = "Preview image size";

		public ushort[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value[0].ToString() + " x " + Value[1].ToString();
		}
	}

	public class MNPreviewImageLength : UIntPentaxMakerNotesEntry
	{
		public MNPreviewImageLength(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0003;

		public const string TagName = "Preview image length";

		public uint Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNPreviewImageStart : UIntPentaxMakerNotesEntry
	{
		public MNPreviewImageStart(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0004;

		public const string TagName = "Preview image start";

		public uint Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNPentaxModelID : UIntPentaxMakerNotesEntry
	{
		public MNPentaxModelID(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
            if (DictPentaxModelIDs.ContainsKey(mValue[0]))
                aPefFile.K3Specific = DictPentaxModelIDs[mValue[0]] == "K-3";
            else
                aPefFile.K3Specific = false;
		}

		public const ushort TagID = 0x0005;

		public const string TagName = "Pentax Model ID";

		public string Value
        {
			get 
            {
                if (DictPentaxModelIDs.ContainsKey(mValue[0]))
                    return DictPentaxModelIDs[mValue[0]];
                else
                    return mValue[0].ToString(); 
            }
		}

		public override string ToString()
		{
			return TagName + ": " + Value;
		}
	}

	public class MNDate : BytePentaxMakerNotesEntry
	{
		DateTime dt_value;

		public MNDate(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			int year = (mValue[0] << 8) + mValue[1];
			int month = mValue[2];
			int day = mValue[3];
			dt_value = new DateTime(year, month, day);
		}

		public const ushort TagID = 0x0006;

		public const string TagName = "Date";

		public DateTime Value
		{
			get { return dt_value; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString("D");
		}
	}

	public class MNTime : BytePentaxMakerNotesEntry
	{
		TimeSpan ts_value;

		public MNTime(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			int hour = mValue[0];
			int min = mValue[1];
			int sec = mValue[2];
			ts_value = new TimeSpan(hour, min, sec);
		}

		public const ushort TagID = 0x0007;

		public const string TagName = "Time";

		public TimeSpan Value
		{
			get { return ts_value; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNQuality : UShortPentaxMakerNotesEntry
	{
		public enum Quality
		{ 
			Good = 0,
			Better = 1,
			Best = 2,
			TIFF = 3,
			RAW = 4,
			Premium = 5,
			NA = 65535
		}

		public MNQuality(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0008;

		public const string TagName = "Quality";

		public Quality Value
		{
			get { return (Quality)mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value;
		}
	}

	public class MNPentaxImageSize : UShortPentaxMakerNotesEntry
	{
		public MNPentaxImageSize(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0009;

		public const string TagName = "Pentax image size";

		public ushort[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.Length.ToString() + " entries";
		}
	}

	public class MNFlashMode : UShortPentaxMakerNotesEntry
	{
		public MNFlashMode(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x000c;

		public const string TagName = "Flash mode";

		public ushort[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			if (Value.Length == 1)
				return TagName + ": " + DictFlashModesA[Value[0]];
			else
				return TagName + ": " + DictFlashModesA[Value[0]] + "; " + DictFlashModesB[Value[1]];
		}
	}

	public class MNFocusMode : UShortPentaxMakerNotesEntry
	{
		public MNFocusMode(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x000d;

		public const string TagName = "Focus mode";

		public string Value
		{
			get { return DictFocusModes[mValue[0]]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value;
		}
	}

	public class MNAFPointSelected : UShortPentaxMakerNotesEntry
	{
		public MNAFPointSelected(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x000e;

		public const string TagName = "AF point selected";

		public string Value
		{
			get {
				string val = string.Empty;
				if (mMakerNotes.K3Specific)
					val += DictAFPointSelectedK3[mValue[0]];
				else
					val += DictAFPointSelectedNonK3[mValue[0]];

				if (mValue.Length > 1)
					val += "; " + DictAFPointSelectedValue2[mValue[1]];
				return val;
			}
		}

		public override string ToString()
		{
			return TagName + ": " + Value;
		}
	}

	public class MNAFPointsInFocus : UIntPentaxMakerNotesEntry
	{
		public class AFPointsInFocus_Base
		{ 
			
		}

		public class AFPointsInFocus : AFPointsInFocus_Base
		{
			public enum AFPointsInFocusEnum : ushort
			{
				FixedCenterOrMultiple = 0,
				TopLeft = 1,
				TopCenter = 2,
				TopRight = 3,
				Left = 4,
				Center = 5,
				Right = 6,
				BottomLeft = 7,
				BottomCenter = 8,
				BottomRight = 9,
				None = 0xffff
			}

			AFPointsInFocusEnum mValue;

			public AFPointsInFocus(ushort aValue)
			{
				mValue = (AFPointsInFocusEnum)aValue;
			}

			public AFPointsInFocusEnum Value
			{
				get { return mValue; }
			}

			public override string ToString()
			{
				return mValue.ToString();
			}
		}

		public class AFPointsInFocusK3 : AFPointsInFocus_Base
		{
			[Flags]
			public enum AFPointsInFocusEnum : uint
			{
				None = 0,
				TopLeft = 1,
				TopNearLeft = 2,
				Top = 4,
				TopNearRight = 8,
				TopRight = 16,
				UpperLeft = 32,
				UpperNearLeft = 64,
				UpperMiddle = 128,
				UpperNearRight = 256,
				UpperRight = 512,
				FarLeft = 1024,
				Left = 2048,
				NearLeft = 4096,
				Center = 8192,
				NearRight = 16384,
				Right = 32768,
				FarRight = 65536,
				LowerLeft = 131072,
				LowerNearLeft = 262144,
				LowerMiddle = 524288,
				LowerNearRight = 1048576,
				LowerRight = 2097152,
				BottomLeft = 4194304,
				BottomNearLeft = 8388608,
				Bottom = 16777216,
				BottomNearRight = 33554432,
				BottomRight = 67108864
			}

			AFPointsInFocusEnum mValue;

			public AFPointsInFocusK3(uint aValue)
			{
				mValue = (AFPointsInFocusEnum)aValue;
			}

			public AFPointsInFocusEnum Value
			{
				get { return mValue; }
			}

			public override string ToString()
			{
				return mValue.ToString();
			}
		}

		AFPointsInFocus_Base af_Value;

		public MNAFPointsInFocus(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			if (mMakerNotes.K3Specific)
				af_Value = new AFPointsInFocusK3(mValue[0]);
			else
				af_Value = new AFPointsInFocus((ushort)mValue[0]);
		}

		public const ushort TagID = 0x000f;

		public const string TagName = "AF points in focus";

		public AFPointsInFocus_Base Value
		{
			get { return af_Value; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNFocusPosition : UShortPentaxMakerNotesEntry
	{
		public MNFocusPosition(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0010;

		public const string TagName = "Focus position";

		public ushort Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNExposureTime : UIntPentaxMakerNotesEntry
	{
		public MNExposureTime(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0012;

		public const string TagName = "Exposure time";

		public float Value
		{
			get { return mValue[0] / 100000.0f; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNFNumber : UShortPentaxMakerNotesEntry
	{
		public MNFNumber(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0013;

		public const string TagName = "F number";

		public float Value
		{
			get { return mValue[0] / 10.0f; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNISO : UShortPentaxMakerNotesEntry
	{
		public MNISO(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0014;

		public const string TagName = "ISO";

		public uint Value
		{
			get { return DictISO[mValue[0]]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNLightReading : UShortPentaxMakerNotesEntry
	{
		public MNLightReading(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0015;

		public const string TagName = "Light reading";

		public ushort Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNExposureCompensation : UShortPentaxMakerNotesEntry
	{
		public MNExposureCompensation(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0016;

		public const string TagName = "Exposure compensation";

		public float Value
		{
			get { return ((float)mValue[0] - 50.0f) / 10.0f; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNMeteringMode : UShortPentaxMakerNotesEntry
	{
		public enum MeteringMode
		{ 
			MultiSegment = 0,
			CenterWeightedAverage = 1,
			Spot = 2
		}

		public MNMeteringMode(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0017;

		public const string TagName = "Metering mode";

		public MeteringMode Value
		{
			get { return (MeteringMode)mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNAutoBracketing : UShortPentaxMakerNotesEntry
	{
		public MNAutoBracketing(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0018;

		public const string TagName = "Auto bracketing";

		public ushort[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.Length.ToString() + " entries";
		}
	}

	public class MNWhiteBalance : UShortPentaxMakerNotesEntry
	{
		public enum WhiteBalance
		{ 
			Auto = 0,
			Daylight = 1,
			Shade = 2,
			Fluorescent = 3,
			Tungsten = 4,
			Manual = 5,
			DaylightFluorescent = 6,
			DayWhiteFluorescent = 7,
			WhiteFluorescent = 8,
			Flash = 9,
			Cloudy = 10,
			WarmWhiteFluorescent = 11,
			MultiAuto = 14,
			ColorTemperatureEnhancement = 15,
			Kelvin = 17,
			Unknown = 65534,
			UserSelected = 65535
		}

		public MNWhiteBalance(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0019;

		public const string TagName = "White balance";

		public WhiteBalance Value
		{
			get { return (WhiteBalance)mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNWhiteBalanceMode : UShortPentaxMakerNotesEntry
	{
		public enum WhiteBalanceMode
		{ 
			AutoDaylight = 1,
			AutoShade = 2,
			AutoFlash = 3,
			AutoTungsten = 4,
			AutoDaylightFluorescent = 6,
			AutoDayWhiteFluorescent = 7,
			AutoWhiteFluorescent = 8,
			AutoCloudy = 10, 
			Unknown = 65534,
			UserSelected = 65535
		}

		public MNWhiteBalanceMode(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x001a;

		public const string TagName = "White balance mode";

		public WhiteBalanceMode Value
		{
			get { return (WhiteBalanceMode)mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNBlueBalance : UShortPentaxMakerNotesEntry
	{
		public MNBlueBalance(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x001b;

		public const string TagName = "Blue balance";

		public float Value
		{
			get { return mValue[0] / 256.0f; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNRedBalance : UShortPentaxMakerNotesEntry
	{
		public MNRedBalance(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x001c;

		public const string TagName = "Red balance";

		public float Value
		{
			get { return mValue[0] / 256.0f; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNFocalLength : UIntPentaxMakerNotesEntry
	{
		public MNFocalLength(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x001d;

		public const string TagName = "Focal length";

		public float Value
		{
			get { return mValue[0] / 100.0f; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNDigitalZoom : UShortPentaxMakerNotesEntry
	{
		public MNDigitalZoom(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x001e;

		public const string TagName = "Digital zoom";

		public ushort Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNSaturation : UShortPentaxMakerNotesEntry
	{
		public MNSaturation(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x001f;

		public const string TagName = "Saturation";

		public ushort[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.Length.ToString() + " entries";
		}
	}

	public class MNContrast : UShortPentaxMakerNotesEntry
	{
		public MNContrast(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0020;

		public const string TagName = "Contrast";

		public ushort[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.Length.ToString() + " entries";
		}
	}

	public class MNSharpness : UShortPentaxMakerNotesEntry
	{
		public MNSharpness(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0021;

		public const string TagName = "Sharpness";

		public ushort[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.Length.ToString() + " entries";
		}
	}

	public class MNWorldTimeLocation : UShortPentaxMakerNotesEntry
	{
		public enum WorldTimeLocation
		{
			Hometown = 0,
			Destination = 1
		}

		public MNWorldTimeLocation(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0022;

		public const string TagName = "World time location";

		public WorldTimeLocation Value
		{
			get { return (WorldTimeLocation)mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNHometownCity : UShortPentaxMakerNotesEntry
	{
		public MNHometownCity(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0023;

		public const string TagName = "Hometown city";

		public string Value
		{
			get { return DictCities[mValue[0]]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value;
		}
	}

	public class MNDestinationCity : UShortPentaxMakerNotesEntry
	{
		public MNDestinationCity(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0024;

		public const string TagName = "Destination city";

		public string Value
		{
			get { return DictCities[mValue[0]]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value;
		}
	}

	public class MNHometownDST : UShortPentaxMakerNotesEntry
	{
		public MNHometownDST(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0025;

		public const string TagName = "Hometown DST";

		public bool Value
		{
			get { return mValue[0] == 1; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNDestinationDST : UShortPentaxMakerNotesEntry
	{
		public MNDestinationDST(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0026;

		public const string TagName = "Destination DST";

		public bool Value
		{
			get { return mValue[0] == 1; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNDSPFirmwareVersion : BytePentaxMakerNotesEntry
	{
		public MNDSPFirmwareVersion(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0027;

		public const string TagName = "DSP firmware version";

		public byte[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.Length.ToString() + " entries";
		}
	}

	public class MNCPUFirmwareVersion : BytePentaxMakerNotesEntry
	{
		public MNCPUFirmwareVersion(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0028;

		public const string TagName = "CPU firmware version";

		public byte[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.Length.ToString() + " entries";
		}
	}

	public class MNFrameNumber : UIntPentaxMakerNotesEntry
	{
		public MNFrameNumber(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0029;

		public const string TagName = "Frame number";

		public uint Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNEffectiveLV : UIntPentaxMakerNotesEntry
	{
		public MNEffectiveLV(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x002d;

		public const string TagName = "Effective LV";

		public uint Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNImageEditing : BytePentaxMakerNotesEntry
	{
		public MNImageEditing(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0032;

		public const string TagName = "Image editing";

		public byte[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.Length.ToString() + " entries";
		}
	}

	public class MNPictureMode : BytePentaxMakerNotesEntry
	{
		public MNPictureMode(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0033;

		public const string TagName = "Picture mode";

		public byte[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.Length.ToString() + " entries";
		}
	}

	public class MNDriveMode : BytePentaxMakerNotesEntry
	{
		public MNDriveMode(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0034;

		public const string TagName = "Drive mode";

		public byte[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.Length.ToString() + " entries";
		}
	}

	public class MNSensorSize : UShortPentaxMakerNotesEntry
	{
		float[] f_value;

		public MNSensorSize(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			f_value = new float[mValue.Length];
			for (int i = 0; i < mValue.Length; i++)
			{
				f_value[i] = mValue[i] / 500.0f;
			}
		}

		public const ushort TagID = 0x0035;

		public const string TagName = "Sensor size";

		public float[] Value
		{
			get { return f_value; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value[0] + " x " + Value[1];
		}
	}

	public class MNColorSpace : UShortPentaxMakerNotesEntry
	{
		public enum ColorSpace
		{ 
			sRGB = 0,
			AdobeRGB = 1
		}

		public MNColorSpace(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0037;

		public const string TagName = "Color space";

		public ColorSpace Value
		{
			get { return (ColorSpace)mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNImageAreaOffset : UShortPentaxMakerNotesEntry
	{
		public MNImageAreaOffset(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0038;

		public const string TagName = "Image area offset";

		public ushort[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value[0] + " x " + Value[1];
		}
	}

	public class MNRawImageSize : UShortPentaxMakerNotesEntry
	{
		public MNRawImageSize(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0039;

		public const string TagName = "Raw image size";

		public ushort[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value[0] + " x " + Value[1];
		}
	}

	public class MNDataScaling : UShortPentaxMakerNotesEntry
	{
		public MNDataScaling(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x003d;

		public const string TagName = "Data scaling";

		public ushort Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNPreviewImageBorders : BytePentaxMakerNotesEntry
	{
		public MNPreviewImageBorders(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x003e;

		public const string TagName = "Preview image borders";

		public byte[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": Top=" + Value[0] + "; Bottom=" + Value[1] + "; Left=" + Value[2] + "; Right=" + Value[3];
		}
	}

	public class MNLensRec : BytePentaxMakerNotesEntry
	{
		public MNLensRec(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x003f;

		public const string TagName = "Lens rec";

		public string Value
		{
			get { return DictLenses[mValue[0] * 256 + mValue[1]]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value;
		}
	}

	public class MNSensitivityAdjust : UShortPentaxMakerNotesEntry
	{
		public MNSensitivityAdjust(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0040;

		public const string TagName = "Sensitivity adjust";

		public ushort Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNImageEditCount : UShortPentaxMakerNotesEntry
	{
		public MNImageEditCount(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0041;

		public const string TagName = "Image edit count";

		public ushort Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNCameraTemperature : SBytePentaxMakerNotesEntry
	{
		public MNCameraTemperature(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0047;

		public const string TagName = "Camera temperature";

		public sbyte Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString() + " °C";
		}
	}

	public class MNAELock : UShortPentaxMakerNotesEntry
	{
		public MNAELock(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0048;

		public const string TagName = "AE lock";

		public bool Value
		{
			get { return mValue[0] == 1; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNNoiseReduction : UShortPentaxMakerNotesEntry
	{
		public MNNoiseReduction(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0049;

		public const string TagName = "Noise reduction";

		public bool Value
		{
			get { return mValue[0] == 1; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNFlashExposureComp : SBytePentaxMakerNotesEntry
	{
		public MNFlashExposureComp(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x004d;

		public const string TagName = "Flash exposure comp";

		public float Value
		{
			get { return mValue[0] / 6.0f; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNImageTone : UShortPentaxMakerNotesEntry
	{
		public enum ImageTone
		{ 
			Natural = 0, 
			Bright = 1,
			Portrait = 2,
			Landscape = 3,
			Vibrant = 4,
			Monochrome = 5,
			Muted = 6,
			ReversalFilm = 7,
			BleachBypass = 8,
			Radiant = 9
		}

		public MNImageTone(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x004f;

		public const string TagName = "Image tone";

		public ImageTone Value
		{
			get { return (ImageTone)mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNColorTemperature : UShortPentaxMakerNotesEntry
	{
		public MNColorTemperature(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0050;

		public const string TagName = "Color temperature";

		public int Value
		{
			get { return 53190  - mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNColorTempDaylight : BytePentaxMakerNotesEntry
	{
		public MNColorTempDaylight(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0053;

		public const string TagName = "Color temp daylight";

		public Byte[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.Length.ToString() + " entries";
		}
	}

	public class MNColorTempShade : BytePentaxMakerNotesEntry
	{
		public MNColorTempShade(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0054;

		public const string TagName = "Color temp shade";

		public Byte[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.Length.ToString() + " entries";
		}
	}

	public class MNColorTempCloudy : BytePentaxMakerNotesEntry
	{
		public MNColorTempCloudy(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0055;

		public const string TagName = "Color temp cloudy";

		public Byte[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.Length.ToString() + " entries";
		}
	}

	public class MNColorTempTungsten : BytePentaxMakerNotesEntry
	{
		public MNColorTempTungsten(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0056;

		public const string TagName = "Color temp tungsten";

		public Byte[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.Length.ToString() + " entries";
		}
	}

	public class MNColorTempFluorescentD : BytePentaxMakerNotesEntry
	{
		public MNColorTempFluorescentD(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0057;

		public const string TagName = "Color temp fluorescent D";

		public Byte[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.Length.ToString() + " entries";
		}
	}

	public class MNColorTempFluorescentN : BytePentaxMakerNotesEntry
	{
		public MNColorTempFluorescentN(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0058;

		public const string TagName = "Color temp fluorescent N";

		public Byte[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.Length.ToString() + " entries";
		}
	}

	public class MNColorTempFluorescentW : BytePentaxMakerNotesEntry
	{
		public MNColorTempFluorescentW(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0059;

		public const string TagName = "Color temp fluorescent W";

		public Byte[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.Length.ToString() + " entries";
		}
	}

	public class MNColorTempFlash : BytePentaxMakerNotesEntry
	{
		public MNColorTempFlash(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x005a;

		public const string TagName = "Color temp flash";

		public Byte[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.Length.ToString() + " entries";
		}
	}

	public class MNShakeReductionInfo : BytePentaxMakerNotesEntry
	{
		public enum ShakeRecuctionInfo
		{ 
			Off = 0,
			On = 1,
			OffAASimulationOff = 4,
			OnButDisabled = 5,
			OnVideo = 6,
			OnAASimulationOff = 7,
			OffAASimulationType1 = 12,
			OnAASimulationType1 = 15,
			OffAASimulationType2 = 20,
			OnAASimulationType2 = 23
		}

		public MNShakeReductionInfo(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x005c;

		public const string TagName = "Shake reduction info";

		public ShakeRecuctionInfo Value
		{
			get { return (ShakeRecuctionInfo)mValue[1]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNShutterCount : BytePentaxMakerNotesEntry
	{
		public MNShutterCount(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x005d;

		public const string TagName = "Shutter count";

		public byte[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.Length.ToString() + " entries";
		}
	}

	public class MNFaceInfo : BytePentaxMakerNotesEntry
	{
		public MNFaceInfo(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0060;

		public const string TagName = "Face info";

		public byte[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value[0].ToString() + " detected. Position main: " + Value[2].ToString() + "%; " + Value[3].ToString() + "%";
		}
	}

	public class MNRawDevelopmentProcess : UShortPentaxMakerNotesEntry
	{
		public MNRawDevelopmentProcess(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0062;

		public const string TagName = "Raw development process";

		public ushort Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNHue : UShortPentaxMakerNotesEntry
	{
		public MNHue(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0067;

		public const string TagName = "Hue";

		public ushort Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			string val;
			switch (Value)
			{
				case 0:
					val = "-2"; break;
				case 1:
					val = "Normal"; break;
				case 2:
					val = "2"; break;
				case 3:
					val = "-1"; break;
				case 4:
					val = "1"; break;
				case 5:
					val = "-3"; break;
				case 6:
					val = "3"; break;
				case 7:
					val = "-4"; break;
				case 8:
					val = "4"; break;
				default:
					val = "None";break;
			}
			return TagName + ": " + val;
		}
	}

	public class MNAWBInfo : BytePentaxMakerNotesEntry
	{
		public MNAWBInfo(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0068;

		public const string TagName = "AWB info";

		public byte[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": WhiteBalanceAutoAdjustment = " + (Value[0] == 1).ToString() + "; TungstenAWB = " + (Value[1] == 0 ? "Subtle correction" : "Strong correction");
		}
	}

	public class MNDynamicRangeExpansion : BytePentaxMakerNotesEntry
	{
		public MNDynamicRangeExpansion(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0069;

		public const string TagName = "Dynamic range expansion";

		public byte[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			string val1 = "Off";
			if (Value[0] == 1) val1 = "On";
			string val2 = "0";
			if (Value[1] == 1) val2 = "Enabled";
			if (Value[1] == 2) val2 = "Auto";

			return TagName + ": " + val1 + "; " + val2;
		}
	}

	public class MNTimeInfo : BytePentaxMakerNotesEntry
	{
		public class TimeInfo
		{
			public enum WorldTimeLocationEnum
			{ 
				Hometown = 0,
				Destination = 1
			}

			public WorldTimeLocationEnum WorldTimeLocation;
			public bool HometownDST;
			public bool DestinationDST;
			public string HometownCity;
			public string DestinationCity;

			public TimeInfo(byte[] aData)
			{
				WorldTimeLocation = (WorldTimeLocationEnum)(aData[0] & 1);
				HometownDST = ((aData[0] >> 1) & 1) == 1;
				DestinationDST = ((aData[0] >> 2) & 1) == 1;
				HometownCity = DictCities[aData[2]];
				DestinationCity = DictCities[aData[3]];
			}

			public override string ToString()
			{
				if (WorldTimeLocation == WorldTimeLocationEnum.Hometown) return HometownCity;
				if (WorldTimeLocation == WorldTimeLocationEnum.Destination) return DestinationCity;
				return "";
			}
		}

		TimeInfo ti_value;

		public MNTimeInfo(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			ti_value = new TimeInfo(mValue);
		}

		public const ushort TagID = 0x006b;

		public const string TagName = "Time info";

		public TimeInfo Value
		{
			get { return ti_value; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNHighLowKeyAdj : ShortPentaxMakerNotesEntry
	{
		public MNHighLowKeyAdj(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x006c;

		public const string TagName = "High low key adjust";

		public short Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNContrastHighlight : ShortPentaxMakerNotesEntry
	{
		public MNContrastHighlight(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x006d;

		public const string TagName = "Contrast highlight";

		public short Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNContrastShadow : ShortPentaxMakerNotesEntry
	{
		public MNContrastShadow(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x006e;

		public const string TagName = "Contrast shadow";

		public short Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNContrastHighlightShadowAdj : BytePentaxMakerNotesEntry
	{
		public MNContrastHighlightShadowAdj(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x006f;

		public const string TagName = "Contrast highlight shadow adjust";

		public bool Value
		{
			get { return mValue[0] == 1; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNFineSharpness : BytePentaxMakerNotesEntry
	{
		public MNFineSharpness(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0070;

		public const string TagName = "Fine sharpness";

		public byte[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + (Value[0] == 1).ToString() + "; " + (Value[1] == 2 ? "Extra fine" : "Normal");
		}
	}

	public class MNHighISONoiseReduction : BytePentaxMakerNotesEntry
	{
		public MNHighISONoiseReduction(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0071;

		public const string TagName = "High ISO noise reduction";

		public byte[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.Length.ToString() + " entries";
		}
	}

	public class MNAFAdjustment : ShortPentaxMakerNotesEntry
	{
		public MNAFAdjustment(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0072;

		public const string TagName = "AF adjustment";

		public short Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNMonochromeFilterEffect : UShortPentaxMakerNotesEntry
	{
		public enum MonochromeFilterEffect
		{ 
			Green = 1,
			Yellow = 2,
			Orange = 3,
			Red = 4,
			Magenta = 5,
			Blue = 6,
			Cyan = 7,
			Infrared = 8,
			None = 65535
		}

		public MNMonochromeFilterEffect(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0073;

		public const string TagName = "Monochrome filter effect";

		public MonochromeFilterEffect Value
		{
			get { return (MonochromeFilterEffect)mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNMonochromeToning : UShortPentaxMakerNotesEntry
	{
		public MNMonochromeToning(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0074;

		public const string TagName = "Monochrome toning";

		public ushort Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNFaceDetect : BytePentaxMakerNotesEntry
	{
		public MNFaceDetect(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0076;

		public const string TagName = "Face detect";

		public byte[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value[0].ToString() + " x " + Value[1].ToString();
		}
	}

	public class MNFaceDetectFrameSize : UShortPentaxMakerNotesEntry
	{
		public MNFaceDetectFrameSize(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0077;

		public const string TagName = "FaceDetect frame size";

		public ushort[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value[0].ToString() + " x " + Value[1].ToString();
		}
	}

	public class MNShadowCorrection : BytePentaxMakerNotesEntry
	{
		public MNShadowCorrection(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0079;

		public const string TagName = "Shadow correction";

		public byte[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.Length.ToString() + " entries";
		}
	}

	public class MNISOAutoParameters : BytePentaxMakerNotesEntry
	{
		public enum ISOAutoParameters
		{ 
			Slow = 1,
			Standard = 2,
			Fast = 3
		}

		public MNISOAutoParameters(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x007a;

		public const string TagName = "ISO auto parameters";

		public ISOAutoParameters Value
		{
			get { return (ISOAutoParameters)mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNCrossProcess : BytePentaxMakerNotesEntry
	{
		public enum CrossProcess
		{ 
			Off = 0,
			Random = 1,
			Preset1 = 2,
			Preset2 = 3,
			Preset3 = 4,
			Favorite1 = 33,
			Favorite2 = 34,
			Favorite3 = 35
		}

		public MNCrossProcess(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x007b;

		public const string TagName = "Cross process";

		public CrossProcess Value
		{
			get { return (CrossProcess)mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNLensCorr : BytePentaxMakerNotesEntry
	{
		public class LensCorr
		{
			public bool DistortionCorrection;
			public bool ChromaticAberrationCorrection;
			public bool VignettingCorrection;

			public LensCorr(byte[] aData)
			{
				DistortionCorrection = aData[0] == 1;
				ChromaticAberrationCorrection = aData[1] == 1;
				VignettingCorrection = aData[2] == 1;
			}

			public override string ToString()
			{
				StringBuilder sb = new StringBuilder();
				sb.Append("Distortion correction: ");
				if (DistortionCorrection)
					sb.Append("On");
				else
					sb.Append("Off");
				sb.Append("; Chromatic aberration correction: ");
				if (ChromaticAberrationCorrection)
					sb.Append("On");
				else
					sb.Append("Off");
				sb.Append("; Vignetting correction: ");
				if (VignettingCorrection)
					sb.Append("On");
				else
					sb.Append("Off");
				return sb.ToString();
			}
		}

		LensCorr lc_value;

		public MNLensCorr(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			lc_value = new LensCorr(mValue);
		}

		public const ushort TagID = 0x007d;

		public const string TagName = "Lens correction";

		public LensCorr Value
		{
			get { return lc_value; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNWhiteLevel : UIntPentaxMakerNotesEntry
	{
		public MNWhiteLevel(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
		}

		public const ushort TagID = 0x007e;

		public const string TagName = "White level";

		public uint Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNBleachBypassToning : UShortPentaxMakerNotesEntry
	{
		public enum BleachBypassToning
		{
			Green = 1,
			Yellow = 2,
			Orange = 3,
			Red = 4,
			Magenta = 5,
			Purple = 6,
			Blue = 7,
			Cyan = 8,
			None = 65535
		}

		public MNBleachBypassToning(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x007f;

		public const string TagName = "Bleach bypass toning";

		public BleachBypassToning Value
		{
			get { return (BleachBypassToning)mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNAspectRatio : BytePentaxMakerNotesEntry
	{
		public enum AspectRatio
		{
			Ratio4x3 = 0,
			Ratio3x2 = 1,
			Ratio16x9 = 2,
			Ratio1x1 = 3
		}

		public MNAspectRatio(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0080;

		public const string TagName = "Aspect ratio";

		public AspectRatio Value
		{
			get { return (AspectRatio)mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNBlurControl : BytePentaxMakerNotesEntry
	{
		public enum BlurControl
		{
			Off = 0,
			Low = 1,
			Medium = 2,
			High = 3
		}

		public MNBlurControl(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0082;

		public const string TagName = "Blur control";

		public BlurControl Value
		{
			get { return (BlurControl)mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNHDR : BytePentaxMakerNotesEntry
	{
		public MNHDR(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0085;

		public const string TagName = "HDR";

		public byte[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.Length.ToString() + " entries";
		}
	}

	public class MNNeutralDensityFilter : BytePentaxMakerNotesEntry
	{
		public MNNeutralDensityFilter(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0088;

		public const string TagName = "Neutral density filter";

		public bool Value
		{
			get { return mValue[0] == 1; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNISO2 : UIntPentaxMakerNotesEntry
	{
		public MNISO2(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x008b;

		public const string TagName = "ISO";

		public uint Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNBlackPoint : UShortPentaxMakerNotesEntry
	{
		public MNBlackPoint(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0200;

		public const string TagName = "Black point";

		public ushort[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value[0].ToString() + "; " + Value[1].ToString() + "; " + Value[2].ToString() + "; " + Value[3].ToString();
		}
	}

	public class MNWhitePoint : UShortPentaxMakerNotesEntry
	{
		public MNWhitePoint(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0201;

		public const string TagName = "White point";

		public ushort[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value[0].ToString() + "; " + Value[1].ToString() + "; " + Value[2].ToString() + "; " + Value[3].ToString();
		}
	}

	public class MNColorMatrixA : ShortPentaxMakerNotesEntry
	{
		public MNColorMatrixA(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0203;

		public const string TagName = "Color matrix A";

		public short[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			StringBuilder sb = new StringBuilder();
			sb.Append(TagName).Append(": ");
			for (int i = 0; i < Value.Length; i++)
			{
				sb.Append(Value[i]);
				if (i < Value.Length - 1)
					sb.Append("; ");
			}
			return sb.ToString();
		}
	}

	public class MNColorMatrixB : ShortPentaxMakerNotesEntry
	{
		public MNColorMatrixB(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0204;

		public const string TagName = "Color matrix B";

		public short[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			StringBuilder sb = new StringBuilder();
			sb.Append(TagName).Append(": ");
			for (int i = 0; i < Value.Length; i++)
			{
				sb.Append(Value[i]);
				if (i < Value.Length - 1)
					sb.Append("; ");
			}
			return sb.ToString();
		}
	}

	public class MNCameraSettings : BytePentaxMakerNotesEntry
	{
		public class CameraSettings
		{ 
				
		}

		public MNCameraSettings(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0205;

		public const string TagName = "Camera settings";

		public byte[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.Length.ToString() + " entries";
		}
	}

	public class MNAEInfo : BytePentaxMakerNotesEntry
	{
		public class AEInfo
		{ 
				
		}

		public MNAEInfo(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0206;

		public const string TagName = "AE info";

		public byte[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.Length.ToString() + " entries";
		}
	}

	public class MNLensInfo : BytePentaxMakerNotesEntry
	{
		public class LensInfo
		{
			public int LensTypeA;
			public int LensTypeB;
			public string Lens;
			public float FocalLength;
			public float NominalMaxAperture;// int8u [Mask 0xf0] 
			public float NominalMinAperture;


			public LensInfo(byte[] aData)
			{
				LensTypeA = aData[0+1] & 0x0f;
				LensTypeB = aData[3+1] * 256 + aData[4+1];
				try
				{
					Lens = DictLenses[LensTypeA * 256 + LensTypeB];
				}
				catch
				{
					Lens = string.Empty;
				}

				int temp = aData[24];
				FocalLength = 10 * (temp >> 2) * (float)Math.Pow(4, (temp & 0x03) - 2);

				temp = aData[25];
				NominalMaxAperture = (float)Math.Pow(2, ((temp & 0xf0) >> 4) / 4.0);
				NominalMinAperture = (float)Math.Pow(2, ((temp & 0x0f) + 10) / 4.0);
			}
		}

		LensInfo li_value;

		public MNLensInfo(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			li_value = new LensInfo(mValue);
		}

		public const ushort TagID = 0x0207;

		public const string TagName = "Lens info";

		public LensInfo Value
		{
			get { return li_value; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNFlashInfo : BytePentaxMakerNotesEntry
	{
		public class FlashInfo
		{ 
				
		}

		public MNFlashInfo(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0208;

		public const string TagName = "Flash info";

		public byte[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.Length.ToString() + " entries";
		}
	}

	public class MNAEMeteringSegments : BytePentaxMakerNotesEntry
	{
		public MNAEMeteringSegments(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0209;

		public const string TagName = "AE metering segments";

		public byte[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.Length.ToString() + " entries";
		}
	}

	public class MNFlashMeteringSegments : BytePentaxMakerNotesEntry
	{
		public MNFlashMeteringSegments(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x020a;

		public const string TagName = "Flash metering segments";

		public byte[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.Length.ToString() + " entries";
		}
	}

	public class MNSlaveFlashMeteringSegments : BytePentaxMakerNotesEntry
	{
		public MNSlaveFlashMeteringSegments(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x020b;

		public const string TagName = "Slave flash metering segments";

		public byte[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.Length.ToString() + " entries";
		}
	}

	public class MNWB_RGGBLevelsDaylight : UShortPentaxMakerNotesEntry
	{
		public MNWB_RGGBLevelsDaylight(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x020d;

		public const string TagName = "WB_RGGBLevelsDaylight";

		public ushort[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value[0].ToString() + "; " + Value[1].ToString() + "; " + Value[2].ToString() + "; " + Value[3].ToString();
		}
	}

	public class MNWB_RGGBLevelsShade : UShortPentaxMakerNotesEntry
	{
		public MNWB_RGGBLevelsShade(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x020e;

		public const string TagName = "WB_RGGBLevelsShade";

		public ushort[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value[0].ToString() + "; " + Value[1].ToString() + "; " + Value[2].ToString() + "; " + Value[3].ToString();
		}
	}

	public class MNWB_RGGBLevelsCloudy : UShortPentaxMakerNotesEntry
	{
		public MNWB_RGGBLevelsCloudy(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x020f;

		public const string TagName = "WB_RGGBLevelsCloudy";

		public ushort[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value[0].ToString() + "; " + Value[1].ToString() + "; " + Value[2].ToString() + "; " + Value[3].ToString();
		}
	}

	public class MNWB_RGGBLevelsTungsten : UShortPentaxMakerNotesEntry
	{
		public MNWB_RGGBLevelsTungsten(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0210;

		public const string TagName = "WB_RGGBLevelsTungsten";

		public ushort[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value[0].ToString() + "; " + Value[1].ToString() + "; " + Value[2].ToString() + "; " + Value[3].ToString();
		}
	}

	public class MNWB_RGGBLevelsFluorescentD : UShortPentaxMakerNotesEntry
	{
		public MNWB_RGGBLevelsFluorescentD(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0211;

		public const string TagName = "WB_RGGBLevelsFluorescentD";

		public ushort[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value[0].ToString() + "; " + Value[1].ToString() + "; " + Value[2].ToString() + "; " + Value[3].ToString();
		}
	}

	public class MNWB_RGGBLevelsFluorescentN : UShortPentaxMakerNotesEntry
	{
		public MNWB_RGGBLevelsFluorescentN(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0212;

		public const string TagName = "WB_RGGBLevelsFluorescentN";

		public ushort[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value[0].ToString() + "; " + Value[1].ToString() + "; " + Value[2].ToString() + "; " + Value[3].ToString();
		}
	}

	public class MNWB_RGGBLevelsFluorescentW : UShortPentaxMakerNotesEntry
	{
		public MNWB_RGGBLevelsFluorescentW(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0213;

		public const string TagName = "WB_RGGBLevelsFluorescentW";

		public ushort[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value[0].ToString() + "; " + Value[1].ToString() + "; " + Value[2].ToString() + "; " + Value[3].ToString();
		}
	}

	public class MNWB_RGGBLevelsFlash : UShortPentaxMakerNotesEntry
	{
		public MNWB_RGGBLevelsFlash(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{

		}

		public const ushort TagID = 0x0214;

		public const string TagName = "WB_RGGBLevelsFlash";

		public ushort[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value[0].ToString() + "; " + Value[1].ToString() + "; " + Value[2].ToString() + "; " + Value[3].ToString();
		}
	}

	public class MNCameraInfo : UIntPentaxMakerNotesEntry
	{
		public class CameraInfo
		{
			public string ModelID;
			public DateTime ManufactureDate;
			public uint ProductionCodeA;
			public uint ProductionCodeB;
			public uint InternalSerialNumber;

			public CameraInfo(uint[] aData)
			{
				try
				{
					ModelID = DictPentaxModelIDs[aData[0]];
					int year = (int)aData[1] / 10000;
					int month = ((int)aData[1] - year * 10000) / 100;
					int day = (int)aData[1] - year * 10000 - month * 100;
					ManufactureDate = new DateTime(year, month, day);
				}
				catch (Exception)
				{
					ModelID = string.Empty;
					ManufactureDate = new DateTime();
				}
				ProductionCodeA = aData[2];
				ProductionCodeB = aData[3];
				InternalSerialNumber = aData[4];
			}

			public override string ToString()
			{
				return ModelID + " build on " + ManufactureDate.ToString("D") + "; Internal S/N: " + InternalSerialNumber.ToString();
			}
		}

		public MNCameraInfo(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			ci_value = new CameraInfo(mValue);
		}

		CameraInfo ci_value;

		public const ushort TagID = 0x0215;

		public const string TagName = "Camera info";

		public CameraInfo Value
		{
			get { return ci_value; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNBatteryInfo : BytePentaxMakerNotesEntry
	{
		public class BatteryInfo
		{
			public enum PowerSource
			{ 
				BodyBattery = 2,
				GripBattery = 3,
				ExternalPowerSupply = 4
			}
			public enum BatteryState
			{
				EmptyOrMissing = 0x1,
				AlmostEmpty = 0x2,
				RunningLow = 0x3,
				CloseToFull = 0x4,
				Full = 0x5
			}

			public PowerSource Powersource;
			public BatteryState BodyBatteryState;
			public BatteryState GripBatteryState;
			public float BodyBatteryVoltage1;
			public float BodyBatteryVoltage2;

			public BatteryInfo(byte[] aData)
			{
				Powersource = (PowerSource)(aData[0] & 0x0f);
				BodyBatteryState = (BatteryState)((aData[1] & 0xf0) >> 4);
				GripBatteryState = (BatteryState)(aData[1] & 0x0f);
				BodyBatteryVoltage1 = (aData[2] * 256 + aData[3]) / 100.0f;
                if (aData.Length > 4)
				    BodyBatteryVoltage2 = (aData[4] * 256 + aData[5]) / 100.0f;
			}

			public override string ToString()
			{
				return Powersource.ToString() + " ; Body battary state: " + BodyBatteryState.ToString() + "; Grip battary state: " + GripBatteryState.ToString();
			}
		}

		public MNBatteryInfo(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			bi_value = new BatteryInfo(mValue);
		}

		BatteryInfo bi_value;

		public const ushort TagID = 0x0216;

		public const string TagName = "Battery info";

		public BatteryInfo Value
		{
			get { return bi_value; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class MNHuffmanTable : BytePentaxMakerNotesEntry
	{
		ushort[] huffmanTable = new ushort[4097];

		public MNHuffmanTable(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			ushort[,] bit = new ushort[2, 15];
			int dep;
			FileReader fr = new FileReader(new System.IO.MemoryStream(mValue));
            if (aPefFile.EndianSwap)
            {
                ushort temp = fr.ReadUI2();
                temp = (ushort)((temp >> 8) | (temp << 8));
                dep = (temp + 12) & 15;
                fr.Seek(12, System.IO.SeekOrigin.Current);

                for (int c = 0; c < dep; c++)
                {
                    temp = fr.ReadUI2();
                    temp = (ushort)((temp >> 8) | (temp << 8));
                    bit[0, c] = temp;
                }

                for (int c = 0; c < dep; c++)
                {
                    bit[1, c] = fr.ReadUI1();
                }

                for (int c = 0; c < dep; c++)
                {
                    for (int i = bit[0, c]; i <= ((bit[0, c] + (4096 >> bit[1, c]) - 1) & 4095);)
                        huffmanTable[++i] = (ushort)(bit[1, c] << 8 | c);
                    huffmanTable[0] = 12;
                }
            }
            else
            { 
                dep = (fr.ReadUI2() + 12) & 15;
                fr.Seek(12, System.IO.SeekOrigin.Current);

                for (int c = 0; c < dep; c++)
                {
                    bit[0, c] = fr.ReadUI2();
                }

                for (int c = 0; c < dep; c++)
                {
                    bit[1, c] = fr.ReadUI1();
                }

                for (int c = 0; c < dep; c++)
                {
                    for (int i = bit[0, c]; i <= ((bit[0, c] + (4096 >> bit[1, c]) - 1) & 4095);)
                        huffmanTable[++i] = (ushort)(bit[1, c] << 8 | c);
                    huffmanTable[0] = 12;
                }
            }
		}

		public const ushort TagID = 0x0220;

		public const string TagName = "HuffmanTable";

		public ushort[] Value
		{
			get { return huffmanTable; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.Length.ToString() + " entries";
		}
	}

	public class MNTempInfo : SBytePentaxMakerNotesEntry
	{
		public MNTempInfo(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
				
		}

		public const ushort TagID = 0x03ff;

		public const string TagName = "Temp info";

		public sbyte[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.Length.ToString() + " entries";
		}
	}

	public class MNLevelInfo : SBytePentaxMakerNotesEntry
	{
		public class LevelInfo
		{
			public enum LevelOrientation
			{
				Horizontal = 1,
				Rotate180 = 2,
				Rotate90CW = 3,
				Rotate270CW = 4,
				HorizontalOffLevel = 9,
				Rotate180OffLevel = 10,
				Rotate90CWOffLevel = 11,
				Rotate270CWOffLevel = 12,
				Upwards = 13,
				Downwards = 14
			}

			public enum CompositionAdjust
			{
				Off = 0,
				CompositionAdjust = 0x20,
				CompositionAdjustAndHorizonCorrection = 0xA0,
				HorizonCorrection = 0xC0
			}

			public LevelOrientation Orientation;
			public CompositionAdjust Composition;
			public float RollAngle;
			public float PitchAngle;
			public float CompositionAdjustX;
			public float CompositionAdjustY;
			public float CompositionAdjustRotation;

			public LevelInfo(sbyte[] aData)
			{
				Orientation = (LevelOrientation)(aData[0] & 0x0f);
				Composition = (CompositionAdjust)(aData[0] & 0xf0);
				RollAngle = -aData[1] / 2.0f; //degrees of clockwise camera rotation
				PitchAngle = -aData[2] / 2.0f; //degrees of upward camera tilt
				CompositionAdjustX = -aData[5]; //steps up, 1/16 mm per step
				CompositionAdjustY = -aData[6]; //steps up, 1/16 mm per step
				CompositionAdjustRotation = -aData[7] / 2.0f * (1/8.0f); //degreess CW
			}

			public override string ToString()
			{
				string val = "Orientation: " + Orientation.ToString();
				val += "; CompostionAdjust: " + Composition.ToString();
				val += "; RollAngle: " + RollAngle.ToString();
				val += "; PitchAngle: " + PitchAngle.ToString();
				val += "; AdjustX: " + CompositionAdjustX.ToString();
				val += "; AdjustY: " + CompositionAdjustY.ToString();
				val += "; AdjustRotation: " + CompositionAdjustRotation.ToString();
				return val;
			}
		}

		LevelInfo li_value;

		public MNLevelInfo(PentaxMakerNotes aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			li_value = new LevelInfo(mValue);
		}

		public const ushort TagID = 0x022b;

		public const string TagName = "Level info";

		public LevelInfo Value
		{
			get { return li_value; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}


}
