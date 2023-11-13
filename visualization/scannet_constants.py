### ScanNet Benchmark constants ###
VALID_CLASS_IDS_20 = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)

CLASS_LABELS_20 = (
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refrigerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
    "otherfurniture",
)

SCANNET_COLOR_MAP_20 = {
    0: (0.0, 0.0, 0.0),
    1: (174.0, 199.0, 232.0),
    2: (152.0, 223.0, 138.0),
    3: (31.0, 119.0, 180.0),
    4: (255.0, 187.0, 120.0),
    5: (188.0, 189.0, 34.0),
    6: (140.0, 86.0, 75.0),
    7: (255.0, 152.0, 150.0),
    8: (214.0, 39.0, 40.0),
    9: (197.0, 176.0, 213.0),
    10: (148.0, 103.0, 189.0),
    11: (196.0, 156.0, 148.0),
    12: (23.0, 190.0, 207.0),
    14: (247.0, 182.0, 210.0),
    15: (66.0, 188.0, 102.0),
    16: (219.0, 219.0, 141.0),
    17: (140.0, 57.0, 197.0),
    18: (202.0, 185.0, 52.0),
    19: (51.0, 176.0, 203.0),
    20: (200.0, 54.0, 131.0),
    21: (92.0, 193.0, 61.0),
    22: (78.0, 71.0, 183.0),
    23: (172.0, 114.0, 82.0),
    24: (255.0, 127.0, 14.0),
    25: (91.0, 163.0, 138.0),
    26: (153.0, 98.0, 156.0),
    27: (140.0, 153.0, 101.0),
    28: (158.0, 218.0, 229.0),
    29: (100.0, 125.0, 154.0),
    30: (178.0, 127.0, 135.0),
    32: (146.0, 111.0, 194.0),
    33: (44.0, 160.0, 44.0),
    34: (112.0, 128.0, 144.0),
    35: (96.0, 207.0, 209.0),
    36: (227.0, 119.0, 194.0),
    37: (213.0, 92.0, 176.0),
    38: (94.0, 106.0, 211.0),
    39: (82.0, 84.0, 163.0),
    40: (100.0, 85.0, 144.0),
}

### ScanNet200 Benchmark constants ###
VALID_CLASS_IDS_200 = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    21,
    22,
    23,
    24,
    26,
    27,
    28,
    29,
    31,
    32,
    33,
    34,
    35,
    36,
    38,
    39,
    40,
    41,
    42,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    54,
    55,
    56,
    57,
    58,
    59,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    82,
    84,
    86,
    87,
    88,
    89,
    90,
    93,
    95,
    96,
    97,
    98,
    99,
    100,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    110,
    112,
    115,
    116,
    118,
    120,
    121,
    122,
    125,
    128,
    130,
    131,
    132,
    134,
    136,
    138,
    139,
    140,
    141,
    145,
    148,
    154,
    155,
    156,
    157,
    159,
    161,
    163,
    165,
    166,
    168,
    169,
    170,
    177,
    180,
    185,
    188,
    191,
    193,
    195,
    202,
    208,
    213,
    214,
    221,
    229,
    230,
    232,
    233,
    242,
    250,
    261,
    264,
    276,
    283,
    286,
    300,
    304,
    312,
    323,
    325,
    331,
    342,
    356,
    370,
    392,
    395,
    399,
    408,
    417,
    488,
    540,
    562,
    570,
    572,
    581,
    609,
    748,
    776,
    1156,
    1163,
    1164,
    1165,
    1166,
    1167,
    1168,
    1169,
    1170,
    1171,
    1172,
    1173,
    1174,
    1175,
    1176,
    1178,
    1179,
    1180,
    1181,
    1182,
    1183,
    1184,
    1185,
    1186,
    1187,
    1188,
    1189,
    1190,
    1191,
)

CLASS_LABELS_200 = (
    "wall",
    "chair",
    "floor",
    "table",
    "door",
    "couch",
    "cabinet",
    "shelf",
    "desk",
    "office chair",
    "bed",
    "pillow",
    "sink",
    "picture",
    "window",
    "toilet",
    "bookshelf",
    "monitor",
    "curtain",
    "book",
    "armchair",
    "coffee table",
    "box",
    "refrigerator",
    "lamp",
    "kitchen cabinet",
    "towel",
    "clothes",
    "tv",
    "nightstand",
    "counter",
    "dresser",
    "stool",
    "cushion",
    "plant",
    "ceiling",
    "bathtub",
    "end table",
    "dining table",
    "keyboard",
    "bag",
    "backpack",
    "toilet paper",
    "printer",
    "tv stand",
    "whiteboard",
    "blanket",
    "shower curtain",
    "trash can",
    "closet",
    "stairs",
    "microwave",
    "stove",
    "shoe",
    "computer tower",
    "bottle",
    "bin",
    "ottoman",
    "bench",
    "board",
    "washing machine",
    "mirror",
    "copier",
    "basket",
    "sofa chair",
    "file cabinet",
    "fan",
    "laptop",
    "shower",
    "paper",
    "person",
    "paper towel dispenser",
    "oven",
    "blinds",
    "rack",
    "plate",
    "blackboard",
    "piano",
    "suitcase",
    "rail",
    "radiator",
    "recycling bin",
    "container",
    "wardrobe",
    "soap dispenser",
    "telephone",
    "bucket",
    "clock",
    "stand",
    "light",
    "laundry basket",
    "pipe",
    "clothes dryer",
    "guitar",
    "toilet paper holder",
    "seat",
    "speaker",
    "column",
    "bicycle",
    "ladder",
    "bathroom stall",
    "shower wall",
    "cup",
    "jacket",
    "storage bin",
    "coffee maker",
    "dishwasher",
    "paper towel roll",
    "machine",
    "mat",
    "windowsill",
    "bar",
    "toaster",
    "bulletin board",
    "ironing board",
    "fireplace",
    "soap dish",
    "kitchen counter",
    "doorframe",
    "toilet paper dispenser",
    "mini fridge",
    "fire extinguisher",
    "ball",
    "hat",
    "shower curtain rod",
    "water cooler",
    "paper cutter",
    "tray",
    "shower door",
    "pillar",
    "ledge",
    "toaster oven",
    "mouse",
    "toilet seat cover dispenser",
    "furniture",
    "cart",
    "storage container",
    "scale",
    "tissue box",
    "light switch",
    "crate",
    "power outlet",
    "decoration",
    "sign",
    "projector",
    "closet door",
    "vacuum cleaner",
    "candle",
    "plunger",
    "stuffed animal",
    "headphones",
    "dish rack",
    "broom",
    "guitar case",
    "range hood",
    "dustpan",
    "hair dryer",
    "water bottle",
    "handicap bar",
    "purse",
    "vent",
    "shower floor",
    "water pitcher",
    "mailbox",
    "bowl",
    "paper bag",
    "alarm clock",
    "music stand",
    "projector screen",
    "divider",
    "laundry detergent",
    "bathroom counter",
    "object",
    "bathroom vanity",
    "closet wall",
    "laundry hamper",
    "bathroom stall door",
    "ceiling light",
    "trash bin",
    "dumbbell",
    "stair rail",
    "tube",
    "bathroom cabinet",
    "cd case",
    "closet rod",
    "coffee kettle",
    "structure",
    "shower head",
    "keyboard piano",
    "case of water bottles",
    "coat rack",
    "storage organizer",
    "folded chair",
    "fire alarm",
    "power strip",
    "calendar",
    "poster",
    "potted plant",
    "luggage",
    "mattress",
)

SCANNET_COLOR_MAP_200 = {
    0: (0.0, 0.0, 0.0),
    1: (174.0, 199.0, 232.0),
    2: (188.0, 189.0, 34.0),
    3: (152.0, 223.0, 138.0),
    4: (255.0, 152.0, 150.0),
    5: (214.0, 39.0, 40.0),
    6: (91.0, 135.0, 229.0),
    7: (31.0, 119.0, 180.0),
    8: (229.0, 91.0, 104.0),
    9: (247.0, 182.0, 210.0),
    10: (91.0, 229.0, 110.0),
    11: (255.0, 187.0, 120.0),
    13: (141.0, 91.0, 229.0),
    14: (112.0, 128.0, 144.0),
    15: (196.0, 156.0, 148.0),
    16: (197.0, 176.0, 213.0),
    17: (44.0, 160.0, 44.0),
    18: (148.0, 103.0, 189.0),
    19: (229.0, 91.0, 223.0),
    21: (219.0, 219.0, 141.0),
    22: (192.0, 229.0, 91.0),
    23: (88.0, 218.0, 137.0),
    24: (58.0, 98.0, 137.0),
    26: (177.0, 82.0, 239.0),
    27: (255.0, 127.0, 14.0),
    28: (237.0, 204.0, 37.0),
    29: (41.0, 206.0, 32.0),
    31: (62.0, 143.0, 148.0),
    32: (34.0, 14.0, 130.0),
    33: (143.0, 45.0, 115.0),
    34: (137.0, 63.0, 14.0),
    35: (23.0, 190.0, 207.0),
    36: (16.0, 212.0, 139.0),
    38: (90.0, 119.0, 201.0),
    39: (125.0, 30.0, 141.0),
    40: (150.0, 53.0, 56.0),
    41: (186.0, 197.0, 62.0),
    42: (227.0, 119.0, 194.0),
    44: (38.0, 100.0, 128.0),
    45: (120.0, 31.0, 243.0),
    46: (154.0, 59.0, 103.0),
    47: (169.0, 137.0, 78.0),
    48: (143.0, 245.0, 111.0),
    49: (37.0, 230.0, 205.0),
    50: (14.0, 16.0, 155.0),
    51: (196.0, 51.0, 182.0),
    52: (237.0, 80.0, 38.0),
    54: (138.0, 175.0, 62.0),
    55: (158.0, 218.0, 229.0),
    56: (38.0, 96.0, 167.0),
    57: (190.0, 77.0, 246.0),
    58: (208.0, 49.0, 84.0),
    59: (208.0, 193.0, 72.0),
    62: (55.0, 220.0, 57.0),
    63: (10.0, 125.0, 140.0),
    64: (76.0, 38.0, 202.0),
    65: (191.0, 28.0, 135.0),
    66: (211.0, 120.0, 42.0),
    67: (118.0, 174.0, 76.0),
    68: (17.0, 242.0, 171.0),
    69: (20.0, 65.0, 247.0),
    70: (208.0, 61.0, 222.0),
    71: (162.0, 62.0, 60.0),
    72: (210.0, 235.0, 62.0),
    73: (45.0, 152.0, 72.0),
    74: (35.0, 107.0, 149.0),
    75: (160.0, 89.0, 237.0),
    76: (227.0, 56.0, 125.0),
    77: (169.0, 143.0, 81.0),
    78: (42.0, 143.0, 20.0),
    79: (25.0, 160.0, 151.0),
    80: (82.0, 75.0, 227.0),
    82: (253.0, 59.0, 222.0),
    84: (240.0, 130.0, 89.0),
    86: (123.0, 172.0, 47.0),
    87: (71.0, 194.0, 133.0),
    88: (24.0, 94.0, 205.0),
    89: (134.0, 16.0, 179.0),
    90: (159.0, 32.0, 52.0),
    93: (213.0, 208.0, 88.0),
    95: (64.0, 158.0, 70.0),
    96: (18.0, 163.0, 194.0),
    97: (65.0, 29.0, 153.0),
    98: (177.0, 10.0, 109.0),
    99: (152.0, 83.0, 7.0),
    100: (83.0, 175.0, 30.0),
    101: (18.0, 199.0, 153.0),
    102: (61.0, 81.0, 208.0),
    103: (213.0, 85.0, 216.0),
    104: (170.0, 53.0, 42.0),
    105: (161.0, 192.0, 38.0),
    106: (23.0, 241.0, 91.0),
    107: (12.0, 103.0, 170.0),
    110: (151.0, 41.0, 245.0),
    112: (133.0, 51.0, 80.0),
    115: (184.0, 162.0, 91.0),
    116: (50.0, 138.0, 38.0),
    118: (31.0, 237.0, 236.0),
    120: (39.0, 19.0, 208.0),
    121: (223.0, 27.0, 180.0),
    122: (254.0, 141.0, 85.0),
    125: (97.0, 144.0, 39.0),
    128: (106.0, 231.0, 176.0),
    130: (12.0, 61.0, 162.0),
    131: (124.0, 66.0, 140.0),
    132: (137.0, 66.0, 73.0),
    134: (250.0, 253.0, 26.0),
    136: (55.0, 191.0, 73.0),
    138: (60.0, 126.0, 146.0),
    139: (153.0, 108.0, 234.0),
    140: (184.0, 58.0, 125.0),
    141: (135.0, 84.0, 14.0),
    145: (139.0, 248.0, 91.0),
    148: (53.0, 200.0, 172.0),
    154: (63.0, 69.0, 134.0),
    155: (190.0, 75.0, 186.0),
    156: (127.0, 63.0, 52.0),
    157: (141.0, 182.0, 25.0),
    159: (56.0, 144.0, 89.0),
    161: (64.0, 160.0, 250.0),
    163: (182.0, 86.0, 245.0),
    165: (139.0, 18.0, 53.0),
    166: (134.0, 120.0, 54.0),
    168: (49.0, 165.0, 42.0),
    169: (51.0, 128.0, 133.0),
    170: (44.0, 21.0, 163.0),
    177: (232.0, 93.0, 193.0),
    180: (176.0, 102.0, 54.0),
    185: (116.0, 217.0, 17.0),
    188: (54.0, 209.0, 150.0),
    191: (60.0, 99.0, 204.0),
    193: (129.0, 43.0, 144.0),
    195: (252.0, 100.0, 106.0),
    202: (187.0, 196.0, 73.0),
    208: (13.0, 158.0, 40.0),
    213: (52.0, 122.0, 152.0),
    214: (128.0, 76.0, 202.0),
    221: (187.0, 50.0, 115.0),
    229: (180.0, 141.0, 71.0),
    230: (77.0, 208.0, 35.0),
    232: (72.0, 183.0, 168.0),
    233: (97.0, 99.0, 203.0),
    242: (172.0, 22.0, 158.0),
    250: (155.0, 64.0, 40.0),
    261: (118.0, 159.0, 30.0),
    264: (69.0, 252.0, 148.0),
    276: (45.0, 103.0, 173.0),
    283: (111.0, 38.0, 149.0),
    286: (184.0, 9.0, 49.0),
    300: (188.0, 174.0, 67.0),
    304: (53.0, 206.0, 53.0),
    312: (97.0, 235.0, 252.0),
    323: (66.0, 32.0, 182.0),
    325: (236.0, 114.0, 195.0),
    331: (241.0, 154.0, 83.0),
    342: (133.0, 240.0, 52.0),
    356: (16.0, 205.0, 144.0),
    370: (75.0, 101.0, 198.0),
    392: (237.0, 95.0, 251.0),
    395: (191.0, 52.0, 49.0),
    399: (227.0, 254.0, 54.0),
    408: (49.0, 206.0, 87.0),
    417: (48.0, 113.0, 150.0),
    488: (125.0, 73.0, 182.0),
    540: (229.0, 32.0, 114.0),
    562: (158.0, 119.0, 28.0),
    570: (60.0, 205.0, 27.0),
    572: (18.0, 215.0, 201.0),
    581: (79.0, 76.0, 153.0),
    609: (134.0, 13.0, 116.0),
    748: (192.0, 97.0, 63.0),
    776: (108.0, 163.0, 18.0),
    1156: (95.0, 220.0, 156.0),
    1163: (98.0, 141.0, 208.0),
    1164: (144.0, 19.0, 193.0),
    1165: (166.0, 36.0, 57.0),
    1166: (212.0, 202.0, 34.0),
    1167: (23.0, 206.0, 34.0),
    1168: (91.0, 211.0, 236.0),
    1169: (79.0, 55.0, 137.0),
    1170: (182.0, 19.0, 117.0),
    1171: (134.0, 76.0, 14.0),
    1172: (87.0, 185.0, 28.0),
    1173: (82.0, 224.0, 187.0),
    1174: (92.0, 110.0, 214.0),
    1175: (168.0, 80.0, 171.0),
    1176: (197.0, 63.0, 51.0),
    1178: (175.0, 199.0, 77.0),
    1179: (62.0, 180.0, 98.0),
    1180: (8.0, 91.0, 150.0),
    1181: (77.0, 15.0, 130.0),
    1182: (154.0, 65.0, 96.0),
    1183: (197.0, 152.0, 11.0),
    1184: (59.0, 155.0, 45.0),
    1185: (12.0, 147.0, 145.0),
    1186: (54.0, 35.0, 219.0),
    1187: (210.0, 73.0, 181.0),
    1188: (221.0, 124.0, 77.0),
    1189: (149.0, 214.0, 66.0),
    1190: (72.0, 185.0, 134.0),
    1191: (42.0, 94.0, 198.0),
}

### For instance segmentation the non-object categories ###
VALID_PANOPTIC_IDS = (1, 3)

CLASS_LABELS_PANOPTIC = ("wall", "floor")
