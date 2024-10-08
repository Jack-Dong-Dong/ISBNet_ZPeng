import numpy as np

list_of_species = {
    'AbiAlb': 0,
    'AceCam': 1,
    'AcePse': 2,
    'BetPen': 3,
    'CarBet': 4,
    'FagSyl': 5,
    'FraExc': 6,
    'JugReg': 7,
    'LarDec': 8,
    'PicAbi': 9,
    'PinSyl': 10,
    'PruAvi': 11,
    'PruSer': 12,
    'PseMen': 13,
    'QuePet': 14,
    'QueRob': 15,
    'QueRub': 16,
    'RobPse': 17,
    'SalCap': 18,
    'SorTor': 19,
    'TilSpe': 20,
    'TsuHet': 21,
}

list_of_species_colors = np.array([
    (174, 199, 232),  # AbiAlb
    (152, 223, 138),  # AceCam
    (31, 119, 180),  # AcePse
    (255, 187, 120),  # BetPen
    (188, 189, 34),  # CarBet
    (140, 86, 75),  # FagSyl
    (255, 152, 150),  # FraExc
    (214, 39, 40),  # JugReg
    (197, 176, 213),  # LarDec
    (148, 103, 189),  # PicAbi
    (196, 156, 148),  # PinSyl
    (23, 190, 207),  # PruAvi
    (178, 76, 76), # PruSer
    (247, 182, 210),  # PseMen
    (66, 188, 102),  # QuePet
    (219, 219, 141),  # QueRob
    (140, 57, 197),  # QueRub
    (202, 185, 52),  # RobPse
    (51, 176, 203),  # SalCap
    (200, 54, 131),  # SorTor
    (92, 193, 61),  # TilSpe
    (78, 71, 183),  # TsuHet
])