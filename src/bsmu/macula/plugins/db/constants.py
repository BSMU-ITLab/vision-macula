DROPDOWN_DB_VALUES = {
    "areds": ["", "1", "2", "3", "4", "4a", "4b", "4c", "4d"],
    "refraction": ["", "1", "2", "3", "4", "5"],
    "rpe_status": ["", "1", "2", "3", "4", "5"],
    "rpe_localisation": ["", "0", "1", "2", "3"],
    "cme_localisation": ["", "0", "1", "2", "3"],
    "serouz_rpe_detachment_localisation": ["", "0", "1", "2", "3"],
    "hemorrhagic_rpe_detachment_localisation": ["", "0", "1", "2", "3"],
    "fibrovascular_rpe_detachment_localisation": ["", "0", "1", "2", "3"],
    "drusenoid_detachment_rpe_localisation": ["", "0", "1", "2", "3"],
    "druses_localisation": ["", "0", "1", "2", "3"],
    "fluid_under_rpe_localisation": ["", "0", "1", "2", "3"],
    "ez_status": ["", "1", "2", "3", "4"],
    "ez_localisation": ["", "0", "1", "2", "3"],
    "myoidnz_status": ["","1", "2", "3"],
    "myoidnz_localisation": ["", "0", "1", "2", "3"],
    "rne_detachment_localisation": ["", "0", "1", "2", "3"],
    "hyperreflective_material_localisation": ["", "0", "1", "2", "3"],
    "type_of_neovascularization": ["", "0", "1", "2", "3"],
    "topkon": [False, True]
}

DROPDOWN_DISPLAY_MAP2 = {
    "rpe_localisation": {
        "1": "хотя бы один дефект присутствует в фовеа",
        "2": "хотя бы один дефект присутствует в фовеоле",
        "3": "дефекты только в макуле; фовеа и фовеола без дефектов"
    },
    "serouz_rpe_detachment_localisation": {
        "1": "Хотя бы часть отслойки присутствует в фовеа",
        "2": "Хотя бы часть отслойки присутствует в фовеоле",
        "3": "Отслойка только в макуле; фовеа и фовеола без отслойки"
    },
    "hemorrhagic_rpe_detachment_localisation": {
        "1": "Хотя бы часть отслойки присутствует в фовеа",
        "2": "Хотя бы часть отслойки присутствует в фовеоле",
        "3": "Отслойка только в макуле; фовеа и фовеола без отслойки"
    },
    "fibrovascular_rpe_detachment_localisation": {
        "1": "Хотя бы часть отслойки присутствует в фовеа",
        "2": "Хотя бы часть отслойки присутствует в фовеоле",
        "3": "Отслойка только в макуле; фовеа и фовеола без отслойки"
    },
    "drusenoid_detachment_rpe_localisation": {
        "1": "Хотя бы часть отслойки присутствует в фовеа",
        "2": "Хотя бы часть отслойки присутствует в фовеоле",
        "3": "Отслойка только в макуле; фовеа и фовеола без отслойки"
    },
    "rne_detachment_localisation": {
        "1": "Хотя бы часть отслойки присутствует в фовеа",
        "2": "Хотя бы часть отслойки присутствует в фовеоле",
        "3": "Отслойка только в макуле; фовеа и фовеола без отслойки"
    }
}
DROPDOWN_DISPLAY_MAP = {
    "topkon": {
        False: "Оптопол",
        True: "Топкон",
    },
    "type_of_neovascularization": {
        "": "",
        "0": "ХНВ отсутствует",
        "1": "ХНВ 1",
        "2": "ХНВ 2",
        "3": "ХНВ 3 (РАП)",
    },
    "areds": {
        "": "",
        "1": "AREDS 1",
        "2": "AREDS 2",
        "3": "AREDS 3",
        "4": "AREDS 4",
        "4a": "AREDS 4a",
        "4b": "AREDS 4b",
        "4c": "AREDS 4c",
        "4d": "AREDS 4d"
    },
    "refraction": {
        "": "",
        "4": "Мсл",
        "5": "Мср",
        "1": "Em",
        "2": "Hmсл",
        "3": "Hmср"
    },
    "rpe_status": {
        "": "",
        "1": "эпителий сохранён",
        "2": "эпителий неравномерный",
        "3": "единичные разрывы",
        "4": "множественные разрывы",
        "5": "эпителий не определяется"
    },
    "rpe_localisation": {
        "": "",
        "0": "дефекты отсутствуют",
        "1": "фовеа + макула (без фовеолы)",
        "2": "фовеола + фовеа + макула",
        "3": "макула (без фовеа и фовеолы)"
    },
    "cme_localisation": {
        "": "",
        "0": "отсутствие отёка",
        "1": "фовеа + макула (без фовеолы)",
        "2": "фовеола + фовеа + макула",
        "3": "макула (без фовеа и фовеолы)"
    },
    "serouz_rpe_detachment_localisation": {
        "": "",
        "0": "отслойка отсутствует",
        "1": "фовеа + макула (без фовеолы)",
        "2": "фовеола + фовеа + макула",
        "3": "макула (без фовеа и фовеолы)"
    },
    "hemorrhagic_rpe_detachment_localisation": {
        "": "",
        "0": "отслойка отсутствует",
        "1": "фовеа + макула (без фовеолы)",
        "2": "фовеола + фовеа + макула",
        "3": "макула (без фовеа и фовеолы)"
    },
    "fibrovascular_rpe_detachment_localisation": {
        "": "",
        "0": "отслойка отсутствует",
        "1": "фовеа + макула (без фовеолы)",
        "2": "фовеола + фовеа + макула",
        "3": "макула (без фовеа и фовеолы)"
    },
    "drusenoid_detachment_rpe_localisation": {
        "": "",
        "0": "отслойка отсутствует",
        "1": "фовеа + макула (без фовеолы)",
        "2": "фовеола + фовеа + макула",
        "3": "макула (без фовеа и фовеолы)"
    },
    "druses_localisation": {
        "": "",
        "0": "отсутствие друз",
        "1": "фовеа + макула (без фовеолы)",
        "2": "фовеола + фовеа + макула",
        "3": "макула (без фовеа и фовеолы)"
    },
    "fluid_under_rpe_localisation": {
        "": "",
        "0": "дефекты отсутствуют",
        "1": "фовеа + макула (без фовеолы)",
        "2": "фовеола + фовеа + макула",
        "3": "макула (без фовеа и фовеолы)"
    },
    "ez_status": {
        "": "",
        "1": "сохранена",
        "2": "неравномерная (фрагментация)",
        "3": "не определяется локально",
        "4": "не определяется"
    },
    "ez_localisation": {
        "": "",
        "0": "дефекты отсутствуют",
        "1": "фовеа + макула (без фовеолы)",
        "2": "фовеола + фовеа + макула",
        "3": "макула (без фовеа и фовеолы)"
    },
    "myoidnz_status": {
        "": "",
        "1": "сохранена",
        "2": "неравномерная (фрагментация)",
        "3": "не определяется"
    },
    "myoidnz_localisation": {
        "": "",
        "0": "дефекты отсутствуют",
        "1": "фовеа + макула (без фовеолы)",
        "2": "фовеола + фовеа + макула",
        "3": "макула (без фовеа и фовеолы)"
    },
    "rne_detachment_localisation": {
        "": "",
        "0": "отслойка отсутствует",
        "1": "фовеа + макула (без фовеолы)",
        "2": "фовеола + фовеа + макула",
        "3": "макула (без фовеа и фовеолы)"
    },
    "hyperreflective_material_localisation": {
        "": "",
        "0": "отсутствие",
        "1": "субретинальный",
        "2": "интраретинальный",
        "3": "субретинальный + интраретинальный"
    }
}

BLOCKS = [
    ("Обследование", [
        # ("Дата посещения", 3),
        # ("Продолжительность заболевания", 4),
        # ("Возраст", 4),
        ("Тип томографа", 5),
        ("Стадия по AREDS", 6),
        ("Тип неоваскуляризации", 8),
        ("Рефракция", 7),
        ('MKO3', 50)
    ]),
    ("Ретинальные показатели", [
        ("Толщина хориоидеи в центре", 9),
        ("Толщина сетчатки в фовеоле", 10),
        ("Толщина сетчатки возле фовеолы", 48),
        ("Толщина сетчатки возле фовеа", 49),
        ("Общий объем", 11),
        ("Средний объем", 12)
    ]),
    ("", [
        ("Состояние РПЭ", 13),
        ("Локализация дефектов РПЭ", 14),
        ("Локализация кистозного макулярного отека", 15)
    ]),
    ("Серозная ОПЭ", [
        ("Локализация", 16),
        ("Ширина", 17),
        ("Высота", 18),
        ("Площадь", 19)
    ]),
    ("Геморрагическая ОПЭ", [
        ("Локализация", 20),
        ("Ширина", 21),
        ("Высота", 22),
        ("Площадь", 23)
    ]),
    ("Фиброваскулярная ОПЭ", [
        ("Локализация", 24),
        ("Ширина", 25),
        ("Высота", 26),
        ("Площадь", 27)
    ]),
    ("Друзеноидная ОПЭ", [
        ("Локализация", 28),
        ("Ширина", 29),
        ("Высота", 30),
        ("Площадь", 31)
    ]),
    ("Друзы", [
        ("Локализация", 32),
        ("Ширина", 33),
        ("Высота", 34),
        ("Площадь", 35)
    ]),
    ("Жидкость под РПЭ", [
        ("Пощадь", 36),
        ("Локализация", 37)
    ]),
    ("Эллипсоидная зона", [
        ("Состояние", 38),
        ("Локализация дефектов", 39)
    ]),
    ("Миоидная зона", [
        ("Состояние", 40),
        ("Локализация дефектов", 41)
    ]),
    ("Отслойка нейросенсорной сетчатки", [
        ("Локализация", 42),
        ("Ширина", 43),
        ("Высота", 44),
        ("Площадь", 45)
    ]),
    ("Гиперрефлективный материал", [
        ("Локализация", 46),
        ("Площадь", 47)
    ])]

COLUMN_INDEX_TO_FIELD_NAME = {
    3: "date",
    4: "duration_of_the_disease",
    5: "topkon",
    6: "areds",
    7: "refraction",
    8: "type_of_neovascularization",
    9: "choroidal_thickness_center",
    10: "cts_foveola",
    11: "total_volume",
    12: "average_volume",
    13: "rpe_status",
    14: "rpe_localisation",
    15: "cme_localisation",
    16: "serouz_rpe_detachment_localisation",
    17: "serouz_rpe_detachment_width",
    18: "serouz_rpe_detachment_height",
    19: "serouz_rpe_detachment_area",
    20: "hemorrhagic_rpe_detachment_localisation",
    21: "hemorrhagic_rpe_detachment_width",
    22: "hemorrhagic_rpe_detachment_heidgt",
    23: "hemorrhagic_rpe_detachment_area",
    24: "fibrovascular_rpe_detachment_localisation",
    25: "fibrovascular_rpe_detachment_width",
    26: "fibrovascular_rpe_detachment_heidgt",
    27: "fibrovascular_rpe_detachment_area",
    28: "drusenoid_detachment_rpe_localisation",
    29: "drusenoid_detachment_rpe_width",
    30: "drusenoid_detachment_rpe_height",
    31: "drusenoid_detachment_rpe_area",
    32: "druses_localisation",
    33: "druses_weigt",
    34: "druses_heigt",
    35: "dzuses_area",
    36: "fluid_under_rpe_area",
    37: "fluid_under_rpe_localisation",
    38: "ez_status",
    39: "ez_localisation",
    40: "myoidnz_status",
    41: "myoidnz_localisation",
    42: "rne_detachment_localisation",
    43: "rne_detachment_width",
    44: "rne_detachment_heigt",
    45: "rne_detachment_area",
    46: "hyperreflective_material_localisation",
    47: "hyperreflective_material_area",
    48: "cts_sup_inner_fovea",
    49: "cts_sup_out_fovea",
    50: "MKO3"
}

COLUMNS_EYES = [
            "id", "eye", "appointment_id", "date", "duration_of_the_disease", "topkon", "areds", "refraction", "type_of_neovascularization",
            "choroidal_thickness_center", "cts_foveola", "total_volume", "average_volume",
            "rpe_status", "rpe_localisation", "cme_localisation",
            "serouz_rpe_detachment_localisation", "serouz_rpe_detachment_width", "serouz_rpe_detachment_height",
            "serouz_rpe_detachment_area",
            "hemorrhagic_rpe_detachment_localisation", "hemorrhagic_rpe_detachment_width",
            "hemorrhagic_rpe_detachment_heidgt", "hemorrhagic_rpe_detachment_area",
            "fibrovascular_rpe_detachment_localisation", "fibrovascular_rpe_detachment_width",
            "fibrovascular_rpe_detachment_heidgt", "fibrovascular_rpe_detachment_area",
            "drusenoid_detachment_rpe_localisation", "drusenoid_detachment_rpe_width",
            "drusenoid_detachment_rpe_height", "drusenoid_detachment_rpe_area",
            "druses_localisation", "druses_weigt", "druses_heigt", "dzuses_area",
            "fluid_under_rpe_area", "fluid_under_rpe_localisation",
            "ez_status", "ez_localisation", "myoidnz_status", "myoidnz_localisation",
            "rne_detachment_localisation", "rne_detachment_width", "rne_detachment_heigt", "rne_detachment_area",
            "hyperreflective_material_localisation", "hyperreflective_material_area", "cts_sup_inner_fovea", "cts_sup_out_fovea", "MKO3"
        ]

EYE_TABLE_NAME = "eyes"
PACIENTS_TABLE_NAME = "pacients"

ERROR_TEXT = "Ошибка"

PARAMETERS_PATIENTS_SELECT = ['id', 'name', 'sex', 'year_of_birthday']
