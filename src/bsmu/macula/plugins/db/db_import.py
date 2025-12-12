# import pandas as pd
# import sqlite3
# from datetime import datetime
#
# def normalize_columns(df):
#     df.columns = [col.strip().lower()
#                       .replace(" ", "_")
#                       .replace(".", "")
#                       .replace("(", "")
#                       .replace(")", "")
#                       .replace("ё", "е")  # для совместимости
#                   for col in df.columns]
#     return df
#
# def create_tables(cursor):
#     cursor.execute("""
#     CREATE TABLE IF NOT EXISTS patients (
#         id               INTEGER PRIMARY KEY,
#         name             TEXT,
#         sex              TEXT,
#         year_of_birth    INTEGER
#     );
#     """)
#
#     cursor.execute("""
#     CREATE TABLE IF NOT EXISTS appointments (
#         id                      INTEGER PRIMARY KEY AUTOINCREMENT,
#         date                    DATE,
#         disease_duration_years INTEGER,
#         patient_id              INTEGER,
#         FOREIGN KEY (patient_id) REFERENCES patients(id)
#     );
#     """)
#
#     cursor.execute("""
#     CREATE TABLE IF NOT EXISTS examinations (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         appointment_id INTEGER,
#         eye TEXT, date DATE, topcon BOOLEAN, optopol BOOLEAN,
#         areds_stage TEXT, refraction TEXT, neovascular_type TEXT,
#         mkoz REAL, choroidal_thickness_center REAL,
#         cts_foveola REAL, cts_sup_inner_fovea REAL, cts_sup_outer_fovea REAL,
#         total_volume REAL, average_volume REAL,
#         rpe_status REAL, rpe_localisation REAL, cme_localisation REAL,
#         serous_rpe_detachment_localisation REAL, serous_rpe_detachment_width REAL,
#         serous_rpe_detachment_height REAL, serous_rpe_detachment_area REAL,
#         hemorrhagic_rpe_detachment_localisation REAL, hemorrhagic_rpe_detachment_width REAL,
#         hemorrhagic_rpe_detachment_height REAL, hemorrhagic_rpe_detachment_area REAL,
#         fibrovascular_rpe_detachment_localisation REAL, fibrovascular_rpe_detachment_width REAL,
#         fibrovascular_rpe_detachment_height REAL, fibrovascular_rpe_detachment_area REAL,
#         drusenoid_rpe_detachment_localisation REAL, drusenoid_rpe_detachment_width REAL,
#         drusenoid_rpe_detachment_height REAL, drusenoid_rpe_detachment_area REAL,
#         druses_localisation REAL, druses_weight REAL, druses_height REAL, druses_area REAL,
#         fluid_under_rpe_area REAL, fluid_under_rpe_localisation REAL,
#         ez_status REAL, ez_localisation REAL, myoidnz_status REAL, myoidnz_localisation REAL,
#         rne_detachment_localisation REAL, rne_detachment_width REAL,
#         rne_detachment_height REAL, rne_detachment_area REAL,
#         hyperreflective_material_localisation REAL, hyperreflective_material_area REAL,
#         lutein_therapy TEXT,
#         avastin TEXT, avastin_injections INTEGER,
#         eylea TEXT, eylea_injections INTEGER,
#         visque TEXT, visque_injections INTEGER,
#         diprospan TEXT, diprospan_injections INTEGER,
#         kenalog TEXT, kenalog_injections INTEGER,
#         lucentis TEXT, lucentis_injections INTEGER,
#         FOREIGN KEY (appointment_id) REFERENCES appointments(id)
#     );
#     """)
#
# def process_row(row, cursor):
#     # 🧑 Patient
#     patient_id = int(row["id_пациента"])
#     name = str(row["фио_пациента"]).replace("'", "''")
#     sex = str(row["пол"])[0].upper()
#     dob = row["возраст_в_годах"]
#     year_of_birth = dob.year if isinstance(dob, datetime) else datetime.strptime(str(dob)[:10], "%Y-%m-%d").year
#
#     cursor.execute("""
#     INSERT OR IGNORE INTO patients (id, name, sex, year_of_birth)
#     VALUES (?, ?, ?, ?)
#     """, (patient_id, name, sex, year_of_birth))
#
#     # 📅 Appointment
#     date_obj = row["дата_обследования"]
#     if not isinstance(date_obj, datetime):
#         date_obj = datetime.strptime(str(date_obj)[:10], "%Y-%m-%d")
#     duration = int(row.get("длитзабол_лет", 0))
#
#     cursor.execute("""
#     INSERT INTO appointments (date, disease_duration_years, patient_id)
#     VALUES (?, ?, ?)
#     """, (date_obj.date(), duration, patient_id))
#
#     appointment_id = cursor.lastrowid
#
#     # 🩺 Examination
#     fields = [
#         "eye", "топкон_true/false", "оптопол_true/false", "стадия_по_areds", "рефракция", "тип_неоваскуляризации",
#         "мкоз", "толщина_хориоидеи_в_центре", "цтс_foveola", "цтс_supinner_fovea", "цтс_supout_fovea",
#         "total_volume", "average_volume", "rpe_status", "rpe_localisation", "cme_localisation",
#         "serouz_rpe_detachment_localisation", "serouz_rpe_detachment_width", "serouz_rpe_detachment_height", "serouz_rpe_detachment_area",
#         "hemorrhagic_rpe_detachment_localisation", "hemorrhagic_rpe_detachment_width", "hemorrhagic_rpe_detachment_heidgt", "hemorrhagic_rpe_detachment_area",
#         "fibrovascular_rpe_detachment_localisation", "fibrovascular_rpe_detachment_width", "fibrovascular_rpe_detachment_heidgt", "fibrovascular_rpe_detachment_area",
#         "drusenoid_detachment_rpe_localisation", "drusenoid_detachment_rpe_width", "drusenoid_detachment_rpe_height", "drusenoid_detachment_rpe_area",
#         "druses_localisation", "druses_weigt", "druses_heigt", "dzuses_area",
#         "fluid_under_rpe_area", "fluid_under_rpe_localisation",
#         "ez_status", "ez_localisation", "myoidnz_status", "myoidnz_localisation",
#         "rne_detachment_localisation", "rne_detachment_width", "rne_detachment_heigt", "rne_detachment_area",
#         "гиперрефлективный_материал_локализация", "гиперрефлективный_материал_площадь",
#         "препараты_лютеина",
#         "бевацизумаб_авастин", "количество_выполненых_инъекций",
#         "афлиберцепт_эйлеа", "количество_выполненых_инъекций1",
#         "бролоцизумаб_визкью", "количество_выполненых_инъекций2",
#         "бетаметазон_дипроспан", "количество_выполненых_инъекций3",
#         "триамциналон_кеналог", "количество_выполненых_инъекций4",
#         "луцентис", "количество_выполненых_инъекций5"
#     ]
#
#     translated = {
#         "топкон_true/false": "topcon",
#         "оптопол_true/false": "optopol",
#         "стадия_по_areds": "areds_stage",
#         "рефракция": "refraction",
#         "тип_неоваскуляризации": "neovascular_type",
#         "мкоз": "mkoz",
#         "толщина_хориоидеи_в_центре": "choroidal_thickness_center",
#         "цтс_foveola": "cts_foveola",
#         "цтс_supinner_fovea": "cts_sup_inner_fovea",
#         "цтс_supout_fovea": "cts_sup_outer_fovea",
#         "total_volume": "total_volume",
#         "average_volume": "average_volume",
#         "rpe_status": "rpe_status",
#         "rpe_localisation": "rpe_localisation",
#         "cme_localisation": "cme_localisation",
#         "серouz_rpe_detachment_localisation": "serous_rpe_detachment_localisation",
#         "серouz_rpe_detachment_width": "serous_rpe_detachment_width",
#         "серouz_rpe_detachment_height": "serous_rpe_detachment_height",
#         "серouz_rpe_detachment_area": "serous_rpe_detachment_area",
#         "hemorrhagic_rpe_detachment_localisation": "hemorrhagic_rpe_detachment_localisation",
#         "hemorrhagic_rpe_detachment_width": "hemorrhagic_rpe_detachment_width",
#         "hemorrhagic_rpe_detachment_heidgt": "hemorrhagic_rpe_detachment_height",
#         "hemorrhagic_rpe_detachment_area": "hemorrhagic_rpe_detachment_area",
#         "fibrovascular_rpe_detachment_localisation": "fibrovascular_rpe_det",
#         "fibrovascular_rpe_detachment_localisation": "fibrovascular_rpe_detachment_localisation",
#         "fibrovascular_rpe_detachment_width": "fibrovascular_rpe_detachment_width",
#         "fibrovascular_rpe_detachment_heidgt": "fibrovascular_rpe_detachment_height",
#         "fibrovascular_rpe_detachment_area": "fibrovascular_rpe_detachment_area",
#         "drusenoid_detachment_rpe_localisation": "drusenoid_rpe_detachment_localisation",
#         "drusenoid_detachment_rpe_width": "drusenoid_rpe_detachment_width",
#         "drusenoid_detachment_rpe_height": "drusenoid_rpe_detachment_height",
#         "drusenoid_detachment_rpe_area": "drusenoid_rpe_detachment_area",
#         "druses_localisation": "druses_localisation",
#         "druses_weigt": "druses_weight",
#         "druses_heigt": "druses_height",
#         "dzuses_area": "druses_area",
#         "fluid_under_rpe_area": "fluid_under_rpe_area",
#         "fluid_under_rpe_localisation": "fluid_under_rpe_localisation",
#         "ez_status": "ez_status",
#         "ez_localisation": "ez_localisation",
#         "myoidnz_status": "myoidnz_status",
#         "myoidnz_localisation": "myoidnz_localisation",
#         "rne_detachment_localisation": "rne_detachment_localisation",
#         "rne_detachment_width": "rne_detachment_width",
#         "rne_detachment_heigt": "rne_detachment_height",
#         "rne_detachment_area": "rne_detachment_area",
#         "гиперрефлективный_материал_локализация": "hyperreflective_material_localisation",
#         "гиперрефлективный_материал_площадь": "hyperreflective_material_area",
#         "препараты_лютеина": "lutein_therapy",
#         "бевацизумаб_авастин": "avastin",
#         "количество_выполненых_инъекций": "avastin_injections",
#         "афлиберцепт_эйлеа": "eylea",
#         "количество_выполненых_инъекций1": "eylea_injections",
#         "бролоцизумаб_визкью": "visque",
#         "количество_выполненых_инъекций2": "visque_injections",
#         "бетаметазон_дипроспан": "diprospan",
#         "количество_выполненых_инъекций3": "diprospan_injections",
#         "триамциналон_кеналог": "kenalog",
#         "количество_выполненых_инъекций4": "kenalog_injections",
#         "луцентис": "lucentis",
#         "количество_выполненых_инъекций5": "lucentis_injections"
#     }
#
#     values = [appointment_id, date_obj.date()]
#     column_names = ["appointment_id", "date"]
#
#     for field in fields:
#         column_names.append(translated.get(field, field))
#         val = row.get(field, None)
#         if isinstance(val, pd.Timestamp):
#             val = val.date()
#         elif isinstance(val, str):
#             val = val.strip()
#         values.append(val)
#
#     placeholders = ", ".join(["?"] * len(values))
#     columns = ", ".join(column_names)
#
#     cursor.execute(f"""
#         INSERT INTO examinations ({columns})
#         VALUES ({placeholders})
#         """, values)
#
#
# def main():
#     df = pd.read_excel("data.xlsx")
#     df = normalize_columns(df)
#
#     conn = sqlite3.connect("medical.db")
#     cursor = conn.cursor()
#
#     create_tables(cursor)
#
#     for _, row in df.iterrows():
#         try:
#             process_row(row, cursor)
#         except Exception as e:
#             print(f"⚠️ Ошибка в строке: {row.get('id_пациента', 'неизвестно')} → {e}")
#
#     conn.commit()
#     conn.close()
#     print("✅ Импорт завершён.")
#
#
# if __name__ == "__main__":
#     main()
# import pandas as pd
# import sqlite3
# from datetime import datetime, date
#
# EXCEL_PATH = "data.xlsx"  # при желании можно подменить на CSV
# DB_PATH = "medical.db"
#
# def normalize_header(s: str) -> str:
#     return (
#         str(s).strip().lower()
#         .replace(" ", "_")
#         .replace(".", "")
#         .replace("(", "")
#         .replace(")", "")
#         .replace("ё", "е")
#     )
#
# def build_ru_to_en_mapping():
#     # Ключи — нормализованные русские заголовки из Excel; значения — целевые английские имена колонок
#     return {
#         # идентификаторы и пациент
#         "id": "row_id",
#         "id_пациента": "patient_id",
#         "фио_пациента": "name",
#         "возраст_в_годах": "date_of_birth",  # по факту у тебя это дата рождения
#         "пол": "sex",
#         "глаз": "eye",
#         "длитзабол_лет": "disease_duration_years",
#         "дата_обследования": "exam_date",
#
#         # базовые параметры
#         "стадия_по_areds": "areds_stage",
#         "тип_неоваскуляризации": "neovascular_type",
#         "топкон_true/false": "topcon",
#         "оптопол_true/false": "optopol",
#         "рефракция": "refraction",
#         "мкоз": "mkoz",
#         "толщина_хориоидеи_в_центре": "choroidal_thickness_center",
#         "цтс_foveola": "cts_foveola",
#         "цтс_supinner_fovea": "cts_sup_inner_fovea",
#         "цтс_supout_fovea": "cts_sup_outer_fovea",
#         "total_volume": "total_volume",
#         "average_volume": "average_volume",
#
#         # RPE/CME
#         "rpe_status": "rpe_status",
#         "rpe_localisation": "rpe_localisation",
#         "cme_localisation": "cme_localisation",
#
#         # Serous RPE detachment (в Excel было "Serouz", выравниваем к serous)
#         "serouz_rpe_detachment_localisation": "serous_rpe_detachment_localisation",
#         "serouz_rpe_detachment_width": "serous_rpe_detachment_width",
#         "serouz_rpe_detachment_height": "serous_rpe_detachment_height",
#         "serouz_rpe_detachment_area": "serous_rpe_detachment_area",
#
#         # Hemorrhagic RPE detachment (исправляем heidgt -> height)
#         "hemorrhagic_rpe_detachment_localisation": "hemorrhagic_rpe_detachment_localisation",
#         "hemorrhagic_rpe_detachment_width": "hemorrhagic_rpe_detachment_width",
#         "hemorrhagic_rpe_detachment_heidgt": "hemorrhagic_rpe_detachment_height",
#         "hemorrhagic_rpe_detachment_area": "hemorrhagic_rpe_detachment_area",
#
#         # Fibrovascular RPE detachment (исправляем heidgt -> height)
#         "fibrovascular__rpe_detachment_localisation": "fibrovascular_rpe_detachment_localisation",
#         "fibrovascular_rpe_detachment_localisation": "fibrovascular_rpe_detachment_localisation",
#         "fibrovascular_rpe_detachment_width": "fibrovascular_rpe_detachment_width",
#         "fibrovascular_rpe_detachment_heidgt": "fibrovascular_rpe_detachment_height",
#         "fibrovascular_rpe_detachment_area": "fibrovascular_rpe_detachment_area",
#
#         # Drusenoid RPE detachment
#         "drusenoid_detachment_rpe_localisation": "drusenoid_rpe_detachment_localisation",
#         "drusenoid_detachment_rpe_width": "drusenoid_rpe_detachment_width",
#         "drusenoid_detachment_rpe_height": "drusenoid_rpe_detachment_height",
#         "drusenoid_detachment_rpe_area": "drusenoid_rpe_detachment_area",
#
#         # Druses (исправляем опечатки: weigt -> weight, heigt -> height, dzuses -> druses)
#         "druses_localisation": "druses_localisation",
#         "druses_weigt": "druses_weight",
#         "druses_heigt": "druses_height",
#         "dzuses_area": "druses_area",
#
#         # Fluid under RPE
#         "fluid_under_rpe_area": "fluid_under_rpe_area",
#         "fluid_under_rpe_localisation": "fluid_under_rpe_localisation",
#
#         # EZ / Myoidnz
#         "ez_status": "ez_status",
#         "ez_localisation": "ez_localisation",
#         "myoidnz_status": "myoidnz_status",
#         "myoidnz_localisation": "myoidnz_localisation",
#
#         # RNE detachment (исправляем heigt -> height)
#         "rne_detachment_localisation": "rne_detachment_localisation",
#         "rne_detachment_width": "rne_detachment_width",
#         "rne_detachment_heigt": "rne_detachment_height",
#         "rne_detachment_area": "rne_detachment_area",
#
#         # Hyperreflective material
#         "гиперрефлективный_материал_локализация": "hyperreflective_material_localisation",
#         "гиперрефлективный_материал_площадь": "hyperreflective_material_area",
#
#         # Drugs + injections
#         "препараты_лютеина": "lutein_therapy",
#         "бевацизумаб_авастин": "avastin",
#         "количество_выполненых_инъекций": "avastin_injections",
#         "афлиберцепт_эйлеа": "eylea",
#         "количество_выполненых_инъекций1": "eylea_injections",
#         "бролоцизумаб_визкью": "visque",
#         "количество_выполненых_инъекций2": "visque_injections",
#         "бетаметазон_дипроспан": "diprospan",
#         "количество_выполненых_инъекций3": "diprospan_injections",
#         "триамциналон_кеналог": "kenalog",
#         "количество_выполненых_инъекций4": "kenalog_injections",
#         "луцентис": "lucentis",
#         "количество_выполненых_инъекций5": "lucentis_injections",
#     }
#
# def rename_columns_to_en(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()
#     df.columns = [normalize_header(c) for c in df.columns]
#     mapping = build_ru_to_en_mapping()
#     df.rename(columns=mapping, inplace=True)
#     return df
#
# def bool_from_any(x):
#     if pd.isna(x):
#         return None
#     if isinstance(x, (int, float)):
#         return int(x) != 0
#     s = str(x).strip().lower()
#     return s in ("1", "true", "yes", "да", "y", "t")
#
# def parse_date_cell(x):
#     if pd.isna(x):
#         return None
#     if isinstance(x, (datetime, pd.Timestamp)):
#         return x.date()
#     s = str(x).strip()
#     # попытка ISO
#     for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%m/%d/%Y"):
#         try:
#             return datetime.strptime(s[:10], fmt).date()
#         except Exception:
#             pass
#     return None
#
# def create_tables(conn: sqlite3.Connection):
#     cur = conn.cursor()
#
#     # Пациенты
#     cur.execute("""
#     CREATE TABLE IF NOT EXISTS patients (
#         id INTEGER PRIMARY KEY,
#         name TEXT,
#         sex TEXT,
#         year_of_birth INTEGER
#     );
#     """)
#
#     # Приёмы
#     cur.execute("""
#     CREATE TABLE IF NOT EXISTS appointments (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         date DATE,
#         disease_duration_years INTEGER,
#         patient_id INTEGER,
#         FOREIGN KEY (patient_id) REFERENCES patients(id)
#     );
#     """)
#
#     # Чтобы гарантированно исправить схему от старых опечаток — дроп и создание examinations
#     cur.execute("DROP TABLE IF EXISTS examinations;")
#
#     # Обследования
#     cur.execute("""
#     CREATE TABLE examinations (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         appointment_id INTEGER,
#         eye TEXT,
#         date DATE,
#         topcon BOOLEAN,
#         optopol BOOLEAN,
#         areds_stage TEXT,
#         refraction TEXT,
#         neovascular_type TEXT,
#         mkoz REAL,
#         choroidal_thickness_center REAL,
#         cts_foveola REAL,
#         cts_sup_inner_fovea REAL,
#         cts_sup_outer_fovea REAL,
#         total_volume REAL,
#         average_volume REAL,
#         rpe_status REAL,
#         rpe_localisation REAL,
#         cme_localisation REAL,
#         serous_rpe_detachment_localisation REAL,
#         serous_rpe_detachment_width REAL,
#         serous_rpe_detachment_height REAL,
#         serous_rpe_detachment_area REAL,
#         hemorrhagic_rpe_detachment_localisation REAL,
#         hemorrhagic_rpe_detachment_width REAL,
#         hemorrhagic_rpe_detachment_height REAL,
#         hemorrhagic_rpe_detachment_area REAL,
#         fibrovascular_rpe_detachment_localisation REAL,
#         fibrovascular_rpe_detachment_width REAL,
#         fibrovascular_rpe_detachment_height REAL,
#         fibrovascular_rpe_detachment_area REAL,
#         drusenoid_rpe_detachment_localisation REAL,
#         drusenoid_rpe_detachment_width REAL,
#         drusenoid_rpe_detachment_height REAL,
#         drusenoid_rpe_detachment_area REAL,
#         druses_localisation REAL,
#         druses_weight REAL,
#         druses_height REAL,
#         druses_area REAL,
#         fluid_under_rpe_area REAL,
#         fluid_under_rpe_localisation REAL,
#         ez_status REAL,
#         ez_localisation REAL,
#         myoidnz_status REAL,
#         myoidnz_localisation REAL,
#         rne_detachment_localisation REAL,
#         rne_detachment_width REAL,
#         rne_detachment_height REAL,
#         rne_detachment_area REAL,
#         hyperreflective_material_localisation REAL,
#         hyperreflective_material_area REAL,
#         lutein_therapy TEXT,
#         avastin TEXT,
#         avastin_injections INTEGER,
#         eylea TEXT,
#         eylea_injections INTEGER,
#         visque TEXT,
#         visque_injections INTEGER,
#         diprospan TEXT,
#         diprospan_injections INTEGER,
#         kenalog TEXT,
#         kenalog_injections INTEGER,
#         lucentis TEXT,
#         lucentis_injections INTEGER,
#         FOREIGN KEY (appointment_id) REFERENCES appointments(id)
#     );
#     """)
#
#     conn.commit()
#
# def upsert_patient(cur, row):
#     patient_id = int(row["patient_id"])
#     name = None if pd.isna(row.get("name")) else str(row["name"]).strip()
#     sex_val = None if pd.isna(row.get("sex")) else str(row["sex"]).strip()[:1].upper()
#
#     # Извлекаем год из date_of_birth (это дата рождения в файле)
#     yob = None
#     dob = row.get("date_of_birth")
#     if isinstance(dob, (datetime, pd.Timestamp, date)):
#         yob = (dob.year if isinstance(dob, date) else dob.date().year)
#     elif isinstance(dob, str):
#         try:
#             yob = datetime.strptime(dob[:10], "%Y-%m-%d").year
#         except Exception:
#             pass
#
#     cur.execute(
#         "INSERT OR IGNORE INTO patients (id, name, sex, year_of_birth) VALUES (?, ?, ?, ?)",
#         (patient_id, name, sex_val, yob)
#     )
#
# def insert_appointment(cur, row):
#     patient_id = int(row["patient_id"])
#     exam_date = parse_date_cell(row.get("exam_date"))
#     duration = None
#     dd = row.get("disease_duration_years")
#     if dd is not None and not pd.isna(dd):
#         try:
#             duration = int(float(dd))
#         except Exception:
#             duration = None
#
#     cur.execute(
#         "INSERT INTO appointments (date, disease_duration_years, patient_id) VALUES (?, ?, ?)",
#         (exam_date, duration, patient_id)
#     )
#     return cur.lastrowid
#
# def insert_examination(cur, row, appointment_id):
#     # Поля examinations (должны совпадать с CREATE TABLE)
#     fields = [
#         "appointment_id",
#         "eye",
#         "date",
#         "topcon",
#         "optopol",
#         "areds_stage",
#         "refraction",
#         "neovascular_type",
#         "mkoz",
#         "choroidal_thickness_center",
#         "cts_foveola",
#         "cts_sup_inner_fovea",
#         "cts_sup_outer_fovea",
#         "total_volume",
#         "average_volume",
#         "rpe_status",
#         "rpe_localisation",
#         "cme_localisation",
#         "serous_rpe_detachment_localisation",
#         "serous_rpe_detachment_width",
#         "serous_rpe_detachment_height",
#         "serous_rpe_detachment_area",
#         "hemorrhagic_rpe_detachment_localisation",
#         "hemorrhagic_rpe_detachment_width",
#         "hemorrhagic_rpe_detachment_height",
#         "hemorrhagic_rpe_detachment_area",
#         "fibrovascular_rpe_detachment_localisation",
#         "fibrovascular_rpe_detachment_width",
#         "fibrovascular_rpe_detachment_height",
#         "fibrovascular_rpe_detachment_area",
#         "drusenoid_rpe_detachment_localisation",
#         "drusenoid_rpe_detachment_width",
#         "drusenoid_rpe_detachment_height",
#         "drusenoid_rpe_detachment_area",
#         "druses_localisation",
#         "druses_weight",
#         "druses_height",
#         "druses_area",
#         "fluid_under_rpe_area",
#         "fluid_under_rpe_localisation",
#         "ez_status",
#         "ez_localisation",
#         "myoidnz_status",
#         "myoidnz_localisation",
#         "rne_detachment_localisation",
#         "rne_detachment_width",
#         "rne_detachment_height",
#         "rne_detachment_area",
#         "hyperreflective_material_localisation",
#         "hyperreflective_material_area",
#         "lutein_therapy",
#         "avastin",
#         "avastin_injections",
#         "eylea",
#         "eylea_injections",
#         "visque",
#         "visque_injections",
#         "diprospan",
#         "diprospan_injections",
#         "kenalog",
#         "kenalog_injections",
#         "lucentis",
#         "lucentis_injections",
#     ]
#
#     # Подготовка значений
#     vals = {}
#     vals["appointment_id"] = appointment_id
#     vals["date"] = parse_date_cell(row.get("exam_date"))
#
#     # Простые переносы + преобразования
#     def get_num(x):
#         if x is None or pd.isna(x):
#             return None
#         try:
#             return float(str(x).replace(",", "."))
#         except Exception:
#             return None
#
#     def get_int(x):
#         if x is None or pd.isna(x):
#             return None
#         try:
#             return int(float(x))
#         except Exception:
#             return None
#
#     def get_str(x):
#         if x is None or pd.isna(x):
#             return None
#         s = str(x).strip()
#         return s if s != "" else None
#
#     # Булевы
#     vals["topcon"] = bool_from_any(row.get("topcon"))
#     vals["optopol"] = bool_from_any(row.get("optopol"))
#
#     # Текстовые
#     for key in [
#         "eye", "areds_stage", "refraction", "neovascular_type", "lutein_therapy",
#         "avastin", "eylea", "visque", "diprospan", "kenalog", "lucentis"
#     ]:
#         vals[key] = get_str(row.get(key))
#
#     # Числовые
#     for key in [
#         "mkoz", "choroidal_thickness_center", "cts_foveola", "cts_sup_inner_fovea",
#         "cts_sup_outer_fovea", "total_volume", "average_volume", "rpe_status",
#         "rpe_localisation", "cme_localisation",
#         "serous_rpe_detachment_localisation", "serous_rpe_detachment_width",
#         "serous_rpe_detachment_height", "serous_rpe_detachment_area",
#         "hemorrhagic_rpe_detachment_localisation", "hemorrhagic_rpe_detachment_width",
#         "hemorrhagic_rpe_detachment_height", "hemorrhagic_rpe_detachment_area",
#         "fibrovascular_rpe_detachment_localisation", "fibrovascular_rpe_detachment_width",
#         "fibrovascular_rpe_detachment_height", "fibrovascular_rpe_detachment_area",
#         "drusenoid_rpe_detachment_localisation", "drusenoid_rpe_detachment_width",
#         "drusenoid_rpe_detachment_height", "drusenoid_rpe_detachment_area",
#         "druses_localisation", "druses_weight", "druses_height", "druses_area",
#         "fluid_under_rpe_area", "fluid_under_rpe_localisation",
#         "ez_status", "ez_localisation", "myoidnz_status", "myoidnz_localisation",
#         "rne_detachment_localisation", "rne_detachment_width", "rne_detachment_height", "rne_detachment_area",
#         "hyperreflective_material_localisation", "hyperreflective_material_area"
#     ]:
#         vals[key] = get_num(row.get(key))
#
#     # Инъекции как целые числа
#     for key in [
#         "avastin_injections", "eylea_injections", "visque_injections",
#         "diprospan_injections", "kenalog_injections", "lucentis_injections"
#     ]:
#         vals[key] = get_int(row.get(key))
#
#     # Формирование и вставка
#     placeholders = ", ".join(["?"] * len(fields))
#     columns = ", ".join(fields)
#     params = [vals.get(k) for k in fields]
#
#     cur.execute(f"INSERT INTO examinations ({columns}) VALUES ({placeholders})", params)
#
# def main():
#     # Загрузка Excel; при отсутствии openpyxl можно сохранить как CSV и заменить на read_csv
#     df = pd.read_excel(EXCEL_PATH)
#     df = rename_columns_to_en(df)
#
#     conn = sqlite3.connect(DB_PATH)
#     cur = conn.cursor()
#
#     create_tables(conn)
#
#     total = 0
#     errors = 0
#     for _, row in df.iterrows():
#         try:
#             # Убедимся, что ключевые поля есть
#             if pd.isna(row.get("patient_id")):
#                 raise ValueError("Missing patient_id")
#
#             # Приведение дат заранее
#             if "date_of_birth" in row and not pd.isna(row["date_of_birth"]):
#                 if isinstance(row["date_of_birth"], str):
#                     try:
#                         row["date_of_birth"] = datetime.strptime(row["date_of_birth"][:10], "%Y-%m-%d").date()
#                     except Exception:
#                         pass
#
#             upsert_patient(cur, row)
#             appointment_id = insert_appointment(cur, row)
#             insert_examination(cur, row, appointment_id)
#             total += 1
#         except Exception as e:
#             errors += 1
#             pid = row.get("patient_id", "unknown")
#             print(f"Warning: row patient_id={pid} failed → {e}")
#
#     conn.commit()
#     conn.close()
#     print(f"Done. Imported: {total}, errors: {errors}")
#
# if __name__ == "__main__":
#     main()
import sqlite3
from datetime import datetime

import pandas as pd

DB_PATH = "medical.db"
EXCEL_PATH = "data.xlsx"


def normalize_columns(df):
    df.columns = [str(c).strip().lower().replace(" ", "_").replace(".", "").replace("(", "").replace(")", "") for c in
                  df.columns]
    return df


def create_injections_table(cursor):
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS injections (
        id                      INTEGER PRIMARY KEY AUTOINCREMENT,
        eye_id                  INTEGER ,
        lutein_therapy          TEXT,
        avastin                 TEXT,
        avastin_injections      INTEGER,
        eylea                   TEXT,
        eylea_injections        INTEGER,
        visque                  TEXT,
        visque_injections       INTEGER,
        diprospan               TEXT,
        diprospan_injections    INTEGER,
        kenalog                 TEXT,
        kenalog_injections      INTEGER,
        lucentis                TEXT,
        lucentis_injections     INTEGER,
        FOREIGN KEY (eye_id) REFERENCES eyes(id)
    );
    """)
def create_table(cursor, text):
    cursor.execute(text)

CREATE_PATIENTS = '''CREATE TABLE IF NOT EXISTS pacients (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    name             VARCHAR(100) ,
    sex              VARCHAR(1)   ,
    year_of_birthday INTEGER
)'''

CREATE_APPOINTMENTS = '''CREATE TABLE IF NOT EXISTS appointments (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    date                    DATE    ,
    duration_of_the_disease INTEGER ,
    pacient_id              INTEGER ,
    FOREIGN KEY (pacient_id) REFERENCES pacients (id)
)'''

CREATE_EYES = '''CREATE TABLE IF NOT EXISTS eyes (
    id                                        INTEGER PRIMARY KEY AUTOINCREMENT,
    eye                                       VARCHAR(2)  ,
    appointment_id                            INTEGER     ,
    date                                      DATE        ,
    duration_of_the_disease                   INTEGER     ,
    topkon                                    BOOLEAN     ,
    optopol                                   BOOLEAN     ,
    areds                                     VARCHAR(10) ,
    refraction                                VARCHAR(10) ,
    type_of_neovascularization                VARCHAR(10) ,
    MKO3                                       NUMERIC     ,
    choroidal_thickness_center                NUMERIC     ,
    cts_foveola                               NUMERIC     ,
    cts_sup_inner_fovea                       NUMERIC     ,
    cts_sup_out_fovea                         NUMERIC     ,
    total_volume                              NUMERIC     ,
    average_volume                            NUMERIC     ,
    rpe_status                                NUMERIC     ,
    rpe_localisation                          NUMERIC     ,
    cme_localisation                          NUMERIC     ,
    serouz_rpe_detachment_localisation        NUMERIC     ,
    serouz_rpe_detachment_width               NUMERIC     ,
    serouz_rpe_detachment_height              NUMERIC     ,
    serouz_rpe_detachment_area                NUMERIC     ,
    hemorrhagic_rpe_detachment_localisation   NUMERIC     ,
    hemorrhagic_rpe_detachment_width          NUMERIC     ,
    hemorrhagic_rpe_detachment_heidgt         NUMERIC     ,
    hemorrhagic_rpe_detachment_area           NUMERIC     ,
    fibrovascular_rpe_detachment_localisation NUMERIC     ,
    fibrovascular_rpe_detachment_width        NUMERIC     ,
    fibrovascular_rpe_detachment_heidgt       NUMERIC     ,
    fibrovascular_rpe_detachment_area         NUMERIC     ,
    drusenoid_detachment_rpe_localisation     NUMERIC     ,
    drusenoid_detachment_rpe_width            NUMERIC     ,
    drusenoid_detachment_rpe_height           NUMERIC     ,
    drusenoid_detachment_rpe_area             NUMERIC     ,
    druses_localisation                       NUMERIC     ,
    druses_weigt                              NUMERIC     ,
    druses_heigt                              NUMERIC     ,
    dzuses_area                               NUMERIC     ,
    fluid_under_rpe_area                      NUMERIC     ,
    fluid_under_rpe_localisation              NUMERIC     ,
    ez_status                                 NUMERIC     ,
    ez_localisation                           NUMERIC     ,
    myoidnz_status                            NUMERIC     ,
    myoidnz_localisation                      NUMERIC     ,
    rne_detachment_localisation               NUMERIC     ,
    rne_detachment_width                      NUMERIC     ,
    rne_detachment_heigt                      NUMERIC     ,
    rne_detachment_area                       NUMERIC     ,
    hyperreflective_material_localisation     NUMERIC     ,
    hyperreflective_material_area             NUMERIC     ,
    FOREIGN KEY (appointment_id) REFERENCES appointments (id)
)'''


def parse_date(value):
    if pd.isna(value):
        return None
    if isinstance(value, datetime):
        return value.date()
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except:
        return None


def boolify(value):
    if pd.isna(value):
        return False
    return str(value).strip().lower() in ["1", "true", "yes", "да"]


def safe_float(value):
    try:
        return float(str(value).replace(",", "."))
    except:
        return 0.0


def safe_int(value):
    try:
        return int(float(value))
    except:
        return 0


def main():
    df = pd.read_excel(EXCEL_PATH)
    df = normalize_columns(df)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    create_table(cursor, CREATE_PATIENTS)
    create_table(cursor, CREATE_APPOINTMENTS)
    create_table(cursor, CREATE_EYES)
    create_injections_table(cursor)
    curPatId = None
    curAppDate = None
    appointment_id = None
    newPat = None
    newApp = None
    pacient_id = None

    for _, row in df.iterrows():
        # 🧑 Insert into pacients
        id = str(row.get("id_пациента", "")).strip()
        name = str(row.get("фио_пациента", "")).strip()
        sex = str(row.get("пол", "")).strip()[:1].upper()
        birth_date = parse_date(row.get("возраст_в_годах"))
        year_of_birth = birth_date.year if birth_date else 1970
        newPat = curPatId is None or curPatId != id
        if (newPat):
            curPatId = id
            cursor.execute("""
                    INSERT OR IGNORE INTO pacients (name, sex, year_of_birthday)
                    VALUES (?, ?, ?)
                    """, (id, sex, year_of_birth))

            pacient_id = cursor.execute("SELECT id FROM pacients WHERE name = ? AND sex = ? AND year_of_birthday = ?",
                                        (id, sex, year_of_birth)).fetchone()[0]

        # 📅 Insert into appointments
        exam_date = parse_date(row.get("дата_обследования"))
        duration = safe_int(row.get("длитзабол_лет"))
        newApp = curAppDate != exam_date or newPat
        if (newApp):
            curAppDate = exam_date
            cursor.execute("""
                                INSERT INTO appointments (date, duration_of_the_disease, pacient_id)
                                VALUES (?, ?, ?)
                                """, (exam_date, duration, pacient_id))

            appointment_id = cursor.lastrowid


        # 👁️ Insert into eyes
        eye_values = (
            str(row.get("глаз", "")).strip(),
            appointment_id,
            exam_date,
            duration,
            boolify(row.get("топкон_true/false")),
            boolify(row.get("оптопол_true/false")),
            str(row.get("стадия_по_areds", "")).strip(),
            str(row.get("рефракция", "")).strip(),
            str(row.get("тип_неоваскуляризации", "")).strip(),
            safe_float(row.get("мкоз")),
            safe_float(row.get("толщина_хориоидеи_в_центре")),
            safe_float(row.get("цтс_foveola")),
            safe_float(row.get("цтс_supinner_fovea")),
            safe_float(row.get("цтс_supout__fovea")),
            safe_float(row.get("total_volume")),
            safe_float(row.get("average_volume")),
            safe_float(row.get("rpe_status")),
            safe_float(row.get("rpe_localisation")),
            safe_float(row.get("cme_localisation")),
            safe_float(row.get("serouz_rpe_detachment_localisation")),
            safe_float(row.get("serouz_rpe_detachment_width")),
            safe_float(row.get("serouz_rpe_detachment_height")),
            safe_float(row.get("serouz_rpe_detachment_area")),
            safe_float(row.get("hemorrhagic_rpe_detachment_localisation")),
            safe_float(row.get("hemorrhagic_rpe_detachment_width")),
            safe_float(row.get("hemorrhagic_rpe_detachment_heidgt")),
            safe_float(row.get("hemorrhagic_rpe_detachment_area")),
            safe_float(row.get("fibrovascular_rpe_detachment_localisation")),
            safe_float(row.get("fibrovascular_rpe_detachment_width")),
            safe_float(row.get("fibrovascular_rpe_detachment_heidgt")),
            safe_float(row.get("fibrovascular_rpe_detachment_area")),
            safe_float(row.get("drusenoid_detachment_rpe_localisation")),
            safe_float(row.get("drusenoid_detachment_rpe_width")),
            safe_float(row.get("drusenoid_detachment_rpe_height")),
            safe_float(row.get("drusenoid_detachment_rpe_area")),
            safe_float(row.get("druses_localisation")),
            safe_float(row.get("druses_weigt")),
            safe_float(row.get("druses_heigt")),
            safe_float(row.get("dzuses_area")),
            safe_float(row.get("fluid_under_rpe_area")),
            safe_float(row.get("fluid_under_rpe_localisation")),
            safe_float(row.get("ez_status")),
            safe_float(row.get("ez_localisation")),
            safe_float(row.get("myoidnz_status")),
            safe_float(row.get("myoidnz_localisation")),
            safe_float(row.get("rne_detachment_localisation")),
            safe_float(row.get("rne_detachment_width")),
            safe_float(row.get("rne_detachment_heigt")),
            safe_float(row.get("rne_detachment_area")),
            safe_float(row.get("гиперрефлективный_материал_локализация")),
            safe_float(row.get("гиперрефлективный_материал_площадь"))
        )

        cursor.execute("""
        INSERT INTO eyes (
            eye, appointment_id, date, duration_of_the_disease, topkon, optopol,
            areds, refraction, type_of_neovascularization, "MKO3", choroidal_thickness_center,
            cts_foveola, cts_sup_inner_fovea, cts_sup_out_fovea, total_volume, average_volume,
            rpe_status, rpe_localisation, cme_localisation,
            serouz_rpe_detachment_localisation, serouz_rpe_detachment_width, serouz_rpe_detachment_height, serouz_rpe_detachment_area,
            hemorrhagic_rpe_detachment_localisation, hemorrhagic_rpe_detachment_width, hemorrhagic_rpe_detachment_heidgt, hemorrhagic_rpe_detachment_area,
            fibrovascular_rpe_detachment_localisation, fibrovascular_rpe_detachment_width, fibrovascular_rpe_detachment_heidgt, fibrovascular_rpe_detachment_area,
            drusenoid_detachment_rpe_localisation, drusenoid_detachment_rpe_width, drusenoid_detachment_rpe_height, drusenoid_detachment_rpe_area,
            druses_localisation, druses_weigt, druses_heigt, dzuses_area,
            fluid_under_rpe_area, fluid_under_rpe_localisation,
            ez_status, ez_localisation, myoidnz_status, myoidnz_localisation,
            rne_detachment_localisation, rne_detachment_width, rne_detachment_heigt, rne_detachment_area,
            hyperreflective_material_localisation, hyperreflective_material_area
        ) VALUES ({})
        """.format(",".join(["?"] * len(eye_values))), eye_values)

        eye_id = cursor.lastrowid

        # 💉 Insert into injections
        # cursor.execute("""
        # INSERT INTO injections (
        #     eye_id, lutein_therapy, avastin, avastin_injections,
        #     eylea, eylea_injections, visque, visque_injections,
        #     diprospan, diprospan_injections, kenalog, kenalog_injections,
        #     lucentis, lucentis_injections
        # ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        # """, (
        #     eye_id,
        #     row.get("препараты_лютеина"),
        #     row.get("бевацизумаб_авастин"),
        #     safe_int(row.get("количество_выполненых_инъекций")),
        #     row.get("афлиберцепт_эйлеа"),
        #     safe_int(row.get("количество_выполненых_инъекций"))
        # ))
        cursor.execute("""
                INSERT INTO injections (
                    eye_id, lutein_therapy, avastin, avastin_injections,
                    eylea, eylea_injections, visque, visque_injections,
                    diprospan, diprospan_injections, kenalog, kenalog_injections,
                    lucentis, lucentis_injections
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
            eye_id,
            row.get("препараты_лютеина"),
            row.get("бевацизумаб_авастин"),
            safe_int(row.get("количество_выполненых_инъекций")),
            row.get("афлиберцепт_эйлеа"),
            safe_int(row.get("количество_выполненых_инъекций1")),
            row.get("бролоцизумаб_визкью"),
            safe_int(row.get("количество_выполненых_инъекций2")),
            row.get("бетаметазон_дипроспан"),
            safe_int(row.get("количество_выполненых_инъекций3")),
            row.get("триамциналон_кеналог"),
            safe_int(row.get("количество_выполненых_инъекций4")),
            row.get("луцентис"),
            safe_int(row.get("количество_выполненых_инъекций5"))
        ))

    conn.commit()
    conn.close()
    print("✅ Импорт завершён успешно.")

if __name__ == "__main__":
    main()

# import pandas as pd
# import sqlite3
# from datetime import datetime
#
# DB_PATH = "medical.db"
# EXCEL_PATH = "data.xlsx"
#
# def normalize_columns(df):
#     df.columns = [str(c).strip().lower().replace(" ", "_").replace(".", "").replace("(", "").replace(")", "") for c in df.columns]
#     return df
#
# def parse_date(value):
#     if value is None or (isinstance(value, float) and pd.isna(value)):
#         return None
#     if isinstance(value, pd.Timestamp):
#         return value.date()
#     if isinstance(value, datetime):
#         return value.date()
#     s = str(value).strip()
#     for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%m/%d/%Y"):
#         try:
#             return datetime.strptime(s[:10], fmt).date()
#         except Exception:
#             pass
#     return None
#
# def boolify(value):
#     if value is None or (isinstance(value, float) and pd.isna(value)):
#         return False
#     return str(value).strip().lower() in ("1", "true", "yes", "y", "да", "t")
#
# def safe_float(value):
#     try:
#         if value is None or (isinstance(value, float) and pd.isna(value)):
#             return 0.0
#         return float(str(value).replace(",", "."))
#     except Exception:
#         return 0.0
#
# def safe_int(value):
#     try:
#         if value is None or (isinstance(value, float) and pd.isna(value)):
#             return 0
#         return int(float(value))
#     except Exception:
#         return 0
#
# def map_eye(code):
#     # Normalize OS/OD → 'OS'/'OD' consistently
#     if code is None:
#         return ""
#     s = str(code).strip().upper()
#     # common variants
#     if s in ("OD", "ОD", "ПРАВ", "RIGHT", "R"):
#         return "OD"
#     if s in ("OS", "ОS", "ЛЕВ", "LEFT", "L"):
#         return "OS"
#     if s in ("OU", "BOTH", "ОБА"):
#         return "OU"
#     return s
#
# def ensure_indexes(cursor):
#     cursor.execute("PRAGMA foreign_keys = ON;")
#     cursor.execute("""
#     CREATE TABLE IF NOT EXISTS injections (
#         id                      INTEGER PRIMARY KEY AUTOINCREMENT,
#         eye_id                  INTEGER NOT NULL,
#         lutein_therapy          TEXT,
#         avastin                 TEXT,
#         avastin_injections      INTEGER,
#         eylea                   TEXT,
#         eylea_injections        INTEGER,
#         visque                  TEXT,
#         visque_injections       INTEGER,
#         diprospan               TEXT,
#         diprospan_injections    INTEGER,
#         kenalog                 TEXT,
#         kenalog_injections      INTEGER,
#         lucentis                TEXT,
#         lucentis_injections     INTEGER,
#         FOREIGN KEY (eye_id) REFERENCES eyes(id)
#     );
#     """)
#     # cursor.execute("""
#     # CREATE UNIQUE INDEX IF NOT EXISTS ux_appointments_patient_date
#     # ON appointments (pacient_id, date);
#     # """)
#
# def create_table(cursor, text):
#     cursor.execute(text)
#
# CREATE_PATIENTS = '''CREATE TABLE IF NOT EXISTS pacients (
#     id               INTEGER PRIMARY KEY AUTOINCREMENT,
#     name             VARCHAR(100) ,
#     sex              VARCHAR(1)   ,
#     year_of_birthday INTEGER
# )'''
#
# CREATE_APPOINTMENTS = '''CREATE TABLE IF NOT EXISTS appointments (
#     id                      INTEGER PRIMARY KEY AUTOINCREMENT,
#     date                    DATE    ,
#     duration_of_the_disease INTEGER ,
#     pacient_id              INTEGER ,
#     FOREIGN KEY (pacient_id) REFERENCES pacients (id)
# )'''
#
# CREATE_EYES = '''CREATE TABLE IF NOT EXISTS eyes (
#     id                                        INTEGER PRIMARY KEY AUTOINCREMENT,
#     eye                                       VARCHAR(2)  ,
#     appointment_id                            INTEGER     ,
#     date                                      DATE        ,
#     duration_of_the_disease                   INTEGER     ,
#     topkon                                    BOOLEAN     ,
#     optopol                                   BOOLEAN     ,
#     areds                                     VARCHAR(10) ,
#     refraction                                VARCHAR(10) ,
#     type_of_neovascularization                VARCHAR(10) ,
#     "МКОЗ"                                    NUMERIC     ,
#     choroidal_thickness_center                NUMERIC     ,
#     cts_foveola                               NUMERIC     ,
#     cts_sup_inner_fovea                       NUMERIC     ,
#     total_volume                              NUMERIC     ,
#     average_volume                            NUMERIC     ,
#     rpe_status                                NUMERIC     ,
#     rpe_localisation                          NUMERIC     ,
#     cme_localisation                          NUMERIC     ,
#     serouz_rpe_detachment_localisation        NUMERIC     ,
#     serouz_rpe_detachment_width               NUMERIC     ,
#     serouz_rpe_detachment_height              NUMERIC     ,
#     serouz_rpe_detachment_area                NUMERIC     ,
#     hemorrhagic_rpe_detachment_localisation   NUMERIC     ,
#     hemorrhagic_rpe_detachment_width          NUMERIC     ,
#     hemorrhagic_rpe_detachment_heidgt         NUMERIC     ,
#     hemorrhagic_rpe_detachment_area           NUMERIC     ,
#     fibrovascular_rpe_detachment_localisation NUMERIC     ,
#     fibrovascular_rpe_detachment_width        NUMERIC     ,
#     fibrovascular_rpe_detachment_heidgt       NUMERIC     ,
#     fibrovascular_rpe_detachment_area         NUMERIC     ,
#     drusenoid_detachment_rpe_localisation     NUMERIC     ,
#     drusenoid_detachment_rpe_width            NUMERIC     ,
#     drusenoid_detachment_rpe_height           NUMERIC     ,
#     drusenoid_detachment_rpe_area             NUMERIC     ,
#     druses_localisation                       NUMERIC     ,
#     druses_weigt                              NUMERIC     ,
#     druses_heigt                              NUMERIC     ,
#     dzuses_area                               NUMERIC     ,
#     fluid_under_rpe_area                      NUMERIC     ,
#     fluid_under_rpe_localisation              NUMERIC     ,
#     ez_status                                 NUMERIC     ,
#     ez_localisation                           NUMERIC     ,
#     myoidnz_status                            NUMERIC     ,
#     myoidnz_localisation                      NUMERIC     ,
#     rne_detachment_localisation               NUMERIC     ,
#     rne_detachment_width                      NUMERIC     ,
#     rne_detachment_heigt                      NUMERIC     ,
#     rne_detachment_area                       NUMERIC     ,
#     hyperreflective_material_localisation     NUMERIC     ,
#     hyperreflective_material_area             NUMERIC     ,
#     FOREIGN KEY (appointment_id) REFERENCES appointments (id)
# )'''
#
# def get_or_create_pacient(cursor, name, sex, year_of_birth):
#     # Try to find existing patient by the same triple
#     row = cursor.execute(
#         "SELECT id FROM pacients WHERE name = ? AND sex = ? AND year_of_birthday = ?",
#         (name, sex, year_of_birth)
#     ).fetchone()
#     if row:
#         return row[0]
#     cursor.execute(
#         "INSERT INTO pacients (name, sex, year_of_birthday) VALUES (?, ?, ?)",
#         (name, sex, year_of_birth)
#     )
#     return cursor.lastrowid
#
# def get_or_create_appointment(cursor, pacient_id, date_obj, duration):
#     # Reuse existing appointment for the same patient and date
#     row = cursor.execute(
#         "SELECT id FROM appointments WHERE pacient_id = ? AND date = ?",
#         (pacient_id, date_obj)
#     ).fetchone()
#     if row:
#         # Optionally update duration if provided and different
#         if duration is not None:
#             cursor.execute(
#                 "UPDATE appointments SET duration_of_the_disease = COALESCE(?, duration_of_the_disease) WHERE id = ?",
#                 (duration, row[0])
#             )
#         return row[0]
#     cursor.execute(
#         "INSERT INTO appointments (date, duration_of_the_disease, pacient_id) VALUES (?, ?, ?)",
#         (date_obj, duration if duration is not None else 0, pacient_id)
#     )
#     return cursor.lastrowid
#
# def insert_eye(cursor, appointment_id, row):
#     eye_values = (
#         map_eye(row.get("глаз")),
#         appointment_id,
#         parse_date(row.get("дата_обследования")),
#         safe_int(row.get("длитзабол_лет")),
#         boolify(row.get("топкон_true/false")),
#         boolify(row.get("оптопол_true/false")),
#         str(row.get("стадия_по_areds") or "").strip(),
#         str(row.get("рефракция") or "").strip(),
#         str(row.get("тип_неоваскуляризации") or "").strip(),
#         safe_float(row.get("мкоз")),
#         safe_float(row.get("толщина_хориоидеи_в_центре")),
#         safe_float(row.get("цтс_foveola")),
#         safe_float(row.get("цтс_supinner_fovea")),
#         safe_float(row.get("цтс_supout_fovea")),
#         safe_float(row.get("total_volume")),
#         safe_float(row.get("average_volume")),
#         safe_float(row.get("rpe_status")),
#         safe_float(row.get("rpe_localisation")),
#         safe_float(row.get("cme_localisation")),
#         safe_float(row.get("serouz_rpe_detachment_localisation")),
#         safe_float(row.get("serouz_rpe_detachment_width")),
#         safe_float(row.get("serouz_rpe_detachment_height")),
#         safe_float(row.get("serouz_rpe_detachment_area")),
#         safe_float(row.get("hemorrhagic_rpe_detachment_localisation")),
#         safe_float(row.get("hemorrhagic_rpe_detachment_width")),
#         safe_float(row.get("hemorrhagic_rpe_detachment_heidgt")),
#         safe_float(row.get("hemorrhagic_rpe_detachment_area")),
#         safe_float(row.get("fibrovascular_rpe_detachment_localisation")),
#         safe_float(row.get("fibrovascular_rpe_detachment_width")),
#         safe_float(row.get("fibrovascular_rpe_detachment_heidgt")),
#         safe_float(row.get("fibrovascular_rpe_detachment_area")),
#         safe_float(row.get("drusenoid_detachment_rpe_localisation")),
#         safe_float(row.get("drusenoid_detachment_rpe_width")),
#         safe_float(row.get("drusenoid_detachment_rpe_height")),
#         safe_float(row.get("drusenoid_detachment_rpe_area")),
#         safe_float(row.get("druses_localisation")),
#         safe_float(row.get("druses_weigt")),
#         safe_float(row.get("druses_heigt")),
#         safe_float(row.get("dzuses_area")),
#         safe_float(row.get("fluid_under_rpe_area")),
#         safe_float(row.get("fluid_under_rpe_localisation")),
#         safe_float(row.get("ez_status")),
#         safe_float(row.get("ez_localisation")),
#         safe_float(row.get("myoidnz_status")),
#         safe_float(row.get("myoidnz_localisation")),
#         safe_float(row.get("rne_detachment_localisation")),
#         safe_float(row.get("rne_detachment_width")),
#         safe_float(row.get("rne_detachment_heigt")),
#         safe_float(row.get("rne_detachment_area")),
#         safe_float(row.get("гиперрефлективный_материал_локализация")),
#         safe_float(row.get("гиперрефлективный_материал_площадь")),
#     )
#
#     cursor.execute("""
#     INSERT INTO eyes (
#         eye, appointment_id, date, duration_of_the_disease, topkon, optopol,
#         areds, refraction, type_of_neovascularization, "МКОЗ", choroidal_thickness_center,
#         cts_foveola, cts_sup_inner_fovea, cts_sup_inner_fovea, total_volume, average_volume,
#         rpe_status, rpe_localisation, cme_localisation,
#         serouz_rpe_detachment_localisation, serouz_rpe_detachment_width, serouz_rpe_detachment_height, serouz_rpe_detachment_area,
#         hemorrhagic_rpe_detachment_localisation, hemorrhagic_rpe_detachment_width, hemorrhagic_rpe_detachment_heidgt, hemorrhagic_rpe_detachment_area,
#         fibrovascular_rpe_detachment_localisation, fibrovascular_rpe_detachment_width, fibrovascular_rpe_detachment_heidgt, fibrovascular_rpe_detachment_area,
#         drusenoid_detachment_rpe_localisation, drusenoid_detachment_rpe_width, drusenoid_detachment_rpe_height, drusenoid_detachment_rpe_area,
#         druses_localisation, druses_weigt, druses_heigt, dzuses_area,
#         fluid_under_rpe_area, fluid_under_rpe_localisation,
#         ez_status, ez_localisation, myoidnz_status, myoidnz_localisation,
#         rne_detachment_localisation, rne_detachment_width, rne_detachment_heigt, rne_detachment_area,
#         hyperreflective_material_localisation, hyperreflective_material_area
#     ) VALUES ({})
#     """.format(",".join(["?"] * len(eye_values))), eye_values)
#
#     return cursor.lastrowid
#
# def insert_injections(cursor, eye_id, row):
#     cursor.execute("""
#     INSERT INTO injections (
#         eye_id, lutein_therapy, avastin, avastin_injections,
#         eylea, eylea_injections, visque, visque_injections,
#         diprospan, diprospan_injections, kenalog, kenalog_injections,
#         lucentis, lucentis_injections
#     ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
#     """, (
#         eye_id,
#         (row.get("препараты_лютеина") or None),
#         (row.get("бевацизумаб_авастин") or None),
#         safe_int(row.get("количество_выполненых_инъекций")),
#         (row.get("афлиберцепт_эйлеа") or None),
#         safe_int(row.get("количество_выполненых_инъекций1")),
#         (row.get("бролоцизумаб_визкью") or None),
#         safe_int(row.get("количество_выполненых_инъекций2")),
#         (row.get("бетаметазон_дипроспан") or None),
#         safe_int(row.get("количество_выполненых_инъекций3")),
#         (row.get("триамциналон_кеналог") or None),
#         safe_int(row.get("количество_выполненых_инъекций4")),
#         (row.get("луцентис") or None),
#         safe_int(row.get("количество_выполненых_инъекций5")),
#     ))
#
# def main():
#     df = pd.read_excel(EXCEL_PATH)
#     df = normalize_columns(df)
#
#     conn = sqlite3.connect(DB_PATH)
#     cur = conn.cursor()
#     create_table(cur, CREATE_PATIENTS)
#     create_table(cur, CREATE_APPOINTMENTS)
#     create_table(cur, CREATE_EYES)
#     ensure_indexes(cur)
#
#     imported, errors = 0, 0
#
#     for _, row in df.iterrows():
#         try:
#             # Start transaction per row
#             conn.execute("BEGIN")
#
#             # Patient
#             name = str(row.get("фио_пациента", "")).strip()
#             sex = str(row.get("пол", "")).strip()[:1].upper()
#             birth_date = parse_date(row.get("возраст_в_годах"))
#             year_of_birth = birth_date.year if birth_date else 1970
#
#             pacient_id = get_or_create_pacient(cur, name, sex, year_of_birth)
#
#             # Appointment (get or create by patient/date)
#             exam_date = parse_date(row.get("дата_обследования"))
#             duration = safe_int(row.get("длитзабол_лет"))
#             appointment_id = get_or_create_appointment(cur, pacient_id, exam_date, duration)
#
#             # Eye
#             eye_id = insert_eye(cur, appointment_id, row)
#
#             # Injections
#             insert_injections(cur, eye_id, row)
#
#             conn.commit()
#             imported += 1
#         except Exception as e:
#             conn.rollback()
#             errors += 1
#             print(f"Warn: patient='{name}', date={exam_date} → {e}")
#
#     conn.close()
#     print(f"Done. Imported: {imported}, errors: {errors}")
#
# if __name__ == "__main__":
#     main()