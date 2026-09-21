"""Схема базы данных клиники.

Единственное место, где описан состав таблиц: используется, когда пользователь
соглашается создать отсутствующий файл базы. Все запросы с ``IF NOT EXISTS``,
поэтому существующие базы не меняются.

Типы колонок повторяют рабочую ``sqlite/ database.db``; ограничения NOT NULL
намеренно не ставим — SQLite их почти не проверяет, а приложение вставляет
строки частично (например, только поля одного блока формы).
"""

from __future__ import annotations

from pathlib import Path

#: Имена таблиц.
PACIENTS_TABLE = "pacients"
APPOINTMENTS_TABLE = "appointments"
EYES_TABLE = "eyes"
INJECTIONS_TABLE = "injections"

#: Пациенты.
CREATE_PACIENTS = """
CREATE TABLE IF NOT EXISTS pacients (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    name             VARCHAR(100),
    sex              VARCHAR(1),
    year_of_birthday INTEGER
)
""".strip()

#: Приёмы пациента.
CREATE_APPOINTMENTS = """
CREATE TABLE IF NOT EXISTS appointments (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    date                    DATE,
    duration_of_the_disease INTEGER,
    pacient_id              INTEGER,
    FOREIGN KEY (pacient_id) REFERENCES pacients (id)
)
""".strip()

#: Данные глаза по приёму. Колонки совпадают с constants.COLUMNS_EYES и
#: COLUMN_INDEX_TO_FIELD_NAME (включая сохранившиеся опечатки имён).
CREATE_EYES = """
CREATE TABLE IF NOT EXISTS eyes (
    id                                        INTEGER PRIMARY KEY AUTOINCREMENT,
    eye                                       VARCHAR(2),
    appointment_id                            INTEGER,
    date                                      DATE,
    duration_of_the_disease                   INTEGER,
    topkon                                    BOOLEAN,
    optopol                                   BOOLEAN,
    areds                                     VARCHAR(10),
    refraction                                VARCHAR(10),
    type_of_neovascularization                VARCHAR(10),
    MKO3                                      NUMERIC,
    choroidal_thickness_center                NUMERIC,
    cts_foveola                               NUMERIC,
    cts_sup_inner_fovea                       NUMERIC,
    cts_sup_out_fovea                         NUMERIC,
    total_volume                              NUMERIC,
    average_volume                            NUMERIC,
    rpe_status                                NUMERIC,
    rpe_localisation                          NUMERIC,
    cme_localisation                          NUMERIC,
    serouz_rpe_detachment_localisation        NUMERIC,
    serouz_rpe_detachment_width               NUMERIC,
    serouz_rpe_detachment_height              NUMERIC,
    serouz_rpe_detachment_area                NUMERIC,
    hemorrhagic_rpe_detachment_localisation   NUMERIC,
    hemorrhagic_rpe_detachment_width          NUMERIC,
    hemorrhagic_rpe_detachment_heidgt         NUMERIC,
    hemorrhagic_rpe_detachment_area           NUMERIC,
    fibrovascular_rpe_detachment_localisation NUMERIC,
    fibrovascular_rpe_detachment_width        NUMERIC,
    fibrovascular_rpe_detachment_heidgt       NUMERIC,
    fibrovascular_rpe_detachment_area         NUMERIC,
    drusenoid_detachment_rpe_localisation     NUMERIC,
    drusenoid_detachment_rpe_width            NUMERIC,
    drusenoid_detachment_rpe_height           NUMERIC,
    drusenoid_detachment_rpe_area             NUMERIC,
    druses_localisation                       NUMERIC,
    druses_weigt                              NUMERIC,
    druses_heigt                              NUMERIC,
    dzuses_area                               NUMERIC,
    fluid_under_rpe_area                      NUMERIC,
    fluid_under_rpe_localisation              NUMERIC,
    ez_status                                 NUMERIC,
    ez_localisation                           NUMERIC,
    myoidnz_status                            NUMERIC,
    myoidnz_localisation                      NUMERIC,
    rne_detachment_localisation               NUMERIC,
    rne_detachment_width                      NUMERIC,
    rne_detachment_heigt                      NUMERIC,
    rne_detachment_area                       NUMERIC,
    hyperreflective_material_localisation     NUMERIC,
    hyperreflective_material_area             NUMERIC,
    FOREIGN KEY (appointment_id) REFERENCES appointments (id)
)
""".strip()

#: Инъекции по глазу (одна строка на запись eyes).
CREATE_INJECTIONS = """
CREATE TABLE IF NOT EXISTS injections (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    eye_id               INTEGER,
    lutein_therapy       TEXT,
    avastin              TEXT,
    avastin_injections   INTEGER,
    eylea                TEXT,
    eylea_injections     INTEGER,
    visque               TEXT,
    visque_injections    INTEGER,
    diprospan            TEXT,
    diprospan_injections INTEGER,
    kenalog              TEXT,
    kenalog_injections   INTEGER,
    lucentis             TEXT,
    lucentis_injections  INTEGER,
    FOREIGN KEY (eye_id) REFERENCES eyes(id)
)
""".strip()

#: Все запросы создания в порядке применения.
CREATE_STATEMENTS: tuple[str, ...] = (
    CREATE_PACIENTS,
    CREATE_APPOINTMENTS,
    CREATE_EYES,
    CREATE_INJECTIONS,
)


def create_schema(cursor) -> None:
    """Создаёт все таблицы схемы через DB-API курсор (sqlite3)."""
    for statement in CREATE_STATEMENTS:
        cursor.execute(statement)


def create_database_file(path: Path | str) -> Path:
    """Создаёт новый файл базы со схемой и возвращает путь.

    Используется, когда пользователь согласился создать отсутствующую базу.
    Каталог создаётся при необходимости.
    """
    import sqlite3

    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(db_path)
    try:
        with connection:
            create_schema(connection.cursor())
    finally:
        connection.close()
    return db_path
