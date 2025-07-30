import sqlite3

from PySide6.QtSql import QSqlQuery, QSqlDatabase
from bsmu.vision.core.plugins import Plugin


class DatabaseManager(Plugin):
    CREATE_PATIENTS = '''CREATE TABLE IF NOT EXISTS pacients (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    name             VARCHAR(100) NOT NULL,
    sex              VARCHAR(1)   NOT NULL,
    year_of_birthday INTEGER      NOT NULL
)'''

    CREATE_APPOINTMENTS = '''CREATE TABLE IF NOT EXISTS appointments (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    date                    DATE    NOT NULL,
    duration_of_the_disease INTEGER NOT NULL,
    pacient_id              INTEGER NOT NULL,
    FOREIGN KEY (pacient_id) REFERENCES pacients (id)
)'''

    CREATE_EYES = '''CREATE TABLE IF NOT EXISTS eyes (
    id                                        INTEGER PRIMARY KEY AUTOINCREMENT,
    eye                                       VARCHAR(2)  NOT NULL,
    appointment_id                            INTEGER     NOT NULL,
    date                                      DATE        NOT NULL,
    duration_of_the_disease                   INTEGER     NOT NULL,
    topkon                                    BOOLEAN     NOT NULL,
    optopol                                   BOOLEAN     NOT NULL,
    areds                                     VARCHAR(10) NOT NULL,
    refraction                                VARCHAR(10) NOT NULL,
    type_of_neovascularization                VARCHAR(10) NOT NULL,
    "МКОЗ"                                    NUMERIC     NOT NULL,
    choroidal_thickness_center                NUMERIC     NOT NULL,
    cts_foveola                               NUMERIC     NOT NULL,
    cts_sup_inner_fovea                       NUMERIC     NOT NULL,
    cts_sup_out_fovea                         NUMERIC     NOT NULL,
    total_volume                              NUMERIC     NOT NULL,
    average_volume                            NUMERIC     NOT NULL,
    rpe_status                                NUMERIC     NOT NULL,
    rpe_localisation                          NUMERIC     NOT NULL,
    cme_localisation                          NUMERIC     NOT NULL,
    serouz_rpe_detachment_localisation        NUMERIC     NOT NULL,
    serouz_rpe_detachment_width               NUMERIC     NOT NULL,
    serouz_rpe_detachment_height              NUMERIC     NOT NULL,
    serouz_rpe_detachment_area                NUMERIC     NOT NULL,
    hemorrhagic_rpe_detachment_localisation   NUMERIC     NOT NULL,
    hemorrhagic_rpe_detachment_width          NUMERIC     NOT NULL,
    hemorrhagic_rpe_detachment_heidgt         NUMERIC     NOT NULL,
    hemorrhagic_rpe_detachment_area           NUMERIC     NOT NULL,
    fibrovascular_rpe_detachment_localisation NUMERIC     NOT NULL,
    fibrovascular_rpe_detachment_width        NUMERIC     NOT NULL,
    fibrovascular_rpe_detachment_heidgt       NUMERIC     NOT NULL,
    fibrovascular_rpe_detachment_area         NUMERIC     NOT NULL,
    drusenoid_detachment_rpe_localisation     NUMERIC     NOT NULL,
    drusenoid_detachment_rpe_width            NUMERIC     NOT NULL,
    drusenoid_detachment_rpe_height           NUMERIC     NOT NULL,
    drusenoid_detachment_rpe_area             NUMERIC     NOT NULL,
    druses_localisation                       NUMERIC     NOT NULL,
    druses_weigt                              NUMERIC     NOT NULL,
    druses_heigt                              NUMERIC     NOT NULL,
    dzuses_area                               NUMERIC     NOT NULL,
    fluid_under_rpe_area                      NUMERIC     NOT NULL,
    fluid_under_rpe_localisation              NUMERIC     NOT NULL,
    ez_status                                 NUMERIC     NOT NULL,
    ez_localisation                           NUMERIC     NOT NULL,
    myoidnz_status                            NUMERIC     NOT NULL,
    myoidnz_localisation                      NUMERIC     NOT NULL,
    rne_detachment_localisation               NUMERIC     NOT NULL,
    rne_detachment_width                      NUMERIC     NOT NULL,
    rne_detachment_heigt                      NUMERIC     NOT NULL,
    rne_detachment_area                       NUMERIC     NOT NULL,
    hyperreflective_material_localisation     NUMERIC     NOT NULL,
    hyperreflective_material_area             NUMERIC     NOT NULL,
    FOREIGN KEY (appointment_id) REFERENCES appointments (id)
)'''

    CREATE_MEDICINES = '''CREATE TABLE IF NOT EXISTS medicines (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    type           VARCHAR(5) NOT NULL,
    amount         INTEGER    NOT NULL,
    appointment_id INTEGER    NOT NULL,
    FOREIGN KEY (appointment_id) REFERENCES appointments (id)
)'''

    # Pacients = "INSERT INTO pacients (name, sex, year_of_birthday) VALUES (?, ?, ?)"
    Pacients = """INSERT INTO pacients (name, sex, year_of_birthday) VALUES ('Иван Иванов', 'М', 1990)
,('Мария Смирнова', 'Ж', 1995)
,('Алексей Петров', 'М', 1988)
,('Екатерина Фролова', 'Ж', 1992)
,('Дмитрий Сидоров', 'М', 1985)
,('Анна Кузнецова', 'Ж', 1997)
,('Сергей Васильев', 'М', 1982)
,('Ольга Попова', 'Ж', 1990)
,('Николай Орлов', 'М', 1989)
,('Елена Михайлова', 'Ж', 1994);"""
    # Appointments = "INSERT INTO appointments (date, duration_of_the_disease, pacient_id) VALUES (?, ?, ?)"
    Appointments = """INSERT INTO appointments (date, duration_of_the_disease, pacient_id) VALUES ('2025-04-01', 10, 1)
,('2025-04-15', 5, 1)

,('2025-03-20', 14, 2)

,('2025-02-28', 7, 3)
,('2025-04-10', 3, 3)

,('2025-01-15', 20, 4)

,('2025-03-05', 11, 5)
,('2025-04-22', 9, 5)

,('2025-02-18', 15, 6)

,('2025-03-01', 8, 7)
,('2025-03-25', 6, 7)

,('2025-04-03', 10, 8)

,('2025-02-14', 18, 9)
,('2025-03-30', 12, 9)

,('2025-04-12', 7, 10);
    """
    Appointmen_1_1 = """INSERT INTO eyes (
    eye, appointment_id, date, duration_of_the_disease, topkon, optopol, areds,
    refraction, type_of_neovascularization, "МКОЗ", choroidal_thickness_center,
    cts_foveola, cts_sup_inner_fovea, cts_sup_out_fovea, total_volume, average_volume,
    rpe_status, rpe_localisation, cme_localisation, serouz_rpe_detachment_localisation,
    serouz_rpe_detachment_width, serouz_rpe_detachment_height, serouz_rpe_detachment_area,
    hemorrhagic_rpe_detachment_localisation, hemorrhagic_rpe_detachment_width,
    hemorrhagic_rpe_detachment_heidgt, hemorrhagic_rpe_detachment_area,
    fibrovascular_rpe_detachment_localisation, fibrovascular_rpe_detachment_width,
    fibrovascular_rpe_detachment_heidgt, fibrovascular_rpe_detachment_area,
    drusenoid_detachment_rpe_localisation, drusenoid_detachment_rpe_width,
    drusenoid_detachment_rpe_height, drusenoid_detachment_rpe_area,
    druses_localisation, druses_weigt, druses_heigt, dzuses_area,
    fluid_under_rpe_area, fluid_under_rpe_localisation, ez_status, ez_localisation,
    myoidnz_status, myoidnz_localisation, rne_detachment_localisation,
    rne_detachment_width, rne_detachment_heigt, rne_detachment_area,
    hyperreflective_material_localisation, hyperreflective_material_area
) VALUES (
    'L', 1, '2025-04-15', 10, FALSE, TRUE, 'AREDS1',
    '-2.0', 'Occult', 300, 160, 210, 190, 200, 9.0, 3.5,
    0, 2, 1, 1, 3.0, 2.0, 1.0,
    1, 1.5, 1.0, 0.5, 2, 3.5, 3.0, 1.5,
    1, 0.7, 0.5, 0.3, 3, 4.0, 2.5, 1.0,
    0.8, 2.0, 1, 2, 1, 1, 0,
    3.8, 2.0, 0, 0, 0
),(
    'R', 1, '2025-04-16', 5, TRUE, FALSE, 'AREDS2',
    '-1.5', 'Classic', 250, 150, 200, 180, 190, 8.5, 3.2,
    1, 1, 2, 0, 2.5, 1.5, 0.5,
    0, 1.2, 0.8, 0.3, 1, 2.8, 2.4, 1.0,
    0, 0.5, 0.3, 0.2, 2, 3.1, 1.8, 0.7,
    0.6, 1.5, 1, 2, 1, 1, 0,
    3.2, 1.5, 0, 0, 0
),(
    'L', 2, '2025-04-15', 10, FALSE, TRUE, 'AREDS1',
    '-2.0', 'Occult', 300, 160, 210, 190, 200, 9.0, 3.5,
    0, 2, 1, 1, 3.0, 2.0, 1.0,
    1, 1.5, 1.0, 0.5, 2, 3.5, 3.0, 1.5,
    1, 0.7, 0.5, 0.3, 3, 4.0, 2.5, 1.0,
    0.8, 2.0, 1, 2, 1, 1, 0,
    3.8, 2.0, 0, 0, 0
),(
    'R', 2, '2025-04-16', 5, TRUE, FALSE, 'AREDS2',
    '-1.5', 'Classic', 250, 150, 200, 180, 190, 8.5, 3.2,
    1, 1, 2, 0, 2.5, 1.5, 0.5,
    0, 1.2, 0.8, 0.3, 1, 2.8, 2.4, 1.0,
    0, 0.5, 0.3, 0.2, 2, 3.1, 1.8, 0.7,
    0.6, 1.5, 1, 2, 1, 1, 0,
    3.2, 1.5, 0, 0, 0
),(
    'L', 3, '2025-04-15', 10, FALSE, TRUE, 'AREDS1',
    '-2.0', 'Occult', 300, 160, 210, 190, 200, 9.0, 3.5,
    0, 2, 1, 1, 3.0, 2.0, 1.0,
    1, 1.5, 1.0, 0.5, 2, 3.5, 3.0, 1.5,
    1, 0.7, 0.5, 0.3, 3, 4.0, 2.5, 1.0,
    0.8, 2.0, 1, 2, 1, 1, 0,
    3.8, 2.0, 0, 0, 0
),(
    'R', 3, '2025-04-16', 5, TRUE, FALSE, 'AREDS2',
    '-1.5', 'Classic', 250, 150, 200, 180, 190, 8.5, 3.2,
    1, 1, 2, 0, 2.5, 1.5, 0.5,
    0, 1.2, 0.8, 0.3, 1, 2.8, 2.4, 1.0,
    0, 0.5, 0.3, 0.2, 2, 3.1, 1.8, 0.7,
    0.6, 1.5, 1, 2, 1, 1, 0,
    3.2, 1.5, 0, 0, 0
),(
    'L', 4, '2025-04-15', 10, FALSE, TRUE, 'AREDS1',
    '-2.0', 'Occult', 300, 160, 210, 190, 200, 9.0, 3.5,
    0, 2, 1, 1, 3.0, 2.0, 1.0,
    1, 1.5, 1.0, 0.5, 2, 3.5, 3.0, 1.5,
    1, 0.7, 0.5, 0.3, 3, 4.0, 2.5, 1.0,
    0.8, 2.0, 1, 2, 1, 1, 0,
    3.8, 2.0, 0, 0, 0
),(
    'R', 4, '2025-04-16', 5, TRUE, FALSE, 'AREDS2',
    '-1.5', 'Classic', 250, 150, 200, 180, 190, 8.5, 3.2,
    1, 1, 2, 0, 2.5, 1.5, 0.5,
    0, 1.2, 0.8, 0.3, 1, 2.8, 2.4, 1.0,
    0, 0.5, 0.3, 0.2, 2, 3.1, 1.8, 0.7,
    0.6, 1.5, 1, 2, 1, 1, 0,
    3.2, 1.5, 0, 0, 0
),(
    'L', 5, '2025-04-15', 10, FALSE, TRUE, 'AREDS1',
    '-2.0', 'Occult', 300, 160, 210, 190, 200, 9.0, 3.5,
    0, 2, 1, 1, 3.0, 2.0, 1.0,
    1, 1.5, 1.0, 0.5, 2, 3.5, 3.0, 1.5,
    1, 0.7, 0.5, 0.3, 3, 4.0, 2.5, 1.0,
    0.8, 2.0, 1, 2, 1, 1, 0,
    3.8, 2.0, 0, 0, 0
),(
    'R', 5, '2025-04-16', 5, TRUE, FALSE, 'AREDS2',
    '-1.5', 'Classic', 250, 150, 200, 180, 190, 8.5, 3.2,
    1, 1, 2, 0, 2.5, 1.5, 0.5,
    0, 1.2, 0.8, 0.3, 1, 2.8, 2.4, 1.0,
    0, 0.5, 0.3, 0.2, 2, 3.1, 1.8, 0.7,
    0.6, 1.5, 1, 2, 1, 1, 0,
    3.2, 1.5, 0, 0, 0
),(
    'L', 6, '2025-04-15', 10, FALSE, TRUE, 'AREDS1',
    '-2.0', 'Occult', 300, 160, 210, 190, 200, 9.0, 3.5,
    0, 2, 1, 1, 3.0, 2.0, 1.0,
    1, 1.5, 1.0, 0.5, 2, 3.5, 3.0, 1.5,
    1, 0.7, 0.5, 0.3, 3, 4.0, 2.5, 1.0,
    0.8, 2.0, 1, 2, 1, 1, 0,
    3.8, 2.0, 0, 0, 0
),(
    'R', 6, '2025-04-16', 5, TRUE, FALSE, 'AREDS2',
    '-1.5', 'Classic', 250, 150, 200, 180, 190, 8.5, 3.2,
    1, 1, 2, 0, 2.5, 1.5, 0.5,
    0, 1.2, 0.8, 0.3, 1, 2.8, 2.4, 1.0,
    0, 0.5, 0.3, 0.2, 2, 3.1, 1.8, 0.7,
    0.6, 1.5, 1, 2, 1, 1, 0,
    3.2, 1.5, 0, 0, 0
),(
    'L', 7, '2025-04-15', 10, FALSE, TRUE, 'AREDS1',
    '-2.0', 'Occult', 300, 160, 210, 190, 200, 9.0, 3.5,
    0, 2, 1, 1, 3.0, 2.0, 1.0,
    1, 1.5, 1.0, 0.5, 2, 3.5, 3.0, 1.5,
    1, 0.7, 0.5, 0.3, 3, 4.0, 2.5, 1.0,
    0.8, 2.0, 1, 2, 1, 1, 0,
    3.8, 2.0, 0, 0, 0
),(
    'R', 7, '2025-04-16', 5, TRUE, FALSE, 'AREDS2',
    '-1.5', 'Classic', 250, 150, 200, 180, 190, 8.5, 3.2,
    1, 1, 2, 0, 2.5, 1.5, 0.5,
    0, 1.2, 0.8, 0.3, 1, 2.8, 2.4, 1.0,
    0, 0.5, 0.3, 0.2, 2, 3.1, 1.8, 0.7,
    0.6, 1.5, 1, 2, 1, 1, 0,
    3.2, 1.5, 0, 0, 0
),(
    'L', 8, '2025-04-15', 10, FALSE, TRUE, 'AREDS1',
    '-2.0', 'Occult', 300, 160, 210, 190, 200, 9.0, 3.5,
    0, 2, 1, 1, 3.0, 2.0, 1.0,
    1, 1.5, 1.0, 0.5, 2, 3.5, 3.0, 1.5,
    1, 0.7, 0.5, 0.3, 3, 4.0, 2.5, 1.0,
    0.8, 2.0, 1, 2, 1, 1, 0,
    3.8, 2.0, 0, 0, 0
),(
    'R', 1, '2025-04-16', 5, TRUE, FALSE, 'AREDS2',
    '-1.5', 'Classic', 250, 150, 200, 180, 190, 8.5, 3.2,
    1, 1, 2, 0, 2.5, 1.5, 0.5,
    0, 1.2, 0.8, 0.3, 1, 2.8, 2.4, 1.0,
    0, 0.5, 0.3, 0.2, 2, 3.1, 1.8, 0.7,
    0.6, 1.5, 1, 2, 1, 1, 0,
    3.2, 1.5, 0, 0, 0
),(
    'L', 9, '2025-04-15', 10, FALSE, TRUE, 'AREDS1',
    '-2.0', 'Occult', 300, 160, 210, 190, 200, 9.0, 3.5,
    0, 2, 1, 1, 3.0, 2.0, 1.0,
    1, 1.5, 1.0, 0.5, 2, 3.5, 3.0, 1.5,
    1, 0.7, 0.5, 0.3, 3, 4.0, 2.5, 1.0,
    0.8, 2.0, 1, 2, 1, 1, 0,
    3.8, 2.0, 0, 0, 0
),(
    'R', 9, '2025-04-16', 5, TRUE, FALSE, 'AREDS2',
    '-1.5', 'Classic', 250, 150, 200, 180, 190, 8.5, 3.2,
    1, 1, 2, 0, 2.5, 1.5, 0.5,
    0, 1.2, 0.8, 0.3, 1, 2.8, 2.4, 1.0,
    0, 0.5, 0.3, 0.2, 2, 3.1, 1.8, 0.7,
    0.6, 1.5, 1, 2, 1, 1, 0,
    3.2, 1.5, 0, 0, 0
),(
    'L', 10, '2025-04-15', 10, FALSE, TRUE, 'AREDS1',
    '-2.0', 'Occult', 300, 160, 210, 190, 200, 9.0, 3.5,
    0, 2, 1, 1, 3.0, 2.0, 1.0,
    1, 1.5, 1.0, 0.5, 2, 3.5, 3.0, 1.5,
    1, 0.7, 0.5, 0.3, 3, 4.0, 2.5, 1.0,
    0.8, 2.0, 1, 2, 1, 1, 0,
    3.8, 2.0, 0, 0, 0
),(
    'R', 10, '2025-04-16', 5, TRUE, FALSE, 'AREDS2',
    '-1.5', 'Classic', 250, 150, 200, 180, 190, 8.5, 3.2,
    1, 1, 2, 0, 2.5, 1.5, 0.5,
    0, 1.2, 0.8, 0.3, 1, 2.8, 2.4, 1.0,
    0, 0.5, 0.3, 0.2, 2, 3.1, 1.8, 0.7,
    0.6, 1.5, 1, 2, 1, 1, 0,
    3.2, 1.5, 0, 0, 0
),(
    'L', 11, '2025-04-15', 10, FALSE, TRUE, 'AREDS1',
    '-2.0', 'Occult', 300, 160, 210, 190, 200, 9.0, 3.5,
    0, 2, 1, 1, 3.0, 2.0, 1.0,
    1, 1.5, 1.0, 0.5, 2, 3.5, 3.0, 1.5,
    1, 0.7, 0.5, 0.3, 3, 4.0, 2.5, 1.0,
    0.8, 2.0, 1, 2, 1, 1, 0,
    3.8, 2.0, 0, 0, 0
),(
    'R', 11, '2025-04-16', 5, TRUE, FALSE, 'AREDS2',
    '-1.5', 'Classic', 250, 150, 200, 180, 190, 8.5, 3.2,
    1, 1, 2, 0, 2.5, 1.5, 0.5,
    0, 1.2, 0.8, 0.3, 1, 2.8, 2.4, 1.0,
    0, 0.5, 0.3, 0.2, 2, 3.1, 1.8, 0.7,
    0.6, 1.5, 1, 2, 1, 1, 0,
    3.2, 1.5, 0, 0, 0
),(
    'L', 12, '2025-04-15', 10, FALSE, TRUE, 'AREDS1',
    '-2.0', 'Occult', 300, 160, 210, 190, 200, 9.0, 3.5,
    0, 2, 1, 1, 3.0, 2.0, 1.0,
    1, 1.5, 1.0, 0.5, 2, 3.5, 3.0, 1.5,
    1, 0.7, 0.5, 0.3, 3, 4.0, 2.5, 1.0,
    0.8, 2.0, 1, 2, 1, 1, 0,
    3.8, 2.0, 0, 0, 0
),(
    'R', 12, '2025-04-16', 5, TRUE, FALSE, 'AREDS2',
    '-1.5', 'Classic', 250, 150, 200, 180, 190, 8.5, 3.2,
    1, 1, 2, 0, 2.5, 1.5, 0.5,
    0, 1.2, 0.8, 0.3, 1, 2.8, 2.4, 1.0,
    0, 0.5, 0.3, 0.2, 2, 3.1, 1.8, 0.7,
    0.6, 1.5, 1, 2, 1, 1, 0,
    3.2, 1.5, 0, 0, 0
),(
    'L', 13, '2025-04-15', 10, FALSE, TRUE, 'AREDS1',
    '-2.0', 'Occult', 300, 160, 210, 190, 200, 9.0, 3.5,
    0, 2, 1, 1, 3.0, 2.0, 1.0,
    1, 1.5, 1.0, 0.5, 2, 3.5, 3.0, 1.5,
    1, 0.7, 0.5, 0.3, 3, 4.0, 2.5, 1.0,
    0.8, 2.0, 1, 2, 1, 1, 0,
    3.8, 2.0, 0, 0, 0
),(
    'R', 13, '2025-04-16', 5, TRUE, FALSE, 'AREDS2',
    '-1.5', 'Classic', 250, 150, 200, 180, 190, 8.5, 3.2,
    1, 1, 2, 0, 2.5, 1.5, 0.5,
    0, 1.2, 0.8, 0.3, 1, 2.8, 2.4, 1.0,
    0, 0.5, 0.3, 0.2, 2, 3.1, 1.8, 0.7,
    0.6, 1.5, 1, 2, 1, 1, 0,
    3.2, 1.5, 0, 0, 0
),(
    'L', 14, '2025-04-15', 10, FALSE, TRUE, 'AREDS1',
    '-2.0', 'Occult', 300, 160, 210, 190, 200, 9.0, 3.5,
    0, 2, 1, 1, 3.0, 2.0, 1.0,
    1, 1.5, 1.0, 0.5, 2, 3.5, 3.0, 1.5,
    1, 0.7, 0.5, 0.3, 3, 4.0, 2.5, 1.0,
    0.8, 2.0, 1, 2, 1, 1, 0,
    3.8, 2.0, 0, 0, 0
),(
    'R', 14, '2025-04-16', 5, TRUE, FALSE, 'AREDS2',
    '-1.5', 'Classic', 250, 150, 200, 180, 190, 8.5, 3.2,
    1, 1, 2, 0, 2.5, 1.5, 0.5,
    0, 1.2, 0.8, 0.3, 1, 2.8, 2.4, 1.0,
    0, 0.5, 0.3, 0.2, 2, 3.1, 1.8, 0.7,
    0.6, 1.5, 1, 2, 1, 1, 0,
    3.2, 1.5, 0, 0, 0
);"""


    _SQL_DIR_NAME = 'sql'
    _DATA_DIRS = (_SQL_DIR_NAME,)

    def __init__(self, db_name="database.db"):
        super().__init__()
        self.db = QSqlDatabase.addDatabase("QSQLITE")
        self.db_name_path = self.data_path(self._SQL_DIR_NAME) / db_name
        self.db.setDatabaseName(db_name)
        self.connection = sqlite3.connect(self.db_name_path)
        self.cursor = self.connection.cursor()
        self.execute_query(self.CREATE_PATIENTS)
        self.execute_query(self.CREATE_APPOINTMENTS)
        self.execute_query(self.CREATE_EYES)
        # self.execute_query(self.Pacients, )
        # self.execute_query(self.Appointments, )
        # self.execute_query(self.Appointmen_1_1)
        self.close_connection()

    def start_connection(self, db_name="database.db"):
        self.connection = sqlite3.connect(self.db_name_path)
        self.cursor = self.connection.cursor()

    def execute_query(self, query, params=()):
        try:
            self.cursor.execute(query, params)
            self.connection.commit()
        except sqlite3.Error as e:
            print(f"Ошибка выполнения запроса: {e}")

    def fetch_results(self, query, params=()):
        try:
            self.cursor.execute(query, params)
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Ошибка получения данных: {e}")
            return []

    def fetch_record_by_id(self, table, record_id):
        self.start_connection()
        query = "SELECT * FROM pacients WHERE id = :record_id"
        params = (record_id)
        self.cursor.execute(query, {"record_id": record_id})
        return self.cursor.fetchall()

    def update_pacient(self, name, sex, year_of_birthday, id):
        self.start_connection()
        query = "UPDATE pacients SET name = :name, sex = :sex, year_of_birthday = :year_of_birthday WHERE id = :id"
        self.cursor.execute(query, {"name": name, "sex": sex, "year_of_birthday": year_of_birthday, "id": id,})
        self.connection.commit()
        self.close_connection()

    def close_connection(self):
        self.connection.close()
