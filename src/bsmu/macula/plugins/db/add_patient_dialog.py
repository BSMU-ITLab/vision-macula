"""Диалог добавления и редактирования пациента."""

from __future__ import annotations

from functools import partial

from PySide6.QtSql import QSqlQuery
from PySide6.QtWidgets import QDialog, QLabel, QLineEdit, QPushButton, QVBoxLayout

from bsmu.macula.plugins.db.debug_utils import logger

#: Таблица и имена полей (совпадают со схемой БД).
PACIENTS_TABLE = "pacients"


class AddRecordDialog(QDialog):
    """Добавляет нового пациента либо правит существующего."""

    def __init__(self, db, patient_id: int):
        super().__init__()
        self.db = db
        self.is_new_user = patient_id == 0
        self.patient_id = patient_id

        self.setWindowTitle(
            "Добавить пациента" if self.is_new_user else "Изменить данные пациента"
        )

        self.name_input = QLineEdit()
        self.sex_input = QLineEdit()
        self.age_input = QLineEdit()
        self.save_button = QPushButton("Сохранить" if self.is_new_user else "Редактировать")
        self.save_button.clicked.connect(
            self.save_data if self.is_new_user else partial(self.edit_data, patient_id)
        )

        layout = QVBoxLayout()
        for label, widget in (
            ("Имя:", self.name_input),
            ("Пол:", self.sex_input),
            ("Год рождения:", self.age_input),
        ):
            layout.addWidget(QLabel(label))
            layout.addWidget(widget)
        layout.addWidget(self.save_button)
        self.setLayout(layout)

        if not self.is_new_user:
            self._load_patient(patient_id)

    # --- данные ---

    def _load_patient(self, patient_id: int) -> None:
        record = self._fetch_patient(patient_id)
        if record is None:
            return
        self.name_input.setText(record["name"])
        self.sex_input.setText(record["sex"])
        self.age_input.setText(str(record["year_of_birthday"]))

    def _fetch_patient(self, patient_id: int) -> dict | None:
        query = QSqlQuery(self.db)
        query.prepare(
            f"SELECT name, sex, year_of_birthday FROM {PACIENTS_TABLE} WHERE id = :id"
        )
        query.bindValue(":id", patient_id)
        if not query.exec() or not query.next():
            logger.error("Пациент id=%s не найден: %s", patient_id, query.lastError().text())
            return None
        return {
            "name": query.value(0),
            "sex": query.value(1),
            "year_of_birthday": query.value(2),
        }

    def _form_values(self) -> tuple[str, str, int] | None:
        name = self.name_input.text().strip()
        sex = self.sex_input.text().strip()
        age = self.age_input.text().strip()
        if not name or not age.isdigit():
            logger.warning("Некорректные данные пациента: имя=%r, год=%r", name, age)
            return None
        return name, sex, int(age)

    # --- действия ---

    def save_data(self) -> None:
        values = self._form_values()
        if values is None:
            return

        query = QSqlQuery(self.db)
        query.prepare(
            f"INSERT INTO {PACIENTS_TABLE} (name, sex, year_of_birthday) VALUES (?, ?, ?)"
        )
        for value in values:
            query.addBindValue(value)
        if not query.exec():
            logger.error("Не удалось добавить пациента: %s", query.lastError().text())
            return

        logger.info("Добавлен пациент id=%s", query.lastInsertId())
        self.close()

    def edit_data(self, patient_id: int) -> None:
        values = self._form_values()
        if values is None:
            return

        query = QSqlQuery(self.db)
        query.prepare(
            f"UPDATE {PACIENTS_TABLE} SET name = ?, sex = ?, year_of_birthday = ? WHERE id = ?"
        )
        for value in (*values, patient_id):
            query.addBindValue(value)
        if not query.exec():
            logger.error("Не удалось изменить пациента: %s", query.lastError().text())
            return

        logger.info("Изменён пациент id=%s", patient_id)
        self.close()
