"""Диалог добавления приёма (дата, длительность, глаз)."""

from __future__ import annotations

from PySide6.QtCore import QDate, Signal
from PySide6.QtSql import QSqlQuery
from PySide6.QtWidgets import (
    QDateEdit,
    QDialog,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)

from bsmu.macula.plugins.db.debug_utils import logger

APPOINTMENTS_TABLE = "appointments"
#: Варианты выбора глаза.
EYE_RIGHT = "R"
EYE_LEFT = "L"


class AddAppoitmentRecordDialog(QDialog):
    """Создаёт запись приёма и сообщает её id вместе с выбранными глазами."""

    dataReady = Signal(dict)

    def __init__(self, db, patient_id: int):
        super().__init__()
        self.db = db
        self.pat_id = patient_id
        self.last_id = None
        self.eyes = [EYE_LEFT, EYE_RIGHT]

        self.setWindowTitle("Добавить приём")

        self.duration_input = QLineEdit()
        self.save_button = QPushButton("Сохранить")
        self.save_button.clicked.connect(self.save_data)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Дата приёма:"))
        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDate(QDate.currentDate())
        layout.addWidget(self.date_edit)
        layout.addWidget(QLabel("Длительность заболевания:"))
        layout.addWidget(self.duration_input)

        self.radio_one = QRadioButton("Записать данные по одному глазу")
        self.radio_two = QRadioButton("Записать данные по обоим глазам")
        self.radio_two.setChecked(True)
        layout.addWidget(self.radio_one)
        layout.addWidget(self.radio_two)

        self.eye_choice_group = QGroupBox("Выберите глаз")
        eye_layout = QVBoxLayout()
        self.radio_right = QRadioButton("Правый глаз")
        self.radio_left = QRadioButton("Левый глаз")
        eye_layout.addWidget(self.radio_right)
        eye_layout.addWidget(self.radio_left)
        self.eye_choice_group.setLayout(eye_layout)
        layout.addWidget(self.eye_choice_group)

        self.radio_one.toggled.connect(self.update_visibility)
        self.update_visibility()
        layout.addWidget(self.save_button)
        self.setLayout(layout)

    def update_visibility(self) -> None:
        """Уточнение по глазу нужно только для режима «один глаз»."""
        self.eye_choice_group.setVisible(self.radio_one.isChecked())

    def save_data(self) -> None:
        duration = self.duration_input.text().strip()
        if not duration:
            logger.warning("Не указана длительность заболевания — приём не создан")
            return

        if self.radio_one.isChecked():
            if self.radio_right.isChecked():
                self.eyes = [EYE_RIGHT]
            elif self.radio_left.isChecked():
                self.eyes = [EYE_LEFT]

        query = QSqlQuery(self.db)
        query.prepare(
            f"INSERT INTO {APPOINTMENTS_TABLE} "
            "(date, duration_of_the_disease, pacient_id) VALUES (?, ?, ?)"
        )
        query.addBindValue(self.date_edit.text())
        query.addBindValue(duration)
        query.addBindValue(self.pat_id)
        if not query.exec():
            logger.error("Не удалось создать приём: %s", query.lastError().text())
            return

        self.last_id = query.lastInsertId()
        logger.info("Создан приём id=%s для пациента %s", self.last_id, self.pat_id)
        self.close()

    def closeEvent(self, event) -> None:
        # Сигнал отдаём только при успешном создании приёма: иначе в окно
        # прилетал None и создавал вкладки без приёма.
        if self.last_id is not None:
            self.dataReady.emit({"e": self.eyes, "ap_id": self.last_id})
        super().closeEvent(event)
