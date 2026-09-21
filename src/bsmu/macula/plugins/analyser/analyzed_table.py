"""Окно результатов измерений: основная таблица и таблицы деталей.

Модуль отвечает только за представление данных. Сборка объекта
``PatientExamData`` вынесена в
:class:`bsmu.macula.plugins.analyser.data_converter.PatientExamDataBuilder`,
а описания отслоек и редактируемых полей — в
:mod:`bsmu.macula.plugins.analyser.schema`.
"""

from __future__ import annotations

import re
from typing import Callable

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QMdiSubWindow,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from bsmu.macula.plugins.analyser.data_converter import PatientExamDataBuilder
from bsmu.macula.plugins.analyser.schema import (
    EDITABLE_PARAMETERS,
    ALL_DETACHMENTS,
    DetachmentSpec,
    RowKey,
    drusen_number,
    is_drusen_parameter,
)

#: Заголовки колонок таблиц.
TABLE_HEADERS = ["Параметр", "Значение"]
#: Суффиксы измерений отслойки, отбрасываемые в заголовках деталей.
_DETACHMENT_SECTION_RE = re.compile(
    r"^(" + "|".join(re.escape(spec.section) for spec in ALL_DETACHMENTS) + r") \("
)
_DRUSEN_PREFIX_RE = re.compile(r"^(.*?)\s*#\d+\s*\(")


def group_measurements_data(data: list[dict]) -> list[dict]:
    """Оставляет строки основной таблицы.

    Друзы и отслойки показываются в отдельных таблицах справа, поэтому из
    основной таблицы они исключаются.
    """
    return [
        item
        for item in data
        if "#" not in item.get(RowKey.PARAMETER, "")
        and not is_detachment_measurement_id(item.get(RowKey.MEASUREMENT_ID, ""))
    ]


def is_detachment_measurement_id(measurement_id: str) -> bool:
    return any(measurement_id.startswith(spec.measurement_id + "_") for spec in ALL_DETACHMENTS)


def group_drusen_measurements(data: list[dict]) -> dict[int, list[dict]]:
    """Группирует строки измерений по номеру друзы."""
    groups: dict[int, list[dict]] = {}
    for item in data:
        if item.get(RowKey.IS_GROUP):
            continue
        number = drusen_number(item.get(RowKey.PARAMETER, ""))
        if number is not None and is_drusen_parameter(item.get(RowKey.PARAMETER, "")):
            groups.setdefault(number, []).append(item)
    return groups


def group_detachment_measurements(data: list[dict]) -> dict[str, list[dict]]:
    """Группирует строки измерений по типу отслойки."""
    groups: dict[str, list[dict]] = {}
    for item in data:
        measurement_id = item.get(RowKey.MEASUREMENT_ID, "")
        for spec in ALL_DETACHMENTS:
            if measurement_id.startswith(spec.measurement_id + "_"):
                groups.setdefault(spec.measurement_id, []).append(item)
                break
    return groups


class ObjectsTableModel(QAbstractTableModel):
    """Модель основной таблицы с раскрываемыми группами и ручными полями."""

    HEADERS = TABLE_HEADERS
    EDITABLE_PARAMETERS = EDITABLE_PARAMETERS

    def __init__(self, data: list[dict], patient_exam_data=None):
        super().__init__()
        self._data = group_measurements_data(data)
        self._patient_exam_data = patient_exam_data
        self._visible_rows = self._build_visible_rows()

    def _build_visible_rows(self) -> list[dict]:
        """Список видимых строк с учётом раскрытости групп."""
        visible: list[dict] = []
        for item in self._data:
            visible.append(item)
            if item.get(RowKey.IS_GROUP) and item.get(RowKey.EXPANDED):
                visible.extend(item.get(RowKey.CHILDREN, []))
        return visible

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self._visible_rows)

    def columnCount(self, parent=QModelIndex()) -> int:
        return len(self.HEADERS)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role not in (Qt.DisplayRole, Qt.EditRole):
            return None

        row = self._visible_rows[index.row()]
        parameter = row.get(RowKey.PARAMETER, "")
        if role == Qt.DisplayRole and row.get(RowKey.IS_GROUP):
            arrow = "▼" if row.get(RowKey.EXPANDED) else "▶"
            parameter = f"{arrow} {parameter}"

        value = f"{row.get(RowKey.VALUE, '')} {row.get(RowKey.UNIT, '')}".strip()
        return [parameter, value][index.column()]

    def flags(self, index):
        if not index.isValid():
            return Qt.NoItemFlags

        flags = super().flags(index)
        if index.column() == 1:
            row = self._visible_rows[index.row()]
            if row.get(RowKey.PARAMETER) in self.EDITABLE_PARAMETERS:
                return flags | Qt.ItemIsEditable
        return flags

    def setData(self, index, value, role=Qt.EditRole) -> bool:
        """Обновляет данные при редактировании и синхронизирует PatientExamData."""
        if role != Qt.EditRole or not index.isValid() or index.column() != 1:
            return False

        row = self._visible_rows[index.row()]
        row[RowKey.VALUE] = str(value)
        self._sync_manual_field(row.get(RowKey.PARAMETER, ""), str(value))
        self.dataChanged.emit(index, index)
        return True

    def _sync_manual_field(self, parameter: str, value: str) -> None:
        """Переносит правку ручного поля в объект ``PatientExamData``."""
        if self._patient_exam_data is None or not value:
            return

        field_by_parameter = {
            "Стадия по AREDS": "areds_criteria",
            "Тип неоваскуляризации": "neovascularization_type",
            "Рефракция": "refraction",
            "МКОЗ": "bcva",
            "Длительность заболевания": "disease_duration",
            "Локализация кистозного макулярного отека": "cmo_location",
        }
        attribute = field_by_parameter.get(parameter)
        if attribute is not None:
            setattr(self._patient_exam_data, attribute, value)

    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.HEADERS[section]

    def toggle_group(self, index) -> None:
        """Переключает состояние раскрытости группы."""
        if not index.isValid():
            return

        row = self._visible_rows[index.row()]
        if row.get(RowKey.IS_GROUP):
            row[RowKey.EXPANDED] = not row.get(RowKey.EXPANDED, False)
            self._visible_rows = self._build_visible_rows()
            self.layoutChanged.emit()

    def get_row_data(self, index) -> dict | None:
        if not index.isValid():
            return None
        return self._visible_rows[index.row()]

    def get_patient_exam_data(self):
        return self._patient_exam_data


class DetachmentDetailModel(QAbstractTableModel):
    """Модель таблицы деталей друзы или отслойки."""

    HEADERS = TABLE_HEADERS

    def __init__(self, measurements: list[dict]):
        super().__init__()
        self._data = measurements

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self._data)

    def columnCount(self, parent=QModelIndex()) -> int:
        return len(self.HEADERS)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role != Qt.DisplayRole:
            return None

        row = self._data[index.row()]
        parameter = short_parameter_name(row.get(RowKey.PARAMETER, ""))
        value = f"{row.get(RowKey.VALUE, '')} {row.get(RowKey.UNIT, '')}".strip()
        return [parameter, value][index.column()]

    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.HEADERS[section]


def short_parameter_name(parameter: str) -> str:
    """Убирает из названия номер друзы и заголовок отслойки для компактности."""
    if _DRUSEN_PREFIX_RE.match(parameter):
        parameter = _DRUSEN_PREFIX_RE.sub("", parameter)
    parameter = _DETACHMENT_SECTION_RE.sub("", parameter)
    return parameter.rstrip(") ").strip()


class TableWindow(QWidget):
    """Окно результатов: основная таблица слева, детали справа."""

    def __init__(self, measurements: list[dict], patient_exam_data=None,
                 highlight_callback: Callable | None = None):
        super().__init__()

        self.setWindowTitle("Результаты измерений")
        # Жёсткий размер не задаём: виджет живёт внутри QMdiSubWindow и должен
        # уметь ужиматься, чтобы помещаться рядом со снимком даже на ноутбуке.

        self._measurements = measurements
        self._highlight_callback = highlight_callback
        self._exam_data_builder = PatientExamDataBuilder()
        self._drusen_groups = group_drusen_measurements(measurements)
        self._detachment_groups = group_detachment_measurements(measurements)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)
        main_layout.addWidget(self._build_splitter(patient_exam_data))
        self.setLayout(main_layout)

        self._update_drusen_details(None)

    # --- построение интерфейса ---

    def _build_splitter(self, patient_exam_data) -> QSplitter:
        """Разделитель: слева таблица измерений, справа детали.

        Правую панель можно перетащить или схлопнуть совсем, а её содержимое
        прокручивается внутри — блоки отслоек больше не растягивают окно.
        """
        left_widget = QWidget()
        left_layout = self._build_left_part(patient_exam_data)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)
        left_widget.setLayout(left_layout)

        right_widget = QWidget()
        right_layout = self._build_right_part()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)
        # Прижимаем блоки к верху, чтобы они не растягивались на всю высоту.
        right_layout.addStretch()
        right_widget.setLayout(right_layout)

        right_scroll = QScrollArea()
        right_scroll.setWidget(right_widget)
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QScrollArea.NoFrame)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_scroll)
        splitter.setStretchFactor(0, 2)  # 2/3 ширины
        splitter.setStretchFactor(1, 1)  # 1/3 ширины
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, True)
        return splitter

    def _build_left_part(self, patient_exam_data) -> QVBoxLayout:
        layout = QVBoxLayout()

        self.table = QTableView()
        self.model = ObjectsTableModel(self._measurements, patient_exam_data)
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.resizeColumnsToContents()
        self._enable_hover(self.table, self._on_entered_index)
        self.table.clicked.connect(self._on_table_clicked)
        layout.addWidget(self.table)

        save_button = QPushButton("Сохранить в БД")
        save_button.clicked.connect(self._save_to_database)
        layout.addWidget(save_button)
        return layout

    def _build_right_part(self) -> QVBoxLayout:
        layout = QVBoxLayout()
        layout.addWidget(self._build_drusen_group())
        for spec in ALL_DETACHMENTS:
            items = self._detachment_groups.get(spec.measurement_id)
            if not items:
                continue
            layout.addWidget(self._build_detachment_group(spec, items))
        return layout

    def _build_drusen_group(self) -> QGroupBox:
        group = QGroupBox("Детали друзы")
        layout = QVBoxLayout()

        selector = QHBoxLayout()
        selector.addWidget(QLabel("Выберите друзу:"))
        self.drusen_combo = QComboBox()
        self.drusen_combo.addItem("(не выбрано)", None)
        for number in sorted(self._drusen_groups):
            self.drusen_combo.addItem(f"Друза #{number}", number)
        self.drusen_combo.currentIndexChanged.connect(self._on_drusen_selected)
        selector.addWidget(self.drusen_combo)
        selector.addStretch()
        layout.addLayout(selector)

        self.drusen_detail_table = QTableView()
        self.drusen_detail_model = DetachmentDetailModel([])
        self.drusen_detail_table.setModel(self.drusen_detail_model)
        self._enable_hover(
            self.drusen_detail_table,
            self._on_drusen_table_hover,
        )
        layout.addWidget(self.drusen_detail_table)

        group.setLayout(layout)
        return group

    def _build_detachment_group(self, spec: DetachmentSpec, items: list[dict]) -> QGroupBox:
        group = QGroupBox(spec.table_item)
        layout = QVBoxLayout()

        table = QTableView()
        model = DetachmentDetailModel(items)
        table.setModel(model)
        table.resizeColumnsToContents()
        self._enable_hover(table, lambda index, rows=items: self._on_detachment_table_hover(rows, index))
        layout.addWidget(table)

        group.setLayout(layout)
        self.detachment_tables = getattr(self, "detachment_tables", {})
        self.detachment_models = getattr(self, "detachment_models", {})
        self.detachment_tables[spec.measurement_id] = table
        self.detachment_models[spec.measurement_id] = model
        return group

    def _enable_hover(self, view: QTableView, handler: Callable) -> None:
        """Включает подсветку объектов при наведении курсора на строку."""
        view.setMouseTracking(True)
        if self._highlight_callback is None:
            return
        view.entered.connect(handler)
        try:
            view.viewportEntered.connect(self._on_viewport_entered)
        except Exception:  # noqa: BLE001 — сигнала может не быть в некоторых сборках Qt
            pass

    # --- обработчики ---

    def _on_drusen_selected(self, index: int) -> None:
        number = self.drusen_combo.itemData(index)
        self._update_drusen_details(number)

        if self._highlight_callback is None:
            return
        if number is None or number not in self._drusen_groups:
            self._highlight_callback(None)
            return
        # Все измерения друзы содержат один и тот же контур.
        self._highlight_callback(self._drusen_groups[number][0])

    def _on_detachment_table_hover(self, items: list[dict], index) -> None:
        if not index.isValid() or not items or self._highlight_callback is None:
            return
        self._highlight_callback(items[0])

    def _on_drusen_table_hover(self, index) -> None:
        if not index.isValid() or self._highlight_callback is None:
            return
        number = self.drusen_combo.currentData()
        if number is not None and number in self._drusen_groups:
            self._highlight_callback(self._drusen_groups[number][0])

    def _update_drusen_details(self, number) -> None:
        items = self._drusen_groups.get(number, []) if number is not None else []
        self.drusen_detail_model = DetachmentDetailModel(items)
        self.drusen_detail_table.setModel(self.drusen_detail_model)
        self.drusen_detail_table.resizeColumnsToContents()

    def _on_table_clicked(self, index) -> None:
        row_data = self.model.get_row_data(index)
        if row_data and row_data.get(RowKey.IS_GROUP):
            self.model.toggle_group(index)

    def _on_entered_index(self, index) -> None:
        if not index.isValid() or self._highlight_callback is None:
            return
        row_data = self.model.get_row_data(index)
        if row_data:
            self._highlight_callback(row_data)

    def _on_viewport_entered(self) -> None:
        if self._highlight_callback is not None:
            self._highlight_callback(None)

    # --- сохранение ---

    def _save_to_database(self) -> None:
        """Собирает ``PatientExamData`` и показывает результат пользователю."""
        try:
            patient_exam_data = self._exam_data_builder.build(self._measurements)
            QMessageBox.information(
                self,
                "Данные подготовлены",
                f"PatientExamData успешно заполнен:\n"
                f"Дата визита: {patient_exam_data.visit_date}\n"
                f"Длительность заболевания: {patient_exam_data.disease_duration}\n"
                f"МКОЗ: {patient_exam_data.bcva}\n"
                f"Объем сетчатки: {patient_exam_data.total_retinal_volume}\n"
                f"Толщина хориоидеи: {patient_exam_data.choroidal_center_thickness}\n"
                f"ЦТС фовеола: {patient_exam_data.cts_near_foveolla}\n"
                f"ЦТС фовеа: {patient_exam_data.cts_near_fovea}\n"
                f"\nДрузы (средние): width={patient_exam_data.drusen.width}, "
                f"height={patient_exam_data.drusen.height}, area={patient_exam_data.drusen.area}",
            )
        except Exception as exc:  # noqa: BLE001 — показываем пользователю любую ошибку
            QMessageBox.critical(
                self, "Ошибка", f"Ошибка при заполнении PatientExamData:\n{exc}"
            )

    def _fill_patient_exam_data(self):
        """Совместимый метод: собрать ``PatientExamData`` из строк таблицы."""
        return self._exam_data_builder.build(self._measurements)


class MeasurementsSubWindow(QMdiSubWindow):
    """Окно результатов измерений внутри ``QMdiArea``.

    Раньше ``TableWindow`` показывался как отдельное top-level окно и
    перекрывался главным окном программы при его активации. Теперь это обычное
    MDI-подокно: его можно тайлить рядом со снимком и видеть картинку и
    измерения одновременно.
    """

    def __init__(self, table_window: TableWindow):
        super().__init__()

        self.setWidget(table_window)
        self.setWindowTitle(table_window.windowTitle())
        self.setAttribute(Qt.WA_DeleteOnClose)


__all__ = [
    "DetachmentDetailModel",
    "MeasurementsSubWindow",
    "ObjectsTableModel",
    "TableWindow",
    "group_detachment_measurements",
    "group_drusen_measurements",
    "group_measurements_data",
    "is_detachment_measurement_id",
    "short_parameter_name",
]
