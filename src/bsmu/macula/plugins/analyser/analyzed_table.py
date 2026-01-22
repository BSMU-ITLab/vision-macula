from PySide6.QtWidgets import (
    QApplication, QWidget, QTableView,
    QVBoxLayout, QHBoxLayout, QAbstractItemView, QComboBox, QLabel, QGroupBox, QPushButton, QMessageBox
)
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from typing import Callable, Optional
from datetime import datetime
import sys

from bsmu.macula.records.eye_info_data import PatientExamData, Measurement, ZoneStatus


# ====== HELPERS ======
def group_measurements_data(data: list[dict]) -> list[dict]:
    """Фильтрует данные, исключая друзы и отслойки (они отображаются в отдельных таблицах)"""
    result = []
    
    # Список measurement_id для отслоек
    detachment_ids = [
        "serous_ped_", "hemorrhagic_ped_", "fibrovascular_ped_", 
        "drusenoid_ped_", "neuroepithelial_detachment_"
    ]
    
    for item in data:
        param = item.get("parameter", "")
        measurement_id = item.get("measurement_id", "")
        
        # Исключаем ВСЁ что содержит "#" в названии параметра
        has_number_sign = "#" in param
        
        # Проверяем, является ли это отслойкой
        is_detachment = any(measurement_id.startswith(det_id) for det_id in detachment_ids)
        
        # Пропускаем элементы с # и отслойки - они отображаются в отдельных таблицах справа
        if not has_number_sign and not is_detachment:
            result.append(item)
    
    return result


# ====== DATA MODEL ======
class ObjectsTableModel(QAbstractTableModel):
    HEADERS = [
        "Параметр",
        "Значение",
        "Единица",
    ]
    
    # Параметры которые должны быть редактируемыми
    EDITABLE_PARAMETERS = {
        "Дата визита",
        "Длительность заболевания",
        "Стадия по AREDS",
        "Тип неоваскуляризации",
        "Рефракция",
        "МКОЗ",
        "Объем сетчатки",
        "Препарат",
        "Количество назначенных инъекций",
        "Количество выполненных инъекций",
        "Локализация кистозного макулярного отека",
    }

    def __init__(self, data: list[dict], patient_exam_data: PatientExamData = None):
        super().__init__()
        self._data = group_measurements_data(data)
        self._patient_exam_data = patient_exam_data or PatientExamData()
        self._visible_rows = self._build_visible_rows()

    def _build_visible_rows(self):
        """Строит список видимых строк на основе раскрытости групп"""
        visible = []
        for item in self._data:
            visible.append(item)
            if item.get("is_group") and item.get("expanded"):
                for child in item.get("children", []):
                    visible.append(child)
        return visible

    def rowCount(self, parent=QModelIndex()):
        return len(self._visible_rows)

    def columnCount(self, parent=QModelIndex()):
        return len(self.HEADERS)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None

        row = self._visible_rows[index.row()]

        if role == Qt.DisplayRole or role == Qt.EditRole:
            param = row.get("parameter", "")
            # Добавляем стрелку для групп
            if role == Qt.DisplayRole and row.get("is_group"):
                arrow = "▼" if row.get("expanded") else "▶"
                param = f"{arrow} {param}"
            
            return [
                param,
                row.get("value", ""),
                row.get("unit", ""),
            ][index.column()]

        return None

    def flags(self, index):
        """Определяет флаги для ячейки (редактируемость)"""
        if not index.isValid():
            return Qt.NoItemFlags
        
        flags = super().flags(index)
        row = self._visible_rows[index.row()]
        
        # Делаем редактируемыми только значения для определённых параметров
        if index.column() == 1:  # Колонка "Значение"
            param = row.get("parameter", "")
            if param in self.EDITABLE_PARAMETERS:
                return flags | Qt.ItemIsEditable
        
        return flags

    def setData(self, index, value, role=Qt.EditRole):
        """Обновляет данные при редактировании и синхронизирует с PatientExamData"""
        if role == Qt.EditRole and index.isValid():
            row = self._visible_rows[index.row()]
            if index.column() == 1:  # Колонка "Значение"
                row["value"] = str(value)
                
                # Синхронизируем с PatientExamData
                param = row.get("parameter", "")
                if param == "Стадия по AREDS":
                    self._patient_exam_data.areds_criteria = str(value) if value else None
                elif param == "Тип неоваскуляризации":
                    self._patient_exam_data.neovascularization_type = str(value) if value else None
                elif param == "Рефракция":
                    self._patient_exam_data.refraction = str(value) if value else None
                elif param == "МКОЗ":
                    self._patient_exam_data.bcva = str(value) if value else None
                
                self.dataChanged.emit(index, index)
                return True
        return False

    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.HEADERS[section]

    def toggle_group(self, index):
        """Переключает состояние раскрытости группы"""
        if not index.isValid():
            return
        
        row = self._visible_rows[index.row()]
        if row.get("is_group"):
            row["expanded"] = not row.get("expanded", False)
            self._visible_rows = self._build_visible_rows()
            self.layoutChanged.emit()
    
    def get_row_data(self, index):
        """Получает данные строки"""
        if not index.isValid():
            return None
        return self._visible_rows[index.row()]
    
    def get_patient_exam_data(self) -> PatientExamData:
        """Возвращает объект PatientExamData с актуальными данными"""
        return self._patient_exam_data


# ====== GUI ======
class TableWindow(QWidget):
    def __init__(self, measurements: list[dict], patient_exam_data: PatientExamData = None, highlight_callback: Optional[Callable] = None):
        super().__init__()

        self.setWindowTitle("Результаты измерений")
        self.resize(1200, 700)

        # Извлекаем все друзы из measurements
        self._all_drusen = []
        for item in measurements:
            param = item.get("parameter", "")
            if "Drusen #" in param or ("друз" in param.lower() and "#" in param):
                # Проверяем, что это измерение друзы (не группа)
                if not item.get("is_group"):
                    self._all_drusen.append(item)
        
        # Группируем друзы по номеру
        self._drusen_groups = {}
        for drusen in self._all_drusen:
            param = drusen.get("parameter", "")
            # Извлекаем номер друзы (например, "Drusen #1 (ширина)" -> 1)
            import re
            match = re.search(r'#(\d+)', param)
            if match:
                drusen_num = int(match.group(1))
                if drusen_num not in self._drusen_groups:
                    self._drusen_groups[drusen_num] = []
                self._drusen_groups[drusen_num].append(drusen)
        
        # Извлекаем все отслойки из measurements
        self._detachment_types = {
            "serous_ped": {"name": "Серозная ОПЭ", "items": []},
            "hemorrhagic_ped": {"name": "Геморрагическая ОПЭ", "items": []},
            "fibrovascular_ped": {"name": "Фиброваскулярная ОПЭ", "items": []},
            "drusenoid_ped": {"name": "Друзеноидная ОПЭ", "items": []},
            "neuroepithelial_detachment": {"name": "Отслойка нейроэпителия", "items": []},
        }
        
        for item in measurements:
            measurement_id = item.get("measurement_id", "")
            for det_type, det_data in self._detachment_types.items():
                if measurement_id.startswith(det_type + "_"):
                    det_data["items"].append(item)
        
        # Основной layout - горизонтальный
        main_layout = QHBoxLayout(self)
        
        # Левая часть - основная таблица
        left_layout = QVBoxLayout()
        
        self.table = QTableView()
        self.model = ObjectsTableModel(measurements, patient_exam_data)
        self.table.setModel(self.model)

        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.horizontalHeader().setStretchLastSection(True)
        
        # Подгоняем ширину колонок
        self.table.resizeColumnsToContents()

        # Enable hover tracking and connect hover signals
        self.table.setMouseTracking(True)
        self._highlight_callback = highlight_callback
        if highlight_callback is not None:
            # emitted when cursor moves over an index
            self.table.entered.connect(self._on_entered_index)
            # emitted when cursor enters the viewport outside any index
            try:
                self.table.viewportEntered.connect(self._on_viewport_entered)
            except Exception:
                pass
        
        # Обработка клика для раскрытия групп
        self.table.clicked.connect(self._on_table_clicked)
        
        left_layout.addWidget(self.table)
        
        # Кнопка "Сохранить в БД"
        save_button = QPushButton("Сохранить в БД")
        save_button.clicked.connect(self._save_to_database)
        left_layout.addWidget(save_button)
        
        # Сохраняем ссылку на measurements для использования при сохранении
        self._measurements = measurements
        
        # Правая часть - детали друз и отслоек
        right_layout = QVBoxLayout()
        
        # --- Блок друз ---
        drusen_group = QGroupBox("Детали друзы")
        drusen_layout = QVBoxLayout()
        
        # Выпадающий список для выбора друзы
        drusen_selector_layout = QHBoxLayout()
        drusen_selector_layout.addWidget(QLabel("Выберите друзу:"))
        self.drusen_combo = QComboBox()
        self.drusen_combo.addItem("(не выбрано)", None)
        for drusen_num in sorted(self._drusen_groups.keys()):
            self.drusen_combo.addItem(f"Друза #{drusen_num}", drusen_num)
        self.drusen_combo.currentIndexChanged.connect(self._on_drusen_selected)
        drusen_selector_layout.addWidget(self.drusen_combo)
        drusen_selector_layout.addStretch()
        drusen_layout.addLayout(drusen_selector_layout)
        
        # Таблица с деталями выбранной друзы
        self.drusen_detail_table = QTableView()
        self.drusen_detail_model = DetachmentDetailModel([])
        self.drusen_detail_table.setModel(self.drusen_detail_model)
        
        # Подключаем визуализацию при наведении на таблицу друз
        self.drusen_detail_table.setMouseTracking(True)
        self.drusen_detail_table.entered.connect(self._on_drusen_table_hover)
        try:
            self.drusen_detail_table.viewportEntered.connect(self._on_viewport_entered)
        except Exception:
            pass
        
        drusen_layout.addWidget(self.drusen_detail_table)
        drusen_group.setLayout(drusen_layout)
        right_layout.addWidget(drusen_group)
        
        # --- Блоки отслоек ---
        self.detachment_tables = {}
        self.detachment_models = {}
        
        for det_type, det_data in self._detachment_types.items():
            if len(det_data["items"]) == 0:
                continue  # Пропускаем отслойки, которые не найдены
            
            det_group = QGroupBox(det_data["name"])
            det_layout = QVBoxLayout()
            
            # Таблица с деталями отслойки (без выпадающего списка)
            det_table = QTableView()
            det_model = DetachmentDetailModel(det_data["items"])
            det_table.setModel(det_model)
            det_table.resizeColumnsToContents()
            
            # Подключаем визуализацию при наведении на строки таблицы
            det_table.setMouseTracking(True)
            det_table.entered.connect(lambda idx, items=det_data["items"]: self._on_detachment_table_hover(items, idx))
            try:
                det_table.viewportEntered.connect(self._on_viewport_entered)
            except Exception:
                pass
            
            det_layout.addWidget(det_table)
            det_group.setLayout(det_layout)
            right_layout.addWidget(det_group)
            
            # Сохраняем ссылки
            self.detachment_tables[det_type] = det_table
            self.detachment_models[det_type] = det_model
        
        # Добавляем левую и правую части в main_layout
        main_layout.addLayout(left_layout, 2)  # 2/3 ширины
        main_layout.addLayout(right_layout, 1)  # 1/3 ширины
        
        self.setLayout(main_layout)
        
        # Инициализируем модель деталей друз
        self._update_drusen_details(None)

    def _on_drusen_selected(self, index):
        """Обработчик выбора друзы из выпадающего списка"""
        drusen_num = self.drusen_combo.itemData(index)
        self._update_drusen_details(drusen_num)
        
        # Визуализируем выбранную друзу
        if drusen_num is not None and drusen_num in self._drusen_groups and self._highlight_callback:
            # Берём первое измерение друзы для визуализации (они все содержат одинаковый контур)
            drusen_measurements = self._drusen_groups[drusen_num]
            if drusen_measurements:
                self._highlight_callback(drusen_measurements[0])
        elif self._highlight_callback:
            # Сбрасываем визуализацию, если ничего не выбрано
            self._highlight_callback(None)
    
    def _on_detachment_table_hover(self, items, index):
        """Обработчик наведения на строку таблицы отслойки"""
        if not index.isValid() or not items or not self._highlight_callback:
            return
        # Визуализируем отслойку при наведении
        self._highlight_callback(items[0])
    
    def _on_drusen_table_hover(self, index):
        """Обработчик наведения на строку таблицы деталей друзы"""
        if not index.isValid() or not self._highlight_callback:
            return
        
        # Получаем текущую выбранную друзу
        drusen_num = self.drusen_combo.currentData()
        if drusen_num is not None and drusen_num in self._drusen_groups:
            drusen_measurements = self._drusen_groups[drusen_num]
            if drusen_measurements:
                self._highlight_callback(drusen_measurements[0])
    
    def _update_drusen_details(self, drusen_num):
        """Обновляет таблицу деталей выбранной друзы"""
        if drusen_num is None or drusen_num not in self._drusen_groups:
            # Пустая таблица
            self.drusen_detail_model = DetachmentDetailModel([])
        else:
            # Получаем все измерения для выбранной друзы
            drusen_measurements = self._drusen_groups[drusen_num]
            self.drusen_detail_model = DetachmentDetailModel(drusen_measurements)
        
        self.drusen_detail_table.setModel(self.drusen_detail_model)
        self.drusen_detail_table.resizeColumnsToContents()

    def _on_table_clicked(self, index):
        """Обработчик клика на таблицу"""
        row_data = self.model.get_row_data(index)
        if row_data and row_data.get("is_group"):
            self.model.toggle_group(index)

    def _on_entered_index(self, index):
        if not index.isValid():
            return
        row_data = self.model.get_row_data(index)
        if row_data and self._highlight_callback:
            self._highlight_callback(row_data)

    def _on_viewport_entered(self):
        if self._highlight_callback:
            self._highlight_callback(None)

    def _save_to_database(self):
        """Сохранить данные в объект PatientExamData и показать результат"""
        try:
            # Получаем данные из таблицы
            patient_exam_data = self._fill_patient_exam_data()
            
            # TODO: Здесь должна быть логика сохранения в базу данных
            # Пока просто показываем сообщение с данными
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
                f"height={patient_exam_data.drusen.height}, area={patient_exam_data.drusen.area}"
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Ошибка",
                f"Ошибка при заполнении PatientExamData:\n{str(e)}"
            )

    def _fill_patient_exam_data(self) -> PatientExamData:
        """Заполняет объект PatientExamData из данных таблицы"""
        exam_data = PatientExamData()
        
        # Получаем значения из ручных полей
        manual_values = {}
        for item in self._measurements:
            param = item.get("parameter", "")
            value = item.get("value", "")
            if param in self.model.EDITABLE_PARAMETERS and value:
                manual_values[param] = value
        
        # Заполняем ручные поля
        if "Дата визита" in manual_values:
            try:
                # Пробуем разные форматы даты
                date_str = manual_values["Дата визита"]
                for fmt in ["%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y"]:
                    try:
                        exam_data.visit_date = datetime.strptime(date_str, fmt).date()
                        break
                    except ValueError:
                        continue
            except Exception:
                pass
        
        exam_data.disease_duration = manual_values.get("Длительность заболевания")
        exam_data.refraction = manual_values.get("Рефракция")
        exam_data.neovascularization_type = manual_values.get("Тип неоваскуляризации")
        exam_data.bcva = manual_values.get("МКОЗ")
        exam_data.cmo_location = manual_values.get("Локализация кистозного макулярного отека")
        
        if "Объем сетчатки" in manual_values:
            try:
                exam_data.total_retinal_volume = float(manual_values["Объем сетчатки"])
            except ValueError:
                pass
        
        # Заполняем измеренные значения
        for item in self._measurements:
            measurement_id = item.get("measurement_id", "")
            value_str = item.get("value", "")
            
            try:
                value = float(value_str) if value_str else None
            except ValueError:
                value = None
            
            # Толщина хориоидеи в центре
            if measurement_id == "choroid_thickness_center" and value:
                exam_data.choroidal_center_thickness = value
            
            # Центральная толщина сетчатки (фовеола)
            elif measurement_id == "cts_foveola" and value:
                exam_data.foveal_retinal_thickness = value
            
            # ЦТС возле фовеолы
            elif measurement_id == "cts_near_foveola" and value:
                exam_data.cts_near_foveolla = value
            
            # ЦТС возле фовеа
            elif measurement_id == "cts_near_fovea" and value:
                exam_data.cts_near_fovea = value
            
            # Состояние РПЭ
            elif item.get("parameter") == "Состояние РПЭ":
                exam_data.rpe_status.condition = value_str
                # Сохраняем координату дефекта если есть
                defect_coord = item.get("defect_coord")
                if defect_coord is not None:
                    exam_data.rpe_status.defect_location = f"{defect_coord[0]},{defect_coord[1]}"
            
            # Локализация дефектов РПЭ (текстовая) - сохраняем координату из defect_coord
            elif item.get("parameter") == "Локализация дефектов РПЭ":
                defect_coord = item.get("defect_coord")
                if defect_coord is not None:
                    # Сохраняем координату в формате "x,y"
                    exam_data.rpe_status.defect_location = f"{defect_coord[0]},{defect_coord[1]}"
            
            # Серозная ОПЭ
            elif measurement_id.startswith("serous_ped_"):
                if measurement_id.endswith("_width") and value:
                    exam_data.serous_ped.width = value
                elif measurement_id.endswith("_height") and value:
                    exam_data.serous_ped.height = value
                elif measurement_id.endswith("_area") and value:
                    exam_data.serous_ped.area = value
                elif measurement_id.endswith("_location"):
                    exam_data.serous_ped.location = value_str
            
            # Геморрагическая ОПЭ
            elif measurement_id.startswith("hemorrhagic_ped_"):
                if measurement_id.endswith("_width") and value:
                    exam_data.hemorrhagic_ped.width = value
                elif measurement_id.endswith("_height") and value:
                    exam_data.hemorrhagic_ped.height = value
                elif measurement_id.endswith("_area") and value:
                    exam_data.hemorrhagic_ped.area = value
                elif measurement_id.endswith("_location"):
                    exam_data.hemorrhagic_ped.location = value_str
            
            # Фиброваскулярная ОПЭ
            elif measurement_id.startswith("fibrovascular_ped_"):
                if measurement_id.endswith("_width") and value:
                    exam_data.fibrovascular_ped.width = value
                elif measurement_id.endswith("_height") and value:
                    exam_data.fibrovascular_ped.height = value
                elif measurement_id.endswith("_area") and value:
                    exam_data.fibrovascular_ped.area = value
                elif measurement_id.endswith("_location"):
                    exam_data.fibrovascular_ped.location = value_str
            
            # Друзеноидная ОПЭ
            elif measurement_id.startswith("drusenoid_ped_"):
                if measurement_id.endswith("_width") and value:
                    exam_data.drusenoid_ped.width = value
                elif measurement_id.endswith("_height") and value:
                    exam_data.drusenoid_ped.height = value
                elif measurement_id.endswith("_area") and value:
                    exam_data.drusenoid_ped.area = value
                elif measurement_id.endswith("_location"):
                    exam_data.drusenoid_ped.location = value_str
            
            # Отслойка нейроэпителия
            elif measurement_id.startswith("neuroepithelial_detachment_"):
                if measurement_id.endswith("_width") and value:
                    exam_data.nsr_detachment.width = value
                elif measurement_id.endswith("_height") and value:
                    exam_data.nsr_detachment.height = value
                elif measurement_id.endswith("_area") and value:
                    exam_data.nsr_detachment.area = value
                elif measurement_id.endswith("_location"):
                    exam_data.nsr_detachment.location = value_str
            
            # Жидкость под РПЭ
            elif measurement_id == "sbf_area" and value:
                exam_data.sub_rpe_fluid.area = value
            elif measurement_id == "sbf_location":
                exam_data.sub_rpe_fluid.location = value_str
            
            # Гиперрефлективный материал
            elif "hyperreflective_" in measurement_id:
                if "area" in measurement_id and value:
                    exam_data.hyperreflective_material.area = value
                elif "location" in measurement_id:
                    exam_data.hyperreflective_material.location = value_str
        
        # Обрабатываем друзы - вычисляем средние значения
        drusen_widths = []
        drusen_heights = []
        drusen_areas = []
        drusen_locations = []
        
        for item in self._measurements:
            param = item.get("parameter", "")
            if "Drusen #" in param or ("друз" in param.lower() and "#" in param):
                value_str = item.get("value", "")
                try:
                    value = float(value_str) if value_str else None
                except ValueError:
                    value = None
                
                if "ширина" in param.lower() or "width" in param.lower():
                    if value is not None:
                        drusen_widths.append(value)
                elif "высота" in param.lower() or "height" in param.lower():
                    if value is not None:
                        drusen_heights.append(value)
                elif "площадь" in param.lower() or "area" in param.lower():
                    if value is not None:
                        drusen_areas.append(value)
                elif "локализация" in param.lower() or "location" in param.lower():
                    if value_str:
                        drusen_locations.append(value_str)
        
        # Заполняем средние значения для друз
        exam_data.drusen = Measurement()
        if drusen_widths:
            exam_data.drusen.width = sum(drusen_widths) / len(drusen_widths)
        if drusen_heights:
            exam_data.drusen.height = sum(drusen_heights) / len(drusen_heights)
        if drusen_areas:
            exam_data.drusen.area = sum(drusen_areas) / len(drusen_areas)
        if drusen_locations:
            # Для локализации берём наиболее частую
            from collections import Counter
            location_counts = Counter(drusen_locations)
            exam_data.drusen.location = location_counts.most_common(1)[0][0]
        
        return exam_data


# ====== Модель для деталей друзы/отслойки ======
class DetachmentDetailModel(QAbstractTableModel):
    HEADERS = ["Параметр", "Значение", "Единица"]
    
    def __init__(self, measurements: list[dict]):
        super().__init__()
        self._data = measurements
    
    def rowCount(self, parent=QModelIndex()):
        return len(self._data)
    
    def columnCount(self, parent=QModelIndex()):
        return len(self.HEADERS)
    
    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        
        if role == Qt.DisplayRole:
            row = self._data[index.row()]
            param = row.get("parameter", "")
            # Убираем номер друзы/префикс отслойки из названия параметра для компактности
            import re
            param = re.sub(r'Drusen #\d+ \(|\) #\d+|#\d+ ', '', param)
            param = re.sub(r'друз[а-я]* #\d+ \(|\)', '', param, flags=re.IGNORECASE)
            # Убираем префиксы отслоек
            param = re.sub(r'^(Серозная отслойка ПЭ \(СОПЭ\)|Геморрагическая отслойка ПЭ \(ГОПЭ\)|Фиброваскулярная отслойка ПЭ \(ФВОПЭ\)|Друзеноидная отслойка ПЭ|Отслойка нейроэпителия) \(', '', param)
            param = re.sub(r'\)$', '', param)
            
            return [
                param.strip(),
                row.get("value", ""),
                row.get("unit", ""),
            ][index.column()]
        
        return None
    
    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.HEADERS[section]
