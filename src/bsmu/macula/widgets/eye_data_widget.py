from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QGroupBox, QFormLayout, QLabel, QScrollArea

from bsmu.macula.records.eye_info_data import PatientExamData


class EyeDataWidget(QWidget):
    def __init__(self, eye_data: PatientExamData, parent=None):
        super().__init__(parent)
        self.eye_data = eye_data
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        scroll_area = self.create_scrollable_area()
        layout.addWidget(scroll_area)
        self.setLayout(layout)

    def create_block(self, title, fields):
        """Создание блока с формой для группы полей"""
        group_box = QGroupBox(title)
        form_layout = QFormLayout()

        for label_text, field_value in fields:
            label = QLabel(label_text)
            value = str(field_value) if field_value is not None else ""
            value_label = QLabel(value)
            value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            form_layout.addRow(label, value_label)

        group_box.setLayout(form_layout)
        return group_box

    def create_scrollable_area(self):
        """Создаёт область прокрутки с блоками данных"""
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout()

        scroll_layout.addWidget(self.create_block("Обследование", [
            ("Дата посещения", self.eye_data.visit_date),
            ("Продолжительность заболевания", self.eye_data.disease_duration),
            ("Тип томографа ", self.eye_data.tomograph_type),
            ("Стадия по AREDS", self.eye_data.areds_criteria),
            ("Рефракция", self.eye_data.refraction),
            ("Тип неоваскуляризации", self.eye_data.neovascularization_type)
        ]))
        scroll_layout.addWidget(self.create_block("Ретинальные показатели", [
            ("Толщина хориоидеи в центре", self.eye_data.choroidal_center_thickness),
            ("Толщина сетчатки в фовеоле", self.eye_data.foveal_retinal_thickness),
            ("Общий объем", self.eye_data.total_retinal_volume),
            ("Средний объем", self.eye_data.average_retinal_volume)
        ]))
        scroll_layout.addWidget(self.create_block("", [
            ("Состояние РПЭ", self.eye_data.rpe_status.condition),
            ("Локализация дефектов РПЭ", self.eye_data.rpe_status.defect_location),
            ("Локализация кистозного макулярного отека", self.eye_data.cmo_location)
        ]))
        scroll_layout.addWidget(QLabel("Отслойки РПЭ"))
        scroll_layout.addWidget(self.create_block("Серозная ОПЭ", [
            ("Локализация", self.eye_data.serous_ped.location),
            ("Ширина", self.eye_data.serous_ped.width),
            ("Высота", self.eye_data.serous_ped.height),
            ("Площадь", self.eye_data.serous_ped.area)
        ]))
        scroll_layout.addWidget(self.create_block("Геморрагическая ОПЭ", [
            ("Локализация", self.eye_data.hemorrhagic_ped.location),
            ("Ширина", self.eye_data.hemorrhagic_ped.width),
            ("Высота", self.eye_data.hemorrhagic_ped.height),
            ("Площадь", self.eye_data.hemorrhagic_ped.area)
        ]))
        scroll_layout.addWidget(self.create_block("Фиброваскулярная ОПЭ", [
            ("Локализация", self.eye_data.fibrovascular_ped.location),
            ("Ширина", self.eye_data.fibrovascular_ped.width),
            ("Высота", self.eye_data.fibrovascular_ped.height),
            ("Площадь", self.eye_data.fibrovascular_ped.area)
        ]))
        scroll_layout.addWidget(self.create_block("Друзеноидная ОПЭ", [
            ("Локализация", self.eye_data.drusenoid_ped.location),
            ("Ширина", self.eye_data.drusenoid_ped.width),
            ("Высота", self.eye_data.drusenoid_ped.height),
            ("Площадь", self.eye_data.drusenoid_ped.area)
        ]))
        scroll_layout.addWidget(self.create_block("Друзы", [
            ("Локализация", self.eye_data.drusen.location),
            ("Ширина", self.eye_data.drusen.width),
            ("Высота", self.eye_data.drusen.height),
            ("Площадь", self.eye_data.drusen.area)
        ]))
        scroll_layout.addWidget(self.create_block("Жидкость под РПЭ", [
            ("Пощадь", self.eye_data.sub_rpe_fluid.area),
            ("Локализация", self.eye_data.sub_rpe_fluid.location)
        ]))
        scroll_layout.addWidget(self.create_block("Эллипсоидная зона", [
            ("Состояние", self.eye_data.ellipsoid_zone.condition),
            ("Локализация дефектов", self.eye_data.ellipsoid_zone.defect_location)
        ]))
        scroll_layout.addWidget(self.create_block("Миоидная зона", [
            ("Состояние", self.eye_data.myoid_zone.condition),
            ("Локализация дефектов", self.eye_data.myoid_zone.defect_location)
        ]))
        scroll_layout.addWidget(self.create_block("Отслойка нейросенсорной сетчатки", [
            ("Локализация", self.eye_data.nsr_detachment.location),
            ("Ширина", self.eye_data.nsr_detachment.width),
            ("Высота", self.eye_data.nsr_detachment.height),
            ("Площадь", self.eye_data.nsr_detachment.area)
        ]))
        scroll_layout.addWidget(self.create_block("Гиперрефлективный материал", [
            ("Локализация", self.eye_data.hyperreflective_material.location),
            ("Площадь", self.eye_data.hyperreflective_material.area)
        ]))

        # Добавьте остальные блоки по аналогии...

        scroll_content.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_content)

        return scroll_area