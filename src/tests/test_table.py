"""Тесты таблицы результатов и сборки PatientExamData.

Запускаются в offscreen-режиме Qt, поэтому не требуют дисплея.
"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from bsmu.macula.plugins.analyser.analyzer import MaskAnalyser
from bsmu.macula.plugins.analyser.analyzed_table import (
    DetachmentDetailModel,
    ObjectsTableModel,
    TableWindow,
    group_detachment_measurements,
    group_drusen_measurements,
    group_measurements_data,
    short_parameter_name,
)
from bsmu.macula.plugins.analyser.data_converter import DataConverter, PatientExamDataBuilder
from bsmu.macula.plugins.analyser.schema import EDITABLE_PARAMETERS, Parameter

from tests.synthetic import build_class_configs, build_fovea_mask, build_image, build_mask


def _build_rows() -> list[dict]:
    analyser = MaskAnalyser(build_image(), build_mask(), build_fovea_mask(),
                            build_class_configs())
    return analyser.analyze()


class HelpersTestCase(unittest.TestCase):
    def test_short_parameter_name(self) -> None:
        self.assertEqual(short_parameter_name("Drusen #3 (ширина)"), "ширина")
        self.assertEqual(
            short_parameter_name("Серозная отслойка ПЭ (СОПЭ) (высота)"), "высота"
        )
        self.assertEqual(short_parameter_name("Состояние РПЭ"), "Состояние РПЭ")


class DataConverterTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = _build_rows()

    def test_builder_matches_converter(self) -> None:
        from_converter = DataConverter().to_patient_exam_data(self.rows)
        from_builder = PatientExamDataBuilder().build(self.rows)
        self.assertEqual(
            from_converter.choroidal_center_thickness, from_builder.choroidal_center_thickness
        )
        self.assertEqual(from_converter.serous_ped, from_builder.serous_ped)
        self.assertEqual(from_converter.drusen, from_builder.drusen)

    def test_detachments_and_fluid_are_filled(self) -> None:
        data = PatientExamDataBuilder().build(self.rows)
        self.assertIsNotNone(data.serous_ped.width)
        self.assertIsNotNone(data.serous_ped.height)
        self.assertIsNotNone(data.serous_ped.location)
        self.assertIsNotNone(data.sub_rpe_fluid.area)
        self.assertIsNotNone(data.sub_rpe_fluid.location)
        self.assertIsNotNone(data.hyperreflective_material.location)
        self.assertIsNotNone(data.hyperreflective_material.area)
        self.assertIsNotNone(data.nsr_detachment.width)

    def test_drusen_averages(self) -> None:
        data = PatientExamDataBuilder().build(self.rows)
        self.assertIsNotNone(data.drusen.width)
        self.assertIsNotNone(data.drusen.height)
        self.assertIsNotNone(data.drusen.area)
        self.assertIsNotNone(data.drusen.location)

    def test_manual_fields(self) -> None:
        rows = [
            {"parameter": Parameter.VISIT_DATE, "value": "2024-05-06", "unit": ""},
            {"parameter": Parameter.BCVA, "value": "0.7", "unit": ""},
            {"parameter": Parameter.RETINA_VOLUME, "value": "8.5", "unit": "мм³"},
            {"parameter": Parameter.AREDS, "value": "3", "unit": ""},
            {"parameter": "неизвестный параметр", "value": "42", "unit": ""},
        ]
        data = PatientExamDataBuilder().build(rows)
        self.assertEqual(str(data.visit_date), "2024-05-06")
        self.assertEqual(data.bcva, "0.7")
        self.assertEqual(data.total_retinal_volume, 8.5)
        self.assertEqual(data.areds_criteria, "3")

    def test_bad_date_is_ignored(self) -> None:
        data = PatientExamDataBuilder().build(
            [{"parameter": Parameter.VISIT_DATE, "value": "не дата", "unit": ""}]
        )
        self.assertIsNone(data.visit_date)


class TableWindowTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])
        cls.rows = _build_rows()

    def test_grouping_helpers(self) -> None:
        main_rows = group_measurements_data(self.rows)
        parameters = [row["parameter"] for row in main_rows]
        self.assertTrue(any("Состояние РПЭ" in parameter for parameter in parameters))
        self.assertFalse(any("#" in parameter for parameter in parameters))
        self.assertFalse(any("СОПЭ" in parameter for parameter in parameters))

        self.assertEqual(sorted(group_drusen_measurements(self.rows)), [1])
        detachments = group_detachment_measurements(self.rows)
        self.assertIn("serous_ped", detachments)
        self.assertIn("neuroepithelial_detachment", detachments)

    def test_model_exposes_rows_and_groups(self) -> None:
        model = ObjectsTableModel(self.rows)
        self.assertGreater(model.rowCount(), 0)
        self.assertEqual(model.columnCount(), 2)

        group_parameters = [
            model.get_row_data(model.index(row, 0))["parameter"]
            for row in range(model.rowCount())
            if model.get_row_data(model.index(row, 0)).get("is_group")
        ]
        self.assertEqual(len(group_parameters), 1)

    def test_manual_field_is_editable(self) -> None:
        model = ObjectsTableModel(self.rows)
        editable_rows = [
            row
            for row in range(model.rowCount())
            if model.get_row_data(model.index(row, 0))["parameter"] in EDITABLE_PARAMETERS
        ]
        self.assertTrue(editable_rows)
        index = model.index(editable_rows[0], 1)
        self.assertTrue(bool(model.flags(index) & Qt.ItemIsEditable))

    def test_set_data_updates_value(self) -> None:
        model = ObjectsTableModel(self.rows)
        row = next(
            row
            for row in range(model.rowCount())
            if model.get_row_data(model.index(row, 0))["parameter"] == Parameter.BCVA
        )
        index = model.index(row, 1)
        self.assertTrue(model.setData(index, "0.9"))
        self.assertIn("0.9", model.data(index))

    def test_window_builds_tables(self) -> None:
        window = TableWindow(self.rows, None, highlight_callback=lambda data: None)
        try:
            self.assertEqual(window.model.rowCount(), ObjectsTableModel(self.rows).rowCount())
            self.assertIn("serous_ped", window.detachment_tables)
            self.assertIn("serous_ped", window.detachment_models)
            self.assertEqual(window.drusen_combo.count(), 2)  # «(не выбрано)» + друза #1

            window.drusen_combo.setCurrentIndex(1)
            self.assertGreater(window.drusen_detail_model.rowCount(), 0)
            self.assertEqual(window.drusen_detail_model.columnCount(), 2)
        finally:
            window.deleteLater()

    def test_window_highlight_callback(self) -> None:
        seen: list = []
        window = TableWindow(self.rows, None, highlight_callback=seen.append)
        try:
            window._on_viewport_entered()
            self.assertEqual(seen[-1], None)
        finally:
            window.deleteLater()

    def test_detail_model_strips_prefixes(self) -> None:
        detachments = group_detachment_measurements(self.rows)
        model = DetachmentDetailModel(detachments["serous_ped"])
        names = {
            model.data(model.index(row, 0)) for row in range(model.rowCount())
        }
        self.assertEqual(names, {"ширина", "высота", "площадь", "локализация"})


if __name__ == "__main__":
    unittest.main()
