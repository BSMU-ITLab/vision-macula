"""Тесты структур данных осмотра и вспомогательного инференса."""

from __future__ import annotations

import unittest

import numpy as np
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt

from bsmu.macula.inference.utility import RoiTiler, preprocess_for_model, reverse_preprocess
from bsmu.macula.records.eye_info_data import (
    ExamColumn,
    Measurement,
    PatientExamData,
    ZoneStatus,
)


class _StubModel(QAbstractTableModel):
    """Минимальная модель со строкой значений — замена QSqlQueryModel."""

    def __init__(self, values: dict[int, object]):
        super().__init__()
        self._values = values

    def rowCount(self, parent=QModelIndex()) -> int:
        return 1

    def columnCount(self, parent=QModelIndex()) -> int:
        return max(self._values) + 1

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role != Qt.DisplayRole:
            return None
        return self._values.get(index.column())


class MeasurementTestCase(unittest.TestCase):
    def test_from_model(self) -> None:
        model = _StubModel({0: "фовеола", 1: 12.5, 2: "3.5", 3: None})
        measurement = Measurement.from_model(model, 0, 0, 1, 2, 3)
        self.assertEqual(measurement.location, "фовеола")
        self.assertEqual(measurement.width, 12.5)
        self.assertEqual(measurement.height, 3.5)
        self.assertIsNone(measurement.area)

    def test_from_model_with_bad_numbers(self) -> None:
        model = _StubModel({0: None, 1: "не число", 2: "", 3: "1.0"})
        measurement = Measurement.from_model(model, 0, 0, 1, 2, 3)
        self.assertIsNone(measurement.location)
        self.assertIsNone(measurement.width)
        self.assertIsNone(measurement.height)
        self.assertEqual(measurement.area, 1.0)


class PatientExamDataTestCase(unittest.TestCase):
    def test_defaults(self) -> None:
        data = PatientExamData()
        self.assertEqual(data.tomograph_type, "Топкон")
        self.assertIsInstance(data.rpe_status, ZoneStatus)
        self.assertIsInstance(data.drusen, Measurement)
        self.assertIsInstance(data.sub_rpe_fluid, Measurement)
        self.assertIsNone(data.areds_criteria)

    def test_drusen_default_is_not_shared(self) -> None:
        first, second = PatientExamData(), PatientExamData()
        first.drusen.width = 10.0
        self.assertIsNone(second.drusen.width)

    def test_from_query_model_uses_column_map(self) -> None:
        values = {
            ExamColumn.VISIT_DATE: "не дата",
            ExamColumn.DISEASE_DURATION: "5 лет",
            ExamColumn.AREDS: "AREDS2",
            ExamColumn.REFRACTION: "-1.5",
            ExamColumn.NEOVASCULARIZATION: "Classic",
            ExamColumn.CHOROIDAL_THICKNESS: 250,
            ExamColumn.FOVEAL_THICKNESS: "150",
            ExamColumn.TOTAL_VOLUME: 8.5,
            ExamColumn.AVERAGE_VOLUME: 3.2,
            ExamColumn.RPE_CONDITION: "Эпителий сохранен",
            ExamColumn.RPE_DEFECT_LOCATION: "300,170",
            ExamColumn.CME_LOCATION: "0 - Отек отсутствует",
            ExamColumn.SUB_RPE_FLUID_AREA: 1.5,
            ExamColumn.SUB_RPE_FLUID_LOCATION: "0 - Фовеола",
            ExamColumn.HYPERREFLECTIVE_LOCATION: "Субретинальный",
            ExamColumn.HYPERREFLECTIVE_AREA: 2.0,
            19: "3 - Макула", 20: 10.0, 21: 5.0, 22: 50.0,
            41: "Сохранена", 42: None,
            43: "Сохранена", 44: None,
            45: None, 46: 1.0, 47: 2.0, 48: 3.0,
        }
        data = PatientExamData.from_query_model(_StubModel(values), 0)
        self.assertEqual(data.disease_duration, "5 лет")
        self.assertEqual(data.areds_criteria, "AREDS2")
        self.assertEqual(data.choroidal_center_thickness, 250.0)
        self.assertEqual(data.foveal_retinal_thickness, 150.0)
        self.assertEqual(data.rpe_status.condition, "Эпителий сохранен")
        self.assertEqual(data.cmo_location, "0 - Отек отсутствует")
        self.assertEqual(data.serous_ped.width, 10.0)
        self.assertEqual(data.serous_ped.area, 50.0)
        self.assertEqual(data.sub_rpe_fluid.location, "0 - Фовеола")
        self.assertEqual(data.ellipsoid_zone.condition, "Сохранена")
        self.assertEqual(data.nsr_detachment.height, 2.0)
        self.assertEqual(data.hyperreflective_material.area, 2.0)


class InferenceUtilityTestCase(unittest.TestCase):
    def test_roi_tiler_without_contours_yields_whole_image(self) -> None:
        sample = np.full((20, 30), 10, dtype=np.uint8)
        tiler = RoiTiler()
        tiles = list(tiler.split(sample))

        self.assertEqual(len(tiles), 1)
        self.assertEqual(tiles[0].shape, sample.shape)
        tiler.update(np.ones_like(tiles[0]))
        self.assertEqual(tiler.assemble().shape, sample.shape)

    def test_preprocess_roundtrip_shape(self) -> None:
        image = np.full((100, 200), 128, dtype=np.uint8)
        preprocessed, content_shape = preprocess_for_model(image, (64, 32))
        self.assertEqual(preprocessed.shape, (32, 64))
        self.assertLessEqual(content_shape[0], 32)

        restored = reverse_preprocess(np.zeros((32, 64), dtype=np.float32), content_shape, (100, 200))
        self.assertEqual(restored.shape, (100, 200))


if __name__ == "__main__":
    unittest.main()
