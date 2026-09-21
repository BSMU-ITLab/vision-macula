"""Регрессионные тесты анализатора масок.

Тесты проверяют, что после разделения ``analyzer_utils``/``mask_analyser``
на модули весь конвейер по-прежнему проходит от маски до строк таблицы и
объекта ``PatientExamData``, а ключевые измерения остаются осмысленными.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from bsmu.macula.plugins.analyser.analyzer import ANALYSIS_SIZE, MaskAnalyser
from bsmu.macula.plugins.analyser.geometry import (
    extract_upper_boundary,
    get_foveola_center,
    smooth_boundary,
)
from bsmu.macula.plugins.analyser.location import (
    location_by_check_order,
    location_by_zone,
    location_from_fovea_mask,
)
from bsmu.macula.plugins.analyser.measurements.detachment import (
    measure_detachment,
    measure_neuroepithelial_detachment,
)
from bsmu.macula.plugins.analyser.measurements.drusen import measure_drusen
from bsmu.macula.plugins.analyser.measurements.kernel import width_along_boundary
from bsmu.macula.plugins.analyser.measurements.rpe import (
    analyze_rpe,
    combined_rpe_state,
    detect_rpe_defects,
    measure_rpe_thickness,
    rpe_mask,
)
from bsmu.macula.plugins.analyser.scale import calculate_scale_from_L, calculate_scales_from_L
from bsmu.macula.plugins.analyser.schema import (
    CLASS_NEUROEPITHELIAL_DETACHMENT,
    DETACHMENT_BY_CLASS,
    Location,
    MeasurementId,
    SEROUS_PED,
    drusen_number,
    is_drusen_class,
    is_drusen_parameter,
    to_um,
    to_um2,
)

from tests.synthetic import (
    CLASS_DRUSEN,
    CLASS_SEROUS_PED,
    build_class_configs,
    build_fovea_mask,
    build_image,
    build_mask,
)


class GeometryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.mask = build_mask()

    def test_upper_boundary_is_found(self) -> None:
        xs, upper = extract_upper_boundary(self.mask)
        self.assertIsNotNone(xs)
        self.assertIsNotNone(upper)
        self.assertEqual(len(xs), self.mask.shape[1])
        self.assertTrue(150 < float(upper[300]) < 220)

    def test_foveola_center(self) -> None:
        center = get_foveola_center(build_fovea_mask())
        self.assertEqual(center, (299, 149))

    def test_foveola_center_absent(self) -> None:
        self.assertIsNone(get_foveola_center(build_fovea_mask() * 0))

    def test_smooth_boundary_returns_spline(self) -> None:
        xs, upper = extract_upper_boundary(self.mask)
        smoothed, spline = smooth_boundary(xs, upper, smooth_factor=1000)
        self.assertEqual(len(smoothed), len(xs))
        self.assertIsNotNone(spline.derivative()(300))


class LocationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.fovea = build_fovea_mask()

    def test_zone_priority(self) -> None:
        self.assertEqual(
            location_by_zone(True, True, True), Location.FOVEOLA_FOVEA_MACULA
        )
        self.assertEqual(location_by_zone(False, True, True), Location.FOVEA_MACULA)
        self.assertEqual(location_by_zone(False, False, True), Location.MACULA_ONLY)
        self.assertEqual(location_by_zone(False, False, False), Location.ABSENT)

    def test_order_priority(self) -> None:
        self.assertEqual(location_by_check_order(True, False, False), Location.FOVEOLA)
        self.assertEqual(location_by_check_order(False, True, False), Location.FOVEA)
        self.assertEqual(location_by_check_order(False, False, True), Location.MACULA)
        self.assertEqual(location_by_check_order(False, False, False), Location.OUTSIDE_MACULA)

    def test_location_from_mask(self) -> None:
        object_mask = rpe_mask(build_mask())
        zones = self.fovea[object_mask > 0]
        expected = Location.FOVEOLA_FOVEA_MACULA if 1 in zones else Location.FOVEA_MACULA
        self.assertEqual(location_from_fovea_mask(self.fovea, object_mask), expected)
        self.assertIsNone(location_from_fovea_mask(self.fovea, object_mask * 0))


class SchemaTestCase(unittest.TestCase):
    def test_drusen_class_detection(self) -> None:
        self.assertTrue(is_drusen_class("Drusen"))
        self.assertTrue(is_drusen_class("друзы"))
        self.assertFalse(is_drusen_class("Друзеноидная отслойка ПЭ"))
        self.assertFalse(is_drusen_class("Serous PED"))

    def test_drusen_parameter_helpers(self) -> None:
        self.assertTrue(is_drusen_parameter("Drusen #2 (ширина)"))
        self.assertFalse(is_drusen_parameter("Состояние РПЭ"))
        self.assertEqual(drusen_number("Drusen #12 (площадь)"), 12)
        self.assertIsNone(drusen_number("Состояние РПЭ"))

    def test_unit_conversions(self) -> None:
        self.assertAlmostEqual(to_um2(10, 2.0, 3.0), 60.0)
        self.assertIsNone(to_um2(10, None, 3.0))
        self.assertAlmostEqual(to_um((0, 0), (3, 4), 1.0, 1.0), 5.0)
        self.assertIsNone(to_um((0, 0), (3, 4), 1.0, None))


class MeasurementTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.mask = build_mask()
        xs, upper = extract_upper_boundary(self.mask)
        self.smooth_upper, self.spline = smooth_boundary(xs, upper, smooth_factor=1000)
        self.fovea = build_fovea_mask()
        self.scale_x, self.scale_y = 2.0, 3.0

    def test_width_along_boundary_uses_only_object_columns(self) -> None:
        target = (self.mask == CLASS_SEROUS_PED).astype("uint8")
        width_um, segment = width_along_boundary(
            self.smooth_upper, target, self.scale_x, self.scale_y
        )
        self.assertIsNotNone(width_um)
        self.assertGreater(width_um, 0)
        self.assertTrue(all(point is not None for point in segment))

    def test_serous_ped_measurement(self) -> None:
        width, height, area, left_x, right_x, max_point, location, segment = measure_detachment(
            self.mask, self.smooth_upper, self.spline, SEROUS_PED,
            self.fovea, self.scale_x, self.scale_y,
        )
        self.assertIsNotNone(width)
        self.assertIsNotNone(height)
        self.assertIsNotNone(area)
        self.assertIsNotNone(max_point)
        self.assertEqual(left_x, 120)
        self.assertEqual(right_x, 179)
        self.assertIsNotNone(location)
        self.assertTrue(segment)

    def test_absent_detachment_returns_empty(self) -> None:
        spec = DETACHMENT_BY_CLASS[16]
        result = measure_detachment(self.mask, self.smooth_upper, self.spline, spec)
        self.assertTrue(all(item is None for item in result))

    def test_drusen_measurement(self) -> None:
        import cv2

        target = (self.mask == CLASS_DRUSEN).astype("uint8")
        contours, _ = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.assertEqual(len(contours), 1)

        width, height, area, left_x, right_x, max_point, location, segment = measure_drusen(
            self.mask, self.smooth_upper, self.spline, contours[0],
            self.fovea, self.scale_x, self.scale_y,
        )
        self.assertIsNotNone(width)
        self.assertIsNotNone(height)
        self.assertIsNotNone(area)
        self.assertEqual((left_x, right_x), (240, 299))
        self.assertIsNone(location)
        self.assertTrue(segment)

    def test_neuroepithelial_detachment_measurement(self) -> None:
        width, height, area, left_x, right_x, max_point, location, segment = (
            measure_neuroepithelial_detachment(
                self.mask, self.smooth_upper, self.spline,
                CLASS_NEUROEPITHELIAL_DETACHMENT,
                self.fovea, self.scale_x, self.scale_y,
            )
        )
        self.assertIsNotNone(width)
        self.assertIsNotNone(height)
        self.assertEqual((left_x, right_x), (360, 419))
        # Очаг лежит правее фовеолы, поэтому локализация — макула по порядку зон.
        self.assertEqual(location, Location.MACULA)


class RpeTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.mask = build_mask()
        self.fovea = build_fovea_mask()
        xs, upper = extract_upper_boundary(self.mask)
        self.smooth_upper, self.spline = smooth_boundary(xs, upper, smooth_factor=1000)

    def test_gap_is_detected(self) -> None:
        location, gap_contours, coordinate = detect_rpe_defects(
            self.mask, self.fovea, self.smooth_upper, self.spline
        )
        self.assertEqual(len(gap_contours), 1)
        self.assertIsNotNone(coordinate)
        self.assertIn(location, (Location.FOVEA_MACULA, Location.FOVEOLA_FOVEA_MACULA))

    def test_thickness_is_measured(self) -> None:
        skeleton, perpendiculars, mean, std = measure_rpe_thickness(
            self.mask, self.fovea, 2.0, 3.0
        )
        self.assertIsNotNone(skeleton)
        self.assertTrue(perpendiculars)
        self.assertIsNotNone(mean)
        self.assertIsNotNone(std)

    def test_state_with_single_gap(self) -> None:
        state = analyze_rpe(self.mask, self.fovea, self.smooth_upper, self.spline)
        self.assertEqual(state, "Единичные разрывы")

    def test_state_without_rpe(self) -> None:
        empty = self.mask * 0
        state = analyze_rpe(empty, self.fovea, self.smooth_upper, self.spline)
        self.assertEqual(state, "Эпителий не определяется")
        self.assertEqual(
            combined_rpe_state(rpe_mask(empty), [], None, state),
            "Эпителий не определяется",
        )

    def test_combined_state_priority(self) -> None:
        rpe = rpe_mask(self.mask)
        gaps = [1, 2, 3, 4]
        self.assertEqual(
            combined_rpe_state(rpe, gaps, 10.0, "Эпителий сохранен"), "Множественные разрывы"
        )
        self.assertEqual(
            combined_rpe_state(rpe, gaps[:1], 10.0, "Эпителий сохранен"), "Единичные разрывы"
        )
        self.assertEqual(
            combined_rpe_state(rpe, [], 10.0, "Эпителий сохранен"), "Эпителий сохранен"
        )
        self.assertEqual(
            combined_rpe_state(rpe, [], 90.0, "Эпителий сохранен"), "Эпителий неравномерный"
        )


class ScaleTestCase(unittest.TestCase):
    def test_scale_from_l_object(self) -> None:
        image = build_image()
        from bsmu.macula.plugins.analyser.scale import find_L_in_image

        contour = find_L_in_image(image)
        self.assertIsNotNone(contour)

        scale = calculate_scale_from_L(contour)
        scales = calculate_scales_from_L(contour)
        self.assertIsNotNone(scale)
        self.assertIsNotNone(scales)
        self.assertAlmostEqual(scales[0], 200.0 / scales[2])
        self.assertAlmostEqual(scales[1], 200.0 / scales[3])

    def test_invalid_contour(self) -> None:
        self.assertIsNone(calculate_scale_from_L(None))
        self.assertIsNone(calculate_scales_from_L(None))


class AnalyzerPipelineTestCase(unittest.TestCase):
    """Сквозной прогон: маска -> строки таблицы -> PatientExamData."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mask = build_mask()
        cls.fovea = build_fovea_mask()
        cls.image = build_image()
        cls.analyser = MaskAnalyser(
            cls.image, cls.mask, cls.fovea, build_class_configs()
        )
        cls.rows = cls.analyser.analyze()

    def _rows_by_id(self, measurement_id: str) -> list[dict]:
        return [row for row in self.rows if row.get("measurement_id") == measurement_id]

    def test_analysis_size_constant(self) -> None:
        self.assertEqual(ANALYSIS_SIZE, (1024, 512))

    def test_manual_group_is_first(self) -> None:
        first = self.rows[0]
        self.assertTrue(first["is_group"])
        self.assertEqual(len(first["children"]), 10)

    def test_reference_scale_is_applied(self) -> None:
        self.assertAlmostEqual(self.analyser.scale_x, 200.0 / 24.0)
        self.assertAlmostEqual(self.analyser.scale_y, 200.0 / 60.0)

    def test_expected_measurements_present(self) -> None:
        expected_ids = [
            MeasurementId.CTS_FOVEOLA,
            MeasurementId.CHOROID_THICKNESS_CENTER,
            MeasurementId.CTS_NEAR_FOVEOLA,
            MeasurementId.CTS_NEAR_FOVEA,
            "serous_ped_width",
            "serous_ped_height",
            "serous_ped_area",
            "serous_ped_location",
            "neuroepithelial_detachment_width",
            "neuroepithelial_detachment_height",
            "drusen_1_width",
            "drusen_1_height",
            "drusen_1_area",
            "drusen_1_location",
            MeasurementId.SBF_AREA,
            MeasurementId.SBF_LOCATION,
            MeasurementId.HYPERREFLECTIVE_LOCATION,
            MeasurementId.HYPERREFLECTIVE_SUBRETINAL_AREA,
            MeasurementId.HYPERREFLECTIVE_INTRARETINAL_AREA,
            MeasurementId.HYPERREFLECTIVE_TOTAL_AREA,
        ]
        present = {row.get("measurement_id") for row in self.rows}
        for measurement_id in expected_ids:
            self.assertIn(measurement_id, present, measurement_id)

    def test_rpe_rows_exist(self) -> None:
        parameters = {row["parameter"] for row in self.rows}
        self.assertIn("Состояние РПЭ", parameters)
        self.assertIn("Локализация дефектов РПЭ", parameters)
        self.assertIn("Состояние эллипсоидной зоны", parameters)
        self.assertIn("Состояние миоидной зоны", parameters)
        self.assertIn("Локализация кистозного макулярного отека", parameters)

    def test_numeric_values_are_finite(self) -> None:
        for row in self.rows:
            if row.get("unit") in ("мкм", "мкм²"):
                value = float(row["value"])
                self.assertTrue(value == value and abs(value) != float("inf"), row["parameter"])

    def test_detachment_rows_share_bounds(self) -> None:
        rows = [row for row in self.rows if "serous_ped" in row.get("measurement_id", "")]
        self.assertEqual(len(rows), 4)
        bounds = {id(row["detachment_bounds"]) for row in rows}
        self.assertEqual(len(bounds), 1)

    def test_location_matches_order_mode(self) -> None:
        from bsmu.macula.plugins.analyser.analyzer import MaskAnalyser

        analyser = MaskAnalyser(build_image(), build_mask(), build_fovea_mask(),
                                build_class_configs())
        rows = analyser.analyze()
        by_id = {row.get("measurement_id"): row for row in rows}

        self.assertEqual(
            by_id[MeasurementId.SBF_LOCATION]["value"], Location.MACULA
        )
        self.assertEqual(
            by_id["neuroepithelial_detachment_location"]["value"], Location.MACULA
        )
        self.assertEqual(
            by_id["serous_ped_location"]["value"], Location.MACULA_ONLY
        )
        self.assertEqual(
            by_id["drusen_1_location"]["value"], Location.FOVEOLA_FOVEA_MACULA
        )

    def test_patient_exam_data(self) -> None:
        data = self.analyser.to_patient_exam_data(self.rows)
        self.assertIsNotNone(data)
        self.assertIsNotNone(data.choroidal_center_thickness)
        self.assertIsNotNone(data.foveal_retinal_thickness)
        self.assertIsNotNone(data.serous_ped.width)
        self.assertIsNotNone(data.serous_ped.height)
        self.assertIsNotNone(data.serous_ped.location)
        self.assertIsNotNone(data.rpe_status.condition)
        self.assertIsNotNone(data.sub_rpe_fluid.area)
        self.assertIsNotNone(data.hyperreflective_material.location)

    def test_analyze_is_empty_without_choroid(self) -> None:
        analyser = MaskAnalyser(self.image, self.mask * 0, self.fovea, build_class_configs())
        rows = analyser.analyze()
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["is_group"])


class DeletedFacadeTestCase(unittest.TestCase):
    """Старые монолитные модули удалены и не должны возвращаться."""

    def test_legacy_modules_are_gone(self) -> None:
        import importlib

        for module_name in (
            "bsmu.macula.plugins.analyser.analyzer_utils",
            "bsmu.macula.plugins.analyser.mask_analyser",
            "bsmu.macula.plugins.analyser.visualization",
        ):
            with self.assertRaises(ModuleNotFoundError, msg=module_name):
                importlib.import_module(module_name)

    def test_plugin_entry_point_matches_config(self) -> None:
        """Каждый плагин из conf.yaml должен существовать и быть Plugin-классом."""
        import importlib
        import re
        from pathlib import Path

        config = (
            Path(__file__).resolve().parent.parent
            / "bsmu/macula/configs/default/bsmu.macula/app.MaculaApp.conf.yaml"
        )
        # Файл конфигурации записан в смешанной кодировке, поэтому читаем с
        # заменой недекодируемых байтов: имена плагинов в нём ASCII.
        text = config.read_text(encoding="utf-8", errors="replace")

        expected = "bsmu.macula.plugins.analyser.plugin.MaskAnalyserPlugin"
        self.assertIn(expected, text, "конфиг не ссылается на новый модуль плагина")

        matches = re.findall(r"(bsmu\.macula\.[\w.]+)", text)
        self.assertTrue(matches, "в конфиге не найдены плагины bsmu.macula")

        for dotted in matches:
            module_name, _, attribute = dotted.rpartition(".")
            try:
                module = importlib.import_module(module_name)
            except ModuleNotFoundError:
                continue
            self.assertTrue(hasattr(module, attribute), dotted)

            # Ровно эту проверку делает bsmu.vision при старте приложения:
            # класс из plugins: обязан быть Plugin-ом, иначе AttributeError.
            plugin_class = getattr(module, attribute)
            self.assertTrue(
                hasattr(plugin_class, "default_dependency_plugin_full_name_by_key"),
                f"{dotted} указан в plugins:, но не является Plugin-классом",
            )

    def test_macula_plugin_configs_are_found(self) -> None:
        """Файлы конфигов должны называться так, как их ищет bsmu.vision.

        Фреймворк строит имя как ``<пакет класса>.<ClassName>.conf.yaml``,
        поэтому ``plugins.BD.conf.yaml`` не читался вообще: у BD настоящий
        файл — ``plugins.db.BD.conf.yaml``.
        """
        import importlib
        from pathlib import Path

        from bsmu.macula.app.app import MaculaApp
        from bsmu.vision.core.config.united import UnitedConfig

        UnitedConfig.configure_app_class(MaculaApp)

        # Имена файлов заданы фреймворком: <пакет класса без имени приложения>.
        entries = (
            ("bsmu.macula.plugins.main_window", "MaculaMainWindowPlugin",
             "plugins.MaculaMainWindowPlugin.conf.yaml"),
            ("bsmu.macula.plugins.db.BD", "BD", "plugins.db.BD.conf.yaml"),
            ("bsmu.macula.plugins.analyser.plugin", "MaskAnalyserPlugin",
             "plugins.analyser.MaskAnalyserPlugin.conf.yaml"),
            ("bsmu.macula.plugins.ensemble_segmenter", "BinaryEnsemblePlugin",
             "plugins.BinaryEnsemblePlugin.conf.yaml"),
        )
        for module_name, attribute, expected_name in entries:
            plugin_class = getattr(importlib.import_module(module_name), attribute)
            paths = UnitedConfig(plugin_class, plugin_class).priority_config_paths
            names = [Path(path).name for path in paths]
            self.assertIn(
                expected_name, names,
                f"{expected_name} не входит в пути конфигов {plugin_class}: {names}",
            )
            self.assertTrue(
                any(Path(path).exists() for path in paths),
                f"ни один конфиг не найден для {module_name}.{attribute}: {list(paths)}",
            )

    def test_configs_are_ascii_only(self) -> None:
        """В конфигах допустим только ASCII.

        ruamel.yaml читает conf.yaml в кодировке локали (cp1250 на этой машине)
        и пишет результат в ней же, поэтому не-ASCII байты (например, русские
        комментарии) роняют приложение на старте с UnicodeDecodeError.
        """
        configs_root = Path(__file__).resolve().parent.parent / "bsmu/macula/configs"

        offenders = {}
        for path in sorted(configs_root.rglob("*.yaml")):
            extra = [byte for byte in path.read_bytes() if byte > 0x7F]
            if extra:
                offenders[path.name] = len(extra)

        self.assertEqual(offenders, {}, f"в конфигах есть не-ASCII байты: {offenders}")

    def test_measurement_id_prefixes_match_specs(self) -> None:
        for spec in DETACHMENT_BY_CLASS.values():
            self.assertTrue(spec.row_id("width").startswith(spec.measurement_id))


if __name__ == "__main__":
    unittest.main()
