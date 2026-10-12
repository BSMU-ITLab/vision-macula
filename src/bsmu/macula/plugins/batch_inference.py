"""Batch inference over images in a selected folder."""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from bsmu.vision.core.concurrent import ThreadPool
from bsmu.vision.core.plugins import Plugin
from bsmu.vision.core.task import DnnTask
from bsmu.vision.plugins.windows.main import AlgorithmsMenu, MainWindowPlugin

if TYPE_CHECKING:
    from bsmu.macula.plugins.ensemble_segmenter import BinaryEnsemblePlugin
    from bsmu.macula.plugins.keypoint_predictor import KeypointPredictorPlugin
    from bsmu.vision.plugins.storages.task import TaskStoragePlugin
    from bsmu.vision.plugins.windows.main import MainWindow


SUPPORTED_SUFFIXES = {'.jpg', '.jpeg', '.png'}
LAYER_SEG_DIR_NAME = 'layer_seg'
KP_SEG_DIR_NAME = 'kp_seg'


@dataclass
class BatchInferenceSummary:
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)
    fatal_error: str | None = None


class BatchInferenceDialog(QDialog):
    """Folder and option fields for a batch run."""

    start_requested = Signal(dict)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(self.tr('Batch Inference'))
        self.setMinimumWidth(520)

        self.input_folder_edit = QLineEdit()
        self.output_folder_edit = QLineEdit()
        self.layer_seg_checkbox = QCheckBox(self.tr('Include layer_seg'))
        self.kp_seg_checkbox = QCheckBox(self.tr('Include kp_seg'))
        self.overwrite_checkbox = QCheckBox(self.tr('Overwrite existing masks'))
        self.recursive_checkbox = QCheckBox(self.tr('Process folders recursively'))
        self.status_label = QLabel()

        input_row = self._folder_row(self.input_folder_edit, self._browse_input_folder)
        output_row = self._folder_row(self.output_folder_edit, self._browse_output_folder)

        form = QFormLayout()
        form.addRow(self.tr('Input folder:'), input_row)
        form.addRow(self.tr('Output folder:'), output_row)

        options_layout = QVBoxLayout()
        options_layout.addWidget(self.layer_seg_checkbox)
        options_layout.addWidget(self.kp_seg_checkbox)
        options_layout.addWidget(self.overwrite_checkbox)
        options_layout.addWidget(self.recursive_checkbox)

        self.start_button = QPushButton(self.tr('Start'))
        self.close_button = QPushButton(self.tr('Close'))
        self.start_button.clicked.connect(self._start)
        self.close_button.clicked.connect(self.close)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.close_button)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(options_layout)
        layout.addWidget(self.status_label)
        layout.addLayout(buttons)

    @staticmethod
    def _folder_row(edit: QLineEdit, browse_callback) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit, 1)
        browse_button = QPushButton('…')
        browse_button.setFixedWidth(36)
        browse_button.clicked.connect(browse_callback)
        layout.addWidget(browse_button)
        return row

    def _browse_input_folder(self) -> None:
        self._browse_folder(self.input_folder_edit, self.tr('Select input folder'))

    def _browse_output_folder(self) -> None:
        self._browse_folder(self.output_folder_edit, self.tr('Select output folder'))

    def _browse_folder(self, edit: QLineEdit, title: str) -> None:
        folder = QFileDialog.getExistingDirectory(self, title, edit.text())
        if folder:
            edit.setText(folder)

    def _start(self) -> None:
        self.start_requested.emit({
            'input_folder': self.input_folder_edit.text().strip(),
            'output_folder': self.output_folder_edit.text().strip(),
            'include_layer_seg': self.layer_seg_checkbox.isChecked(),
            'include_kp_seg': self.kp_seg_checkbox.isChecked(),
            'overwrite': self.overwrite_checkbox.isChecked(),
            'recursive': self.recursive_checkbox.isChecked(),
        })

    def set_running(self, running: bool) -> None:
        for widget in (
                self.input_folder_edit,
                self.output_folder_edit,
                self.layer_seg_checkbox,
                self.kp_seg_checkbox,
                self.overwrite_checkbox,
                self.recursive_checkbox):
            widget.setEnabled(not running)
        self.start_button.setEnabled(not running)
        self.close_button.setEnabled(not running)
        self.status_label.setText(self.tr('Inference is running…') if running else '')

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt virtual method
        if not self.close_button.isEnabled():
            event.ignore()
            return
        super().closeEvent(event)


class BatchInferenceTask(DnnTask):
    """Sequentially infer and write selected masks for a folder of images."""

    def __init__(
            self,
            input_folder: Path,
            output_folder: Path,
            recursive: bool,
            include_layer_seg: bool,
            include_kp_seg: bool,
            overwrite: bool,
            segmenter,
            keypoint_predictor,
    ):
        super().__init__('Batch Inference')
        self._input_folder = input_folder
        self._output_folder = output_folder
        self._recursive = recursive
        self._include_layer_seg = include_layer_seg
        self._include_kp_seg = include_kp_seg
        self._overwrite = overwrite
        self._segmenter = segmenter
        self._keypoint_predictor = keypoint_predictor

    def _run(self) -> BatchInferenceSummary:
        summary = BatchInferenceSummary()
        try:
            images = self._discover_images()
        except Exception as error:
            summary.fatal_error = str(error)
            return summary

        if not images:
            summary.fatal_error = self.tr('No supported images were found.')
            return summary

        images, conflicts = self._remove_output_name_conflicts(images)
        for conflict in conflicts:
            summary.failed += len(conflict)
            summary.errors.append(
                self.tr('Output filename conflict; skipped: {}').format(
                    ', '.join(str(path) for path in conflict)
                )
            )

        for image_path in images:
            try:
                self._process_image(image_path, summary)
            except Exception as error:
                summary.failed += 1
                summary.errors.append(
                    self.tr('Batch processing failed for {}: {}').format(image_path, error)
                )
        return summary

    def _discover_images(self) -> list[Path]:
        if not self._input_folder.is_dir():
            raise ValueError(self.tr('The input folder does not exist or is not a directory.'))

        output_resolved = self._output_folder.resolve()
        output_is_input = output_resolved == self._input_folder
        output_is_nested = output_resolved != self._input_folder and output_resolved.is_relative_to(
            self._input_folder
        )
        generated_outputs = {
            output_resolved / LAYER_SEG_DIR_NAME,
            output_resolved / KP_SEG_DIR_NAME,
        }

        paths: list[Path] = []
        if not self._recursive:
            candidates = self._input_folder.iterdir()
            for path in candidates:
                if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                    paths.append(path)
        else:
            for root, dirs, files in os.walk(self._input_folder):
                root_path = Path(root)
                if output_is_input and root_path == self._input_folder:
                    dirs[:] = [
                        directory for directory in dirs
                        if directory not in {LAYER_SEG_DIR_NAME, KP_SEG_DIR_NAME}
                    ]
                elif output_is_nested:
                    dirs[:] = [
                        directory for directory in dirs
                        if (root_path / directory).resolve() not in generated_outputs
                    ]

                dirs.sort()
                for file_name in sorted(files):
                    path = root_path / file_name
                    if path.suffix.lower() in SUPPORTED_SUFFIXES and path.is_file():
                        paths.append(path)

        return sorted(paths)

    def _remove_output_name_conflicts(
            self, images: list[Path]) -> tuple[list[Path], list[list[Path]]]:
        groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
        for image_path in images:
            relative_parent = image_path.parent.relative_to(self._input_folder).as_posix()
            groups[(relative_parent.casefold(), image_path.stem.casefold())].append(image_path)

        conflicts = [paths for paths in groups.values() if len(paths) > 1]
        conflicting_paths = {path for conflict in conflicts for path in conflict}
        return [path for path in images if path not in conflicting_paths], conflicts

    def _process_image(self, image_path: Path, summary: BatchInferenceSummary) -> None:
        relative_parent = image_path.parent.relative_to(self._input_folder)
        layer_output = (
            self._output_folder / LAYER_SEG_DIR_NAME / relative_parent / f'{image_path.stem}.png'
            if self._include_layer_seg else None
        )
        kp_output = (
            self._output_folder / KP_SEG_DIR_NAME / relative_parent / f'{image_path.stem}.png'
            if self._include_kp_seg else None
        )

        need_layer = layer_output is not None and (self._overwrite or not layer_output.is_file())
        need_kp = kp_output is not None and (self._overwrite or not kp_output.is_file())
        if not need_layer and not need_kp:
            summary.skipped += 1
            return

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            summary.failed += 1
            summary.errors.append(self.tr('Could not read image: {}').format(image_path))
            return

        image_failed = False
        generated = False
        segmentation = None

        if need_layer:
            try:
                segmentation = self._segmenter.predict(image).astype(np.uint8, copy=False)
                generated = self._write_mask(
                    layer_output, segmentation, self._overwrite
                ) or generated
            except Exception as error:
                image_failed = True
                summary.errors.append(
                    self.tr('Ensemble segmentation failed for {}: {}').format(image_path, error)
                )
        elif need_kp and layer_output is not None and layer_output.is_file():
            try:
                segmentation = cv2.imread(str(layer_output), cv2.IMREAD_UNCHANGED)
                if segmentation is not None and segmentation.ndim == 3:
                    segmentation = segmentation[..., 0]
            except Exception:
                # The predictor can derive a foreground mask itself.
                segmentation = None

        if need_kp:
            try:
                keypoint_mask = self._keypoint_predictor.predict(image, segmentation)
                generated = self._write_mask(
                    kp_output, keypoint_mask, self._overwrite
                ) or generated
            except Exception as error:
                image_failed = True
                summary.errors.append(
                    self.tr('Keypoint segmentation failed for {}: {}').format(image_path, error)
                )

        if image_failed:
            summary.failed += 1
        elif generated:
            summary.processed += 1
        else:
            summary.skipped += 1

    @staticmethod
    def _write_mask(path: Path | None, mask: np.ndarray, overwrite: bool) -> bool:
        if path is None:
            return False
        if path.exists() and not overwrite:
            if path.is_file():
                return False
            raise OSError(f'Output mask path is not a file: {path}')
        path.parent.mkdir(parents=True, exist_ok=True)
        mask = mask.astype(np.uint8, copy=False)
        if overwrite:
            if cv2.imwrite(str(path), mask):
                return True
            raise OSError(f'Could not write output mask: {path}')

        success, encoded_mask = cv2.imencode('.png', mask)
        if not success:
            raise OSError(f'Could not encode output mask: {path}')
        try:
            with path.open('xb') as output_file:
                output_file.write(encoded_mask.tobytes())
        except FileExistsError:
            return False
        return True


class BatchInferencePlugin(Plugin):
    """Adds folder-based ensemble and keypoint inference to the Algorithms menu."""

    _DEFAULT_DEPENDENCY_PLUGIN_FULL_NAME_BY_KEY = {
        'main_window_plugin': 'bsmu.vision.plugins.windows.main.MainWindowPlugin',
        'ensemble_segmenter_plugin': 'bsmu.macula.plugins.ensemble_segmenter.BinaryEnsemblePlugin',
        'keypoint_predictor_plugin': 'bsmu.macula.plugins.keypoint_predictor.KeypointPredictorPlugin',
        'task_storage_plugin': 'bsmu.vision.plugins.storages.task.TaskStoragePlugin',
    }

    def __init__(
            self,
            main_window_plugin: MainWindowPlugin,
            ensemble_segmenter_plugin: BinaryEnsemblePlugin,
            keypoint_predictor_plugin: KeypointPredictorPlugin,
            task_storage_plugin: TaskStoragePlugin,
    ):
        super().__init__()
        self._main_window_plugin = main_window_plugin
        self._ensemble_segmenter_plugin = ensemble_segmenter_plugin
        self._keypoint_predictor_plugin = keypoint_predictor_plugin
        self._task_storage_plugin = task_storage_plugin
        self._main_window: MainWindow | None = None
        self._dialog: BatchInferenceDialog | None = None
        self._active_task: BatchInferenceTask | None = None

    def _enable_gui(self) -> None:
        self._main_window = self._main_window_plugin.main_window
        self._main_window.add_menu_action(
            AlgorithmsMenu, self.tr('Batch Inference'), self._show_dialog
        )

    def _disable(self) -> None:
        if self._dialog is not None:
            self._dialog.set_running(False)
            self._dialog.close()
        self._dialog = None
        self._main_window = None

    def _show_dialog(self) -> None:
        if self._dialog is None:
            self._dialog = BatchInferenceDialog(self._main_window)
            self._dialog.start_requested.connect(self._start_batch)
            if self._keypoint_predictor_plugin.predictor is None:
                self._dialog.kp_seg_checkbox.setEnabled(False)
                self._dialog.kp_seg_checkbox.setToolTip(
                    self.tr('Unavailable because the keypoint model file was not found.')
                )
        self._dialog.show()
        self._dialog.raise_()
        self._dialog.activateWindow()

    def _start_batch(self, options: dict) -> None:
        if self._active_task is not None:
            return
        if not options['include_layer_seg'] and not options['include_kp_seg']:
            QMessageBox.warning(
                self._dialog,
                self.tr('Batch Inference'),
                self.tr('Select at least one mask type.'),
            )
            return
        if options['include_kp_seg'] and self._keypoint_predictor_plugin.predictor is None:
            QMessageBox.warning(
                self._dialog,
                self.tr('Batch Inference'),
                self.tr('Keypoint inference is unavailable because its model was not found.'),
            )
            return

        if not options['input_folder']:
            QMessageBox.warning(
                self._dialog,
                self.tr('Batch Inference'),
                self.tr('Select an input folder.'),
            )
            return
        input_folder = Path(options['input_folder']).expanduser()
        output_folder = Path(options['output_folder']).expanduser()
        if not input_folder.is_dir():
            QMessageBox.warning(
                self._dialog,
                self.tr('Batch Inference'),
                self.tr('Select an existing input folder.'),
            )
            return
        if not options['output_folder']:
            QMessageBox.warning(
                self._dialog,
                self.tr('Batch Inference'),
                self.tr('Select an output folder.'),
            )
            return
        if output_folder.exists() and not output_folder.is_dir():
            QMessageBox.warning(
                self._dialog,
                self.tr('Batch Inference'),
                self.tr('The output path exists and is not a folder.'),
            )
            return

        task = BatchInferenceTask(
            input_folder.resolve(),
            output_folder.resolve(),
            options['recursive'],
            options['include_layer_seg'],
            options['include_kp_seg'],
            options['overwrite'],
            self._ensemble_segmenter_plugin.binary_segmenter,
            self._keypoint_predictor_plugin.predictor,
        )
        task.on_finished = self._on_batch_finished
        self._active_task = task
        self._dialog.set_running(True)
        try:
            self._task_storage_plugin.task_storage.add_item(task)
            ThreadPool.run_async_task(task)
        except Exception as error:
            self._active_task = None
            self._dialog.set_running(False)
            QMessageBox.critical(
                self._dialog,
                self.tr('Batch Inference'),
                self.tr('Could not start batch inference: {}').format(error),
            )

    def _on_batch_finished(self, summary: BatchInferenceSummary) -> None:
        self._active_task = None
        if self._dialog is None:
            return
        self._dialog.set_running(False)

        if summary.fatal_error:
            QMessageBox.warning(
                self._dialog, self.tr('Batch Inference'), summary.fatal_error
            )
            return

        message = self.tr(
            'Processed: {0}\nSkipped: {1}\nFailed: {2}'
        ).format(summary.processed, summary.skipped, summary.failed)
        if summary.errors:
            error_lines = summary.errors[:20]
            if len(summary.errors) > len(error_lines):
                error_lines.append(self.tr('…and {0} more').format(len(summary.errors) - 20))
            message += '\n\n' + '\n'.join(error_lines)
        box = QMessageBox(self._dialog)
        box.setIcon(
            QMessageBox.Icon.Warning if summary.failed else QMessageBox.Icon.Information
        )
        box.setWindowTitle(self.tr('Batch Inference'))
        box.setText(message)
        box.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        box.exec()


__all__ = ['BatchInferenceDialog', 'BatchInferencePlugin', 'BatchInferenceTask']
