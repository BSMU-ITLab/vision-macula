import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QStandardItemModel, QStandardItem
from PySide6.QtSql import QSqlQueryModel, QSqlDatabase
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTableView, QApplication
import pandas as pd

# 👇 Пример данных прямо в коде
injections_data = [
    {"eye_id": 1, "lutein_therapy": "Лютеин Форте"},
    {"eye_id": 2, "lutein_therapy": "Лютеин Форте"},
    {"eye_id": 3, "lutein_therapy": "Лютеин Комплекс"},
    {"eye_id": 4, "lutein_therapy": "Лютеин Форте"},
    {"eye_id": 5, "lutein_therapy": "Лютеин Комплекс"},
    {"eye_id": 6, "lutein_therapy": "Нет данных"},
    {"eye_id": 7, "lutein_therapy": "Лютеин Комплекс"},
    {"eye_id": 8, "lutein_therapy": "Нет данных"},
]

class LuteinSummary(QWidget):
    def __init__(self, summary_df):
        super().__init__()
        self.setWindowTitle("📊 Lutein Therapy Summary")
        self.resize(500, 400)
        self.setup_ui(summary_df)

    def setup_ui(self, df):
        layout = QVBoxLayout(self)

        title = QLabel("📊 Количество записей по lutein_therapy")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        table = QTableView()
        model = QStandardItemModel()
        model.setHorizontalHeaderLabels(["Lutein Type", "Count"])

        for _, row in df.iterrows():
            items = [
                QStandardItem(str(row["lutein_therapy"])),
                QStandardItem(str(row["count"]))
            ]
            for item in items:
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            model.appendRow(items)

        table.setModel(model)
        table.setAlternatingRowColors(True)
        table.setStyleSheet("""
            QTableView {
                font-size: 12pt;
                gridline-color: #ccc;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                padding: 4px;
                font-weight: bold;
            }
        """)
        table.resizeColumnsToContents()
        layout.addWidget(table)

def main():
    app = QApplication(sys.argv)

    # 📊 Группировка данных
    df = pd.DataFrame(injections_data)
    summary_df = df.groupby("lutein_therapy").size().reset_index(name="count")
    summary_df = summary_df.sort_values("count", ascending=False)

    viewer = LuteinSummary(summary_df)
    viewer.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()