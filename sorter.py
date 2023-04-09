import sys
import os
import shutil
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QFormLayout, QMessageBox, QDoubleSpinBox, QComboBox, QCheckBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QColor
from ultralytics import YOLO

class VideoSorter(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("MonkeySorter")
        self.resize(800, 300)

        script_directory = os.path.dirname(os.path.abspath(__file__))
        self.model_directory = os.path.join(script_directory, "models")

        layout = QVBoxLayout()

        # Dark Mode
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
        dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, Qt.black)
        self.setPalette(dark_palette)


        # Parameters
        params_layout = QFormLayout()
        
        self.model_box = QComboBox()
        self.model_box.addItem("n")
        self.model_box.addItem("s")
        self.model_box.addItem("m")
        self.model_box.addItem("l")
        self.model_box.setStyleSheet("color: black")
        params_layout.addRow("Model:", self.model_box)

        self.conf_box = QDoubleSpinBox()
        self.conf_box.setRange(0, 1)
        self.conf_box.setSingleStep(0.01)
        self.conf_box.setValue(0.5)
        self.conf_box.setStyleSheet("color: black")
        params_layout.addRow("Confidence:", self.conf_box)

        self.iou_box = QDoubleSpinBox()
        self.iou_box.setRange(0, 1)
        self.iou_box.setSingleStep(0.01)
        self.iou_box.setValue(0.7)
        self.iou_box.setStyleSheet("color: black")
        params_layout.addRow("IOU:", self.iou_box)

        self.device_box = QComboBox()
        self.device_box.addItem("None")
        self.device_box.addItem("cpu")
        self.device_box.addItem("0")
        self.device_box.setStyleSheet("color: black")
        params_layout.addRow("Device:", self.device_box)

        self.show_box = QCheckBox()
        self.show_box.setStyleSheet("color: black")
        params_layout.addRow("Show:", self.show_box)

        self.save_box = QCheckBox()
        self.save_box.setStyleSheet("color: black")
        params_layout.addRow("Save:", self.save_box)

        self.save_txt_box = QCheckBox()
        self.save_txt_box.setStyleSheet("color: black")
        params_layout.addRow("Save TXT:", self.save_txt_box)

        layout.addLayout(params_layout)
        
        select_video_folder_button = QPushButton("Select Videos Folder")
        select_video_folder_button.clicked.connect(self.select_video_folder)
        select_video_folder_button.setStyleSheet("background-color: #4A4A4A; color: white")
        layout.addWidget(select_video_folder_button)

        run_sorting_button = QPushButton("Run Sorting")
        run_sorting_button.clicked.connect(self.run_sorting)
        run_sorting_button.setStyleSheet("background-color: #4A4A4A; color: white")
        layout.addWidget(run_sorting_button)

        self.setLayout(layout)


    def select_video_folder(self):
        self.video_folder = QFileDialog.getExistingDirectory()

    def process_videos(self, video_folder):

        params = {
            "model": self.model_box.currentText(),
            "conf": self.conf_box.value(),
            "iou": self.iou_box.value(),
            "device": self.device_box.currentText(),
            "show": self.show_box.isChecked(),
            "save": self.save_box.isChecked(),
            "save_txt": self.save_txt_box.isChecked()
        }

        if params["device"] == "None":
            params["device"] = None

        model_name = f"best_{params['model']}.pt"
        self.model = YOLO(os.path.join(self.model_directory, model_name))

        detection_folder = os.path.join(video_folder, "detections")
        no_detection_folder = os.path.join(video_folder, "no_detections")

        os.makedirs(detection_folder, exist_ok=True)
        os.makedirs(no_detection_folder, exist_ok=True)

        for file in os.listdir(video_folder):
            if file.lower().endswith((".mp4", ".avi", ".mov")):
                video_path = os.path.join(video_folder, file)
                results = self.model.predict(video_path, **params)

                has_detections = False
                for result in results:
                    if len(result.boxes.boxes) > 0:
                        has_detections = True
                        break

                output_folder = detection_folder if has_detections else no_detection_folder
                output_path = os.path.join(output_folder, file)

                shutil.copy2(video_path, output_path)
                os.remove(video_path)

    def run_sorting(self):
        if self.video_folder is None:
            QMessageBox.warning(self, "Error", "Please selelct videos directory.")
            return

        self.process_videos(self.video_folder)
        QMessageBox.about(self, "Sorting completed", "Videos folder has been successfully sorted.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoSorter()
    window.show()
    sys.exit(app.exec_())
