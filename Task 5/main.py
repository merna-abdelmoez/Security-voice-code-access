import os
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.uic import loadUiType
import numpy as np
from scipy.io.wavfile import read
from fastdtw import fastdtw
from python_speech_features import mfcc
from scipy.io.wavfile import write
import sounddevice as sd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import librosa
from pyqtgraph import PlotWidget
from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy import signal
from PyQt5.QtCore import QTimer, QCoreApplication
import time


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
ui, _ = loadUiType('ui5.ui')


class MainApp(QMainWindow, ui):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.record_btn = self.findChild(QPushButton, 'record_btn')
        self.record_btn.clicked.connect(self.record)
        self.pattern_btn = self.findChild(QPushButton, 'pattern_btn')
        self.pattern_btn.clicked.connect(self.record_pattern)
        self.mode_1_radio_btn = self.findChild(QRadioButton, 'mode_1')
        self.mode_2_radio_btn = self.findChild(QRadioButton, 'mode_2')
        self.weights = pd.read_csv('weights.csv')
        self.biases = pd.read_csv('biases.csv')
        self.weights_sent = pd.read_csv('weights_sent.csv')
        self.biases_sent = pd.read_csv('biases_sent.csv')
        self.gen_weights = [self.weights_sent , self.weights]
        self.gen_biases = [self.biases_sent , self.biases]
        self.model_names = ["amir", "emad", "farah","khaled", "merna", "Mohamed", "nabil", "Osama"]
        self.model_names_sent = [ "GMA", "OMD", "UTG"]
        self.gen_names = [ self.model_names_sent , self.model_names]
        self.let_label = self.findChild(QLabel, 'let_perc_label')
        self.out_put_label = self.findChild(QLabel, 'out_put_label')
        self.grant_label = self.findChild(QLabel, 'grant_perc_label')
        self.open_label = self.findChild(QLabel, 'open_perc_label')
        #self.def_models()
        self.mode1_btn= self.findChild(QRadioButton, 'mode_1')
        self.mode2_btn = self.findChild(QRadioButton, 'mode_2')
        self.mode1_btn.setChecked(True)
        # spectrogram
        spectro = self.findChild(QWidget, 'spectro')
        widget_layout = QVBoxLayout(spectro)
        widget_layout.addWidget(self.spectro)
        widget_layout.setContentsMargins(0, 0, 0, 0)
        # checkboxes
        self.checkboxes_arr = []
        for i in range(1, 9):
            checkbox = self.findChild(QCheckBox, f'checkBox_{i}')
            self.checkboxes_arr.append(checkbox)
        #labels beta3et el names perc
        self.labels_perc = []
        for i in range(1, 9):
            label = self.findChild(QLabel, f'label_{i}')
            self.labels_perc.append(label)
        # labels
        self.labels_arr = []
        for i in range(1, 9):
            label_names = self.findChild(QLabel, f'label_names_{i}')
            self.labels_arr.append(label_names)
        # self.highest_ratios = {}
        for checkbox in self.checkboxes_arr:
            checkbox.stateChanged.connect(self.check_selected_user)
        # Progress bar
        self.progressBar = self.findChild(QProgressBar, 'progressBar')
        self.progressBar.setValue(0)
        self.record_timer = QTimer(self)
        self.record_timer.timeout.connect(self.update_progress_bar)


    def update_progress_bar(self):
        current_value = self.progressBar.value()
        new_value = min(current_value + 1, 100)  # Increment by 1, limited to 100
        self.progressBar.setValue(new_value)
        QCoreApplication.processEvents()


    def set_labels_ratios(self):
        if isinstance(self.ratios, dict):
            for i, (key, val) in enumerate(self.ratios.items()):
                val = float(val)
                label_name = f"label_p_{i + 1}"
                try:
                    label = getattr(self, label_name)
                    label.setText(f"{val:.2f}%")
                except Exception as e:
                    print(f"Error updating label {label_name}: {e}")

    def set_access_status(self, user, ratio, granted_color="green", denied_color="red", font_size=30):
        text = f"Access Granted " if ratio > 15 else f"Access Denied "
        color = granted_color if ratio > 15 else denied_color

        self.out_put_label.setText(text)
        self.out_put_label.setStyleSheet(f"color: {color}; font-size: {font_size}px;")

    def check_selected_user(self):
        checked_users = [label_names.text() for checkbox, label_names in zip(self.checkboxes_arr, self.labels_arr) if
                         checkbox.isChecked()]
        if self.highest_user in checked_users:
            self.set_access_status(self.highest_user, self.highest_ratio)
        else:
            for checked_user in checked_users:
                self.set_access_status(checked_user, 0, denied_color="red", font_size=30)

    def create_spectrogram(self, signal, widget):
        if hasattr(widget, 'canvas') and widget.canvas:
            widget.canvas.setParent(None)
            widget.fig.clear()
        fig, ax = plt.subplots()
        spec = ax.specgram(signal, Fs=self.fs, cmap='plasma')
        # Add color bar
        cbar = fig.colorbar(spec[3], ax=ax)
        cbar.set_label('Intensity (dB)')
        widget.canvas = FigureCanvas(fig)
        widget_layout = widget.layout()
        widget_layout.addWidget(widget.canvas)
        widget.fig = fig
        widget.canvas.draw()


    def extract_mfccs(self,file_path):
        y, sr = librosa.load(file_path)
        mfccs = librosa.feature.mfcc(y=y, sr=sr)
        mfccs = mfccs.T
        delta_mfccs = librosa.feature.delta(mfccs)
        delta_mfccs = delta_mfccs

        # (216) rows = frames ,  and (26) columns = features
        features = np.hstack([mfccs, delta_mfccs])
        return pd.DataFrame(features)

    def extract_chroma(self, file_path):
        y, sr = librosa.load(file_path)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma = chroma.T
        return pd.DataFrame(chroma)

    def extract_zero_crossings(self, file_path):
        y, sr = librosa.load(file_path)
        zero_crossings = librosa.feature.zero_crossing_rate(y)
        zero_crossings = zero_crossings.T
        return pd.DataFrame(zero_crossings)

    def extract_spectral_contrast(self, file_path):
        y, sr = librosa.load(file_path)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_contrast = spectral_contrast.T
        return pd.DataFrame(spectral_contrast)

    def process_user_folder(self ):
        current_folder = os.getcwd()
        folder_path = os.path.join(current_folder)

        mfccs_list = pd.DataFrame()
        chroma_list = pd.DataFrame()
        zero_crossings_list = pd.DataFrame()
        spectral_contrast_list = pd.DataFrame()

        for file_name in os.listdir(folder_path):
            if file_name == "recording.wav":
                file_path = os.path.join(folder_path, file_name)
                print(file_path)
                mfccs_list = self.extract_mfccs(file_path)
                chroma_list = self.extract_chroma(file_path)
                zero_crossings_list = self.extract_zero_crossings(file_path)
                spectral_contrast_list = self.extract_spectral_contrast(file_path)

        # Data set without the pitch
        combined_features = pd.concat([mfccs_list, chroma_list, zero_crossings_list, spectral_contrast_list], axis=1)
        return combined_features

    def predict(self, features , mode = 1):
        self.highest_ratios = {}
        features = np.array(features)
        weight = np.array(self.gen_weights[mode])
        bias = np.array (self.gen_biases[mode])
        # Calculate the dot product of features and weights, and add the bias
        logits = np.dot(features, weight.T) + bias.T
        # Apply the sigmoid function to obtain probabilities
        probabilities = 1 / (1 + np.exp(-logits))
        # Threshold probabilities to get binary predictions
        predictions = (probabilities >= 0.5).astype(int)
        print(predictions.shape)
        predictions = predictions.T
        for i , name in zip(range(len(predictions)) , self.gen_names[mode]):
            pred  = predictions[i][:]
            print(pred)
            one_count = (pred == 1).sum()
            print(f"{name} : {one_count}")
            ratio = (one_count / pred.size) * 100

            if name in self.highest_ratios:
                previous_ratio = self.highest_ratios[name]
                if ratio > previous_ratio:
                    self.highest_ratios[name] = ratio
            else:
                self.highest_ratios[name] = ratio

            self.highest_user = max(self.highest_ratios, key=self.highest_ratios.get)
            self.highest_ratio = self.highest_ratios[self.highest_user]
            print(f"{name} : {ratio}")
            print(self.highest_ratios)

        return self.highest_ratios ,  self.highest_user

    def record(self):
        try:
            # Sample rate (you can adjust it as needed)
            self.progressBar.setValue(0)
            fs = 44100
            duration = 3
            recording = sd.rec(duration * fs, samplerate=fs, channels=1, dtype=np.int16)
            self.record_timer.start(30)

            start_time = time.time()  # Record the start time
            while time.time() - start_time < duration:
                QCoreApplication.processEvents()
            sd.wait()
            # Save audio to file
            write('recording.wav', fs, recording)
            audio_fs, audio = read('recording.wav')
            self.fs=audio_fs
            frequencies, times, spectrogram = signal.spectrogram(audio, audio_fs)
            self.create_spectrogram(audio, self.spectro)
            self.record_timer.stop()
            self.progressBar.setValue(100)
            # Calculate the percentage similarity for each reference
            perc_sim_let = self.calc_perc_sim('ref_let.wav')
            perc_sim_grant = self.calc_perc_sim('ref_grant.wav')
            perc_sim_open = self.calc_perc_sim('ref_open.wav')

            self.let_label.setText(f"{perc_sim_let:.2%}")
            self.grant_label.setText(f"{perc_sim_grant:.2%}")
            self.open_label.setText(f"{perc_sim_open:.2%}")

            print(f"Percentage Similarity - Let Me In: {perc_sim_let:.2%}")
            print(f"Percentage Similarity - Grant Me Access: {perc_sim_grant:.2%}")
            print(f"Percentage Similarity - Open the Door: {perc_sim_open:.2%}")
            access_granted = False
            if (max([perc_sim_open, perc_sim_grant, perc_sim_let]) >= 0.6 and self.mode1_btn.isChecked()) :
                access_granted = True
                self.out_put_label.setText("Access Granted")
                self.out_put_label.setStyleSheet("color: green; font-size: 30px;")  # Set the color and font size

            else:
                access_granted = False
                if(self.mode1_btn.isChecked()):
                    self.out_put_label.setText("Access Denied")
                    self.out_put_label.setStyleSheet("color: red; font-size: 30px;")  # Set the color and font size
            record_features = self.process_user_folder()
            print(record_features)
            if self.mode_2_radio_btn.isChecked():
                # to check the selected checkboxes
                for checkbox in self.checkboxes_arr:
                    checkbox.stateChanged.connect(self.check_selected_user)

                self.ratios , max_name = self.predict(record_features ,1)
                self.set_labels_ratios()
                self.check_selected_user()

        except Exception as e:
            print("An error occurred:")
            print(e)

    def calc_perc_sim(self, ref_name):
        # Calculate the dynamic time warping distance
        distance = self.calc_dist(ref_name)

        # Normalize the distance to a percentage similarity (closer to 0 is more similar)
        max_distance = 50000
        perc_similarity = 1 - (distance / max_distance)
        bias = 0.5  # el 40 da bias mmkn yetshal
        return perc_similarity + bias

    def calc_dist(self, ref_name):
        # Read the reference audio
        reference_audio_fs, reference_audio = read(ref_name)
        # Record audio
        recording_fs, recording = read('recording.wav')
        # Normalize the signals
        reference_audio = reference_audio / np.max(np.abs(reference_audio))
        recording = recording / np.max(np.abs(recording))
        # Extract features (MFCCs in this example)
        features_ref = self.extract_features(reference_audio, reference_audio_fs)
        features_rec = self.extract_features(recording, recording_fs)
        # Calculate the dynamic time warping distance
        distance, _ = fastdtw(features_ref, features_rec)
        print(ref_name, distance)
        return distance

    def extract_features(self, signal_var, fs):
        # You can use different feature extraction techniques here
        # For example, use MFCCs
        signal_var = signal_var / np.max(np.abs(signal_var))
        mfcc_features = mfcc(signal_var, fs)
        return mfcc_features
    def record_pattern(self):
        # Sample rate (you can adjust it as needed)
        duration = 3  # Recording duration in seconds
        fs = 44100
        # Record audio
        recording = sd.rec(duration * fs, samplerate=fs, channels=1, dtype=np.int16)
        sd.wait()
        # Save audio to file (overwrite the existing my_grant.wav)
        write('ref_let.wav', fs, recording)
        print("Pattern recorded and saved as ref_grant.wav")


def main():
    app = QApplication(sys.argv)
    QApplication.processEvents()
    window = MainApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()