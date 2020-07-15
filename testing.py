import numpy as np
import pandas as pd
import scipy.signal as sp
import xml.etree.ElementTree as ET
import math
from scipy.ndimage import filters
from scipy.signal import butter, lfilter, freqz, filtfilt
# from suite2p.extraction import dcnv
import matplotlib.pyplot as plt
# from calimag.Parse import MicroscopeParsing

# Saved and commented out previously used file locations

# serialdata1 = pd.read_csv('F:\\20190710_Max\\M002\\20190710_M002_serialdata_Cleanedup1-27.csv')
# serialdata2 = pd.read_csv('F:\\20190710_Max\\M002\\20190710_M002_serialdata_Cleanedup28-55.csv')
serialdata1 = pd.read_csv(
    "20200124_m164_cleanpart1.csv"
)

serialdata2 = pd.read_csv(
    "20200124_m164_cleanpart2.csv"
)
serialdata3 = pd.read_csv(
    "20200124_m164_cleanpart3.csv"
)

# Is this column ever used as a time index? Check...

pd.to_datetime(serialdata1["Trial Time"], unit="ms")
pd.to_datetime(serialdata2["Trial Time"], unit="ms")
pd.to_datetime(serialdata3["Trial Time"], unit="ms")

# Concatenate serialdata files

serialdata = [serialdata1, serialdata2, serialdata3]
serialdata = pd.concat(serialdata)

# Make column counting up from 0 in milliseconds, since serial data is sampled at 1000Hz, set as datetimeindex for later downsampling.

# Set millisecond clock from serial data as time index
serialdata["absolute time"] = range(len(serialdata))
serialdata["absolute time"] = serialdata["absolute time"] + 1
serialdata["absolute time"] = pd.to_datetime(serialdata["absolute time"], unit="ms")
serialdata.set_index(
    pd.DatetimeIndex(serialdata["absolute time"]), inplace=True, drop=True
)
serialdata


# function for low pass filtering sniff and wheel signal with Butterworth filter.
# From: https://gist.github.com/junzis/e06eca03747fc194e322


def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


# Smooth sniff signal

# Possibly unnecessary conversion step (Check)
sniff = serialdata["Sniff"].to_numpy()
plotsniff = sniff - 200

# Filter requirements.
order = 4
fs = 1000  # sample rate, Hz
lowcut = 2
highcut = 20

# Filter the data, and plot both the original and filtered signals.
buttersniff = butter_bandpass_filter(sniff, lowcut, highcut, fs, order)

# possibly unnecessary conversion step, invert function if troughs need to be converted to peaks

# forS013
# buttersniff = buttersniff * -1

# find peaks and plot to check results
# scroll bar code copied from stackoverflow user "ImportanceOfBeingErnest" here: https://stackoverflow.com/a/54155302

# good for, M003, M002:
print("hello")
peaks, _ = sp.find_peaks_cwt(buttersniff, np.arange(1,24))
print("goodbye")
# S013:
# peaks, _ = sp.find_peaks(buttersniff, prominence=20, distance=40, width=20)


import sys
import matplotlib

# Make sure that we are using QT5
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import numpy as np


class ScrollableWindow(QtWidgets.QMainWindow):
    def __init__(self, fig, ax, step=0.1):
        plt.close("all")
        if not QtWidgets.QApplication.instance():
            self.app = QtWidgets.QApplication(sys.argv)
        else:
            self.app = QtWidgets.QApplication.instance()

        QtWidgets.QMainWindow.__init__(self)
        self.widget = QtWidgets.QWidget()
        self.setCentralWidget(self.widget)
        self.widget.setLayout(QtWidgets.QVBoxLayout())
        self.widget.layout().setContentsMargins(0, 0, 0, 0)
        self.widget.layout().setSpacing(0)

        self.fig = fig
        self.ax = ax
        self.canvas = FigureCanvas(self.fig)
        self.canvas.draw()
        self.scroll = QtWidgets.QScrollBar(QtCore.Qt.Horizontal)
        self.step = step
        self.setupSlider()
        self.nav = NavigationToolbar(self.canvas, self.widget)
        self.widget.layout().addWidget(self.nav)
        self.widget.layout().addWidget(self.canvas)
        self.widget.layout().addWidget(self.scroll)

        self.canvas.draw()
        self.show()
        self.app.exec_()

    def setupSlider(self):
        self.lims = np.array(self.ax.get_xlim())
        self.scroll.setPageStep(self.step * 100)
        self.scroll.actionTriggered.connect(self.update)
        self.update()

    def update(self, evt=None):
        r = self.scroll.value() / ((1 + self.step) * 100)
        l1 = self.lims[0] + r * np.diff(self.lims)
        l2 = l1 + np.diff(self.lims) * self.step
        self.ax.set_xlim(l1, l2)
        print(self.scroll.value(), l1, l2)
        self.fig.canvas.draw_idle()


# create a figure and some subplots
fig, ax = plt.subplots()
t = peaks
x = buttersniff[peaks]
ax.scatter(t, x, marker="|", s=2000, c="y", linewidth=2)
ax.plot(buttersniff, linewidth=1)
ax.plot(plotsniff, linewidth=0.5, color='gray')

# pass the figure to the custom window
a = ScrollableWindow(fig, ax)

# Replace the raw sniff data with a column where '1' marks every sniff time

sniffevents = np.zeros((expduration,), dtype=int)
sniffevents[peaks] = 1

serialdata["Sniff"] = sniffevents
