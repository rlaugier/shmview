#!/usr/bin/env python3


''' --------------------------------------------------------------------------
         ____  _   _ __  __    __     _____ _______        _______ ____  
        / ___|| | | |  \/  |   \ \   / /_ _| ____\ \      / / ____|  _ \ 
        \___ \| |_| | |\/| |____\ \ / / | ||  _|  \ \ /\ / /|  _| | |_) |
         ___) |  _  | |  | |_____\ V /  | || |___  \ V  V / | |___|  _ < 
        |____/|_| |_|_|  |_|      \_/  |___|_____|  \_/\_/  |_____|_| \_\


- 20170730: Shared memory viewer.
- 20250228: modified during the AIT of Asgard Heimdallr & Baldr, in particular
to properly display the live image that is part of a circular buffer.
-------------------------------------------------------------------------- '''
from xaosim.QtMain import QtMain, QApplication
from PyQt5 import QtCore, QtGui, uic
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QLabel, QMainWindow, QFileDialog
from PyQt5.QtGui import QImage

import threading

import pyqtgraph as pg
import sys
import numpy as np
from xaosim.shmlib import shm
import matplotlib.cm as cm
from matplotlib import color_sequences as cs
import astropy.io.fits as pf
import os
colortraces = [cm.Blues,
              cm.YlOrBr, # This in order to use oranges for the brouwns
              cm.Greens,
              cm.RdPu,
              cm.Purples,
              cm.Oranges,
              cm.Reds,
              cm.Greys,
              cm.YlOrRd,
              cm.GnBu]

# Some colormaps chosen to mirror the default (Cx) series of colors
colortraces_0 = [cm.Blues,
              cm.YlOrBr, # This in order to use oranges for the brouwns
              cm.Greens,
              cm.Reds,
              cm.Purples,
              cm.Oranges,
              cm.RdPu,
              cm.Greys,
              cm.YlOrRd,
              cm.GnBu]

# =====================================================================
home = os.getenv('HOME')
conf_dir = os.path.dirname(__file__) + "/" # "./" # home+'/.config/xaosim/'

# =====================================================================
# =====================================================================
myqt = 0  # to have myqt as a global variable

def main():
    global myqt
    myqt = QtMain()
    gui = MyWindow()

    proxy = pg.SignalProxy(gui.gView_shm.scene().sigMouseMoved,
                           rateLimit=60, slot=gui.mouseMoved)
    myqt.mainloop()
    myqt.gui_quit()
    sys.exit()


# =====================================================================
#                               Tools
# =====================================================================
def bound(val, low, high):
    return max(low, min(high, val))

def arr2im(arr, vmin=False, vmax=False, pwr=1.0, cmap=None, gamma=1.0):
    ''' --------------------------------------------------------
    convert 2D numpy array into image for display

    limits dynamic range, power coefficient and applies colormap
    -------------------------------------------------------- '''
    arr2 = arr.astype('float')  # local array is modified
    
    mmin = np.percentile(arr2, 5) if vmin is False else vmin
    mmax = np.percentile(arr2, 99.9) if vmax is False else vmax
    mycmap = cm.magma if cmap is None else cmap

    arr2 -= mmin
    if mmax != mmin:
        arr2 /= (mmax-mmin)
    arr2 = arr2**pwr

    res = mycmap(arr2)
    res[:, :, 3] = gamma
    return(res)

# =====================================================================
#                          Main GUI object
# =====================================================================
args = sys.argv[1:]


class MyWindow(QMainWindow):
    ''' ------------------------------------------------------
    This is the meat of the program: the class that drives
    the GUI.
    ------------------------------------------------------ '''
    def __init__(self):
        global index
        self.mySHM = None  # handle for mmapped SHM file
        self.vmin = False
        self.vmax = False
        self.pwr = 1.0
        self.mycmap = cm.gray
        self.required_slice = -1
        self.averaging = False

        self.pyi, self.pxi = 0, 0
        self.pxval = 0.0

        super(MyWindow, self).__init__()
        if not os.path.exists(conf_dir + 'shmimview.ui'):
            uic.loadUi('shmimview.ui', self)
        else:
            uic.loadUi(conf_dir + 'shmimview.ui', self)

        # ==============================================
        # prepare the display
        # ==============================================

        # self.gView_shm.hideAxis('left')
        # self.gView_shm.hideAxis('bottom')
        # self.imv_data = pg.PlotWidget()
        # self.imv_data = pg.ImageItem()
        self.overlay = pg.GraphItem()
        self.gView_shm.setBackground("w")
        # self.gView_shm.addItem(self.imv_data)

        # cross-hair
        self.vline = pg.InfiniteLine(angle=90, movable=False)
        self.hline = pg.InfiniteLine(angle=0, movable=False)
        self.gView_shm.addItem(self.vline, ignoreBounds=True)
        self.gView_shm.addItem(self.hline, ignoreBounds=True)

        # ==============================================
        #             GUI widget actions
        # ==============================================
        self.dspB_disp_min.valueChanged[float].connect(self.update_vmin)
        self.dspB_disp_max.valueChanged[float].connect(self.update_vmax)
        self.chB_min.stateChanged[int].connect(self.update_vmin_apply)
        self.chB_max.stateChanged[int].connect(self.update_vmax_apply)
        self.chB_nonlinear.stateChanged[int].connect(self.update_nonlinear)
        self.chB_dark_sub.stateChanged[int].connect(self.update_dark_state)
        self.chB_average.stateChanged[int].connect(self.update_average_state)
        self.cmB_cbar.addItems(
            ['magma', 'gray', 'hot', 'cool', 'bone', 'jet', 'viridis'])
        self.cmB_cbar.activated[str].connect(self.update_cbar)
        self.cmB_cbar.setCurrentIndex(0)
        self.update_cbar()

        # ==============================================
        #             top-menu actions
        # ==============================================
        self.actionQuit.triggered.connect(sys.exit)
        self.actionQuit.setShortcut('Ctrl+Q')

        self.actionOpen.triggered.connect(self.load_shm)
        self.actionSaveAs.triggered.connect(self.save_as_fits)
        self.actionSaveAs.setShortcut('Ctrl+S')

        self.actionLoadDark.triggered.connect(self.load_shm_drk)
        self.setMinimumSize(600, 400)

        if sys.argv[1:] != []:
            target = str(sys.argv[1])
            if os.path.isfile(target):
                self.mySHM = shm(target)
                self.array_title.setText(target)
                self.live_counter = -1
                self.naxis = self.mySHM.mtdata['naxis']
                if self.mySHM.empty:
                    self.mySHM = None
                else:
                    self.auto_resize_window()
            else:
                print("No file loaded")

            if len(sys.argv[1:]) > 1:
                try:
                    self.required_slice = int(sys.argv[2])
                except ValueError:
                    print("Not a valid slice")
                    self.required_slice = -1
                    self.myslice = 0
                self.myslice = min(self.myslice,
                                   self.mySHM.mtdata['size'][-1]-1)
            else:
                self.required_slice = -1
                self.myslice = 0

        self.myDRK = None
        # ==============================================
        self.show()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.refresh_all)
        self.timer.start(100)

    # =========================================================
    def closeEvent(self, event):
        sys.exit()

    # =========================================================
    def mouseMoved(self, evt):
        pass

    # =========================================================
    def update_cbar(self):
        cbar = str(self.cmB_cbar.currentText()).lower()
        try:
            exec('self.mycmap = cm.%s' % (cbar,))
        except AttributeError:
            self.mycmap = cm.jet

    # =========================================================
    def update_vmin_apply(self):
        if not self.chB_min.isChecked():
            self.vmin = False
        else:
            self.vmin = self.dspB_disp_min.value()

    # =========================================================
    def update_vmin(self):
        # self.vmin = False
        self.chB_min.setCheckState(True)
        # if self.chB_min.isChecked():
        self.vmin = self.dspB_disp_min.value()
    
    # =========================================================
    def update_vmax_apply(self):
        if not self.chB_max.isChecked():
            self.vmax = False
        else:
            self.vmax = self.dspB_disp_max.value()

    # =========================================================
    def update_vmax(self):
        # self.vmax = False
        self.chB_max.setCheckState(True)
        # if self.chB_max.isChecked():
        self.vmax = self.dspB_disp_max.value()

    # =========================================================
    def update_nonlinear(self):
        self.pwr = 1.0
        if self.chB_nonlinear.isChecked():
            self.pwr = 0.25

    # =========================================================
    def update_dark_state(self):
        if self.live_counter == self.mySHM.get_counter():
            self.live_counter -= 1
        pass

    # =========================================================
    def update_average_state(self):
        if self.chB_average.isChecked():
            self.averaging = True
        else:
            self.averaging = False
        pass

    # =========================================================
    def refresh_img(self):
        self.imv_data.setImage(arr2im(self.data_img.T,
                                      vmin=self.vmin, vmax=self.vmax,
                                      pwr=self.pwr,
                                      cmap=self.mycmap), border=2)

        self.pxval = self.data_img[self.pyi, self.pxi]
    def refresh_plot(self):
        x_values = np.linspace(0., 1.,  self.data_img.shape[0])
        self.gView_shm.clear()
        for i, adata in enumerate(self.data_img.T):
            self.gView_shm.plot(x_values, adata[:], pen=self.pens[i])
        
    # =========================================================
    def refresh_stats(self, add_msg=None):

        pt_levels = [0, 5, 10, 20, 50, 75, 90, 95, 99, 100]
        pt_values = np.percentile(self.data_img, pt_levels)

        msg = "<pre>\n"

        self.mySHM.read_keywords()
        sz = self.mySHM.mtdata["size"]
        if self.naxis == 2:
            msg += f"imsize = {sz[1]:3d} x {sz[2]:3d}"
        else:
            msg += f"imsize = {sz[0]:3d} x {sz[1]:3d} x {sz[2]:3d}"

        msg += f"\n\ncross-hair ({self.pxi:3d}, {self.pyi:3d})\n"
        msg += f"value = {self.pxval:8.1f}\n\n"

        for ii, kwd in enumerate(self.mySHM.kwds):
            msg += f"{kwd['name']:10s} : {kwd['value']:10s} \n"

        for i, ptile in enumerate(pt_levels):
            msg += f"p-tile {ptile:3d} = {pt_values[i]:8.2f}\n"

        msg += f"img count  = {self.mySHM.get_counter():8d}\n"

        if add_msg is not None:
            msg += f"{add_msg}\n"
        msg += "</pre>"
        self.lbl_stats.setText(msg)

    # =========================================================
    def auto_resize_window(self):
        imsize = self.mySHM.get_data().shape
        self.imsize = imsize
        imratio = 1.0
        if self.naxis == 2:
            imratio = imsize[1] / imsize[0]

        vsize = self.gView_shm.geometry()  # view size

        hratio = vsize.width() / imsize[1]
        vratio = vsize.height() / imsize[0]

        if hratio > vratio:  # better horizontal "resolution"
            self.resize(vsize.width() + 200, int(vsize.width() / imratio) + 46)
        else:
            self.resize(int(vsize.height() * imratio) + 200, vsize.height() + 46)
        return

    # =========================================================
    def save_as_fits(self):
        fname, select_filter = QFileDialog.getSaveFileName(
            self, "Save As", f"{home}", "Images (*.fits)")
        if fname != '':
            fname_parts = fname.split(".")
            if len(fname_parts) == 2:
                dark_fname = fname_parts[0] + "_dark.fits"
            pf.writeto(fname, self.mySHM.get_data(), overwrite=True)
            pf.writeto(dark_fname, self.myDRK, overwrite=True)

    # =========================================================
    def load_shm(self):
        fname = QFileDialog.getOpenFileName(
            self, 'Load SHM file', '/dev/shm/*.im.shm')[0]

        if fname != '':  # a new shm file was indeed selected
            if self.mySHM is not None:
                self.mySHM.close(erase_file=False)
                self.mySHM = None
        else:  # false alert: don't change anything
            return

        self.array_title.setText(fname)
        self.mySHM = shm(str(fname))
        self.live_counter = -1
        self.naxis = self.mySHM.mtdata['naxis']
        self.pens = []
        for acolor in cs["tab10"]:
            mycolor = np.array(acolor) * 255
            self.pens.append(pg.mkPen(mycolor, width=3))
        # self.pens = [pg.mkPen(color=acol) for acol in cs["tab10"]]

        self.myslice = 0
        if self.naxis == 3:
            self.myDRK = np.zeros_like(self.mySHM.get_data()[0]).astype(float)
        else:
            self.myDRK = np.zeros_like(self.mySHM.get_data()).astype(float)

        self.auto_resize_window()

    # =========================================================
    def load_shm_drk(self):
        fname = QFileDialog.getOpenFileName(
            self, 'Load SHM file for dark', '/dev/shm/*.im.shm')[0]

        if fname != '':  # a new shm file was indeed selected
            try:
                temp = shm(str(fname))
                if self.naxis == 3:
                    self.myDRK = temp.get_data().astype(float).mean(0)
                else:
                    self.myDRK = temp.get_data()
                temp = None
            except:
                self.myDRK = np.zeros_like(self.mySHM.get_data())
        else:
            print("No new dark was loaded")
            return

    # =========================================================
    def refresh_all(self):
        ''' ----------------------------------------------------------
        Refresh the display
        ---------------------------------------------------------- '''
        self.test = 0
        add_msg = None
        if self.mySHM is not None:
            mycntr = self.mySHM.get_counter()
            if mycntr == 0:  # simulation has been stopped. Start anew!
                self.live_counter = -1
            if self.live_counter < mycntr:
                if self.naxis == 2:
                    self.data_img = self.mySHM.get_data(False, True)
                else:
                    if self.required_slice != -1:
                        self.data_img = self.mySHM.get_data(False, True)[
                            self.myslice]
                    else:
                        if not self.averaging:
                            # self.data_img = self.mySHM.get_data(False, True)[
                            #     (mycntr - 1) % self.mySHM.mtdata["size"][0]]
                            self.data_img = self.mySHM.get_latest_data_slice()
                        else:
                            self.data_img = self.mySHM.get_data(False, True).mean(0)

                self.data_img = self.data_img.astype(float)
                self.live_counter = self.mySHM.get_counter()

                if self.chB_dark_sub.isChecked():
                    try:
                        self.data_img -= self.myDRK
                    except AttributeError:
                        msg = self.lbl_stats.text()
                        add_msg = "Dark subtraction problem?"
                        self.lbl_stats.setText(msg)
                        pass

            self.refresh_plot()
            self.refresh_stats(add_msg=add_msg)


# ==========================================================
# ==========================================================
if __name__ == "__main__":
    main()
