from copy import deepcopy, copy
import sys
import os
from pathlib import Path
import random
import json
import ast
from pprint import pformat

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QSpacerItem, QLabel,QLineEdit,\
    QTableWidget, QTableWidgetItem, QGridLayout, QHBoxLayout, QVBoxLayout, QTextEdit,\
    QDialog, QPushButton, QAction, QSizePolicy, QMenu, QFileDialog, QMessageBox, QTabWidget,\
    QDialogButtonBox
# from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QPainter, QBrush, QPen
from PyQt5.QtCore import Qt
from PyQt5 import QtGui

import pandas as pd

import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


import networkx as nx

from MCL import MCLClusterizer
from KMedoid import KMedoidsClusterizer
from Spectral import SpectralEmbeddingEstimator, SpectralClusterizer,\
    SpectralConfigurator


matplotlib.use('Qt5Agg')

XPOS = 0
YPOS = 0
WIDTH = 1920
HEIGHT = 1080
MINIMUM_HEIGHT = 480
MINIMUM_HEIGHT = 480

# Graph Example
graph_3 = {1: [2, 3],
           2: [1, 3, 4],
           3: [1, 2, 4],
           4: [2, 3, 5],
           5: [4, 10, 6],
           6: [10, 9, 7],
           7: [6],
           8: [9],
           9: [6, 8],
           10: [5, 6, 11],
           11: [10, 12, 15],
           12: [11, 13],
           13: [12, 14],
           14: [13, 11],
           15: [11]}

graph = {1: [2, 3],
         2: [1, 3, 5],
         3: [1, 2, ],
         4: [5, 6],
         5: [4, 6],
         6: [4, 5]}

graph_2 = {1: [2, 3],
           2: [1, 3, 5],
           3: [1, 2, ],
           }


G = nx.Graph(graph_3)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(XPOS, YPOS, WIDTH, HEIGHT)
        self.setMinimumSize(480, 80)
        self.setWindowTitle("Graph clustering")
        self.mainWidget = QWidget(self)
        self.setCentralWidget(self.mainWidget)
        self.mainLayout = QHBoxLayout()
        self.mainWidget.setLayout(self.mainLayout)

        self.initButtonsMenu()
        self.initTabMenu()
        #
        self.initMenuBar()

    def initButtonsMenu(self):
        btnToolsLayout = QVBoxLayout(self)
        buttonPlotSpectrEmb = QPushButton("Plot nodes' spectral embeds")
        buttonClsMcl = QPushButton("Cluster using MCL")
        buttonClsMedoid = QPushButton("Cluster using KMedoid")
        buttonClsSpectral = QPushButton("Cluster using Spectral clustering")

        buttons2functions = {buttonPlotSpectrEmb: self.buttonPlotSpectrEmbClicked,
                             buttonClsMcl: self.buttonClsMclClicked,
                             buttonClsMedoid: self.buttonClsMedoidClicked,
                             buttonClsSpectral: self.buttonClsSpectralClicked,
                             }

        for button, buttonFunction in buttons2functions.items():
            button.setSizePolicy(QSizePolicy.Expanding,
                                 QSizePolicy.Expanding)
            btnToolsLayout.addWidget(button, 1)
            button.clicked.connect(buttonFunction)
        verticalSpacer = QSpacerItem(100, 100, QSizePolicy.Expanding,
                                               QSizePolicy.Expanding)
        for _ in range(6):
            btnToolsLayout.addItem(verticalSpacer)
        self.mainLayout.addLayout(btnToolsLayout, 2)

    def initTabMenu(self):
        self.tabwidget = QTabWidget(movable=True)
        self.tabwidget.tabBarDoubleClicked.connect(self.dublicateTab)
        self.tabwidget.setTabsClosable(True)
        self.tabwidget.tabCloseRequested.connect(
            lambda index: self.tabwidget.removeTab(index))
        tab_1 = TabGraph(nx.Graph(graph))
        tab_2 = TabGraph(nx.Graph(graph_2))
        tab_3 = TabGraph(nx.Graph(graph_3))
        self.tabwidget.addTab(tab_1, "Example graph 1")
        self.tabwidget.addTab(tab_2, "Example graph 2")
        self.tabwidget.addTab(tab_3, "Example graph 3")
        self.mainLayout.addWidget(self.tabwidget, 8)

    def dublicateTab(self):
        currentTab = self.tabwidget.currentWidget()
        dublicateTab = None
        if isinstance(currentTab, TabGraph):
            dublicateTab = TabGraph.fromTabGraph(currentTab)
        elif isinstance(currentTab, TabEmbeds):
            dublicateTab = TabEmbeds.fromTabEmbeds(currentTab)
        self.tabwidget.addTab(dublicateTab,
                              self.tabwidget.tabText(self.tabwidget.currentIndex()))

    def initMenuBar(self):
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        helpMenu = menubar.addMenu('&Help')
        # fileMenu.addAction(fileAction)

        loadMenu = QMenu('Load graph ', self)
        loadTxtAction = QAction('Load from txt', self)
        loadJSONAction = QAction('Load from JSON', self)
        loadCSVAction = QAction('Load from CSV', self)
        loadTxtAction.triggered.connect(self.loadTxtGraph)
        loadJSONAction.triggered.connect(self.loadJSONGraph)
        loadCSVAction.triggered.connect(self.loadCSVGraph)
        loadMenu.addAction(loadTxtAction)
        loadMenu.addAction(loadJSONAction)
        loadMenu.addAction(loadCSVAction)

        saveMenu = QMenu('Save graph ', self)
        saveTxtAction = QAction('Save from txt', self)
        saveJSONAction = QAction('Save from JSON', self)
        saveCSVAction = QAction('Save from CSV', self)
        saveTxtAction.triggered.connect(self.saveTxtGraph)
        saveJSONAction.triggered.connect(self.saveJSONGraph)
        saveCSVAction.triggered.connect(self.saveCSVGraph)
        saveMenu.addAction(saveTxtAction)
        saveMenu.addAction(saveJSONAction)
        saveMenu.addAction(saveCSVAction)

        exportGraphMenu = QMenu('Export graph image ', self)
        exportGraphPNGAction = QAction('Export as PNG', self)
        exportGraphPNGAction.triggered.connect(self.exportGraphPNGImage)
        exportGraphMenu.addAction(exportGraphPNGAction)

        exitAction = QAction('Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(self.close)

        fileMenu.addMenu(loadMenu)
        fileMenu.addMenu(saveMenu)
        fileMenu.addMenu(exportGraphMenu)
        fileMenu.addAction(exitAction)

        helpAction = QAction('Help', self)
        aboutAction = QAction('About', self)
        helpAction.triggered.connect(self.helpWindowOpen)
        aboutAction.triggered.connect(self.aboutWindowOpen)
        helpMenu.addAction(helpAction)
        helpMenu.addAction(aboutAction)

    def loadTxtGraph(self):
        home_dir = str(Path.home())
        fname, format = QFileDialog.getOpenFileName(self,
                                                    'Open file',
                                                    home_dir,
                                                    "Text files (*.txt)")
        if fname:
            G = nx.read_edgelist(fname, create_using=nx.Graph())
            newTab = TabGraph(G)
            self.tabwidget.addTab(newTab, fname.split("/")
                                  [-1].split(".txt")[0])
            self.tabwidget.setCurrentIndex(-1)

    def loadJSONGraph(self):
        home_dir = str(Path.home())
        fname, format = QFileDialog.getOpenFileName(self,
                                                    'Open file',
                                                    home_dir,
                                                    "Json (*.json)")
        if fname:
            data = {}
            with open(fname, "r") as f:
                data = json.load(f)
            G = nx.from_dict_of_lists(data, create_using=nx.Graph())
            newTab = TabGraph(G)
            self.tabwidget.addTab(newTab, fname.split("/")
                                  [-1].split(".json")[0])
            self.tabwidget.setCurrentIndex(-1)

    def loadCSVGraph(self):
        home_dir = str(Path.home())
        fname, format = QFileDialog.getOpenFileName(self,
                                                    'Open file',
                                                    home_dir,
                                                    "CSV (*.csv)")
        if fname:
            Data = open(fname, "r")
            G = nx.parse_edgelist(Data, comments='t',
                                  delimiter=',',
                                  create_using=nx.Graph())
            newTab = TabGraph(G)
            self.tabwidget.addTab(newTab, fname.split("/")
                                  [-1].split(".csv")[0])
            self.tabwidget.setCurrentIndex(-1)

        pass

    def saveTxtGraph(self):
        currentTab = self.tabwidget.currentWidget()
        if type(currentTab) == TabGraph:
            fileName, extension = QFileDialog.getSaveFileName(
                self, "Save graph", ".txt", "txt (*.txt)")
            if fileName:
                graph = currentTab.graph
                nx.write_edgelist(graph, fileName)

        elif type(currentTab) == TabEmbeds:
            QMessageBox.information(self, "Open graph tab",
                                    "Open tab with graph as network, please.",
                                    QMessageBox.Ok)
            return

    def saveJSONGraph(self):
        currentTab = self.tabwidget.currentWidget()
        if type(currentTab) == TabGraph:
            fileName, extension = QFileDialog.getSaveFileName(
                self, "Save graph", ".json", "JSON (*.json)")
            if fileName:
                graph = currentTab.graph
                data = nx.to_dict_of_lists(graph)
                with open(fileName, "w") as f:
                    json.dump(data, f)

        elif type(currentTab) == TabEmbeds:
            QMessageBox.information(self, "Open graph tab",
                                    "Open tab with graph as network, please.",
                                    QMessageBox.Ok)
            return

    def saveCSVGraph(self):
        currentTab = self.tabwidget.currentWidget()
        if type(currentTab) == TabGraph:
            fileName, extension = QFileDialog.getSaveFileName(
                self, "Save graph", ".csv", "CSV (*.csv)")
            if fileName:
                graph = currentTab.graph
                nx.write_edgelist(graph, fileName, delimiter=",")

        elif type(currentTab) == TabEmbeds:
            QMessageBox.information(self, "Open graph tab",
                                    "Open tab with graph as network, please.",
                                    QMessageBox.Ok)
            return

    def exportGraphPNGImage(self):
        currentTab = self.tabwidget.currentWidget()
        if type(currentTab) == TabGraph:
            fileName, extension = QFileDialog.getSaveFileName(
                self, "Export graph image", ".png", "PNG (*.png)")
            if fileName:
                currentTab.graphCanvas.fig.savefig(fileName)
        elif type(currentTab) == TabEmbeds:
            fileName, extension = QFileDialog.getSaveFileName(self,
                                                              "Export node embedding image",
                                                              ".png", "PNG (*.png)")
            if fileName:
                currentTab.canvas.figure.savefig(fileName)

    def helpWindowOpen(self):
        message_box = QMessageBox(self)
        message_box.setText("Help guide")
        help_text = None
        with open("./app_data/help.txt", "r") as f:
            help_text = ""
            for line in f.readlines():
                help_text += line
        if not help_text:
            QMessageBox().warning(self, "Sorry",
                                  "Something went wrong with help text. Sorry",
                                  QMessageBox.Ok)
            return
        message_box.setInformativeText(help_text)
        message_box.setStandardButtons(QMessageBox.Ok)
        message_box.exec_()

    def aboutWindowOpen(self):
        message_box = QMessageBox(self)
        about_text = None
        with open("./app_data/about.txt", "r") as f:
            about_text = ""
            for line in f.readlines():
                about_text += line
        if not about_text:
            QMessageBox().warning(self, "Sorry",
                                  "Something went wrong with about text. Sorry",
                                  QMessageBox.Ok)
            return
        message_box.setInformativeText(about_text)
        message_box.setStandardButtons(QMessageBox.Ok)
        message_box.exec_()

    def buttonPlotSpectrEmbClicked(self):
        if(type(self.tabwidget.currentWidget()) != TabGraph):
            return
        graph = self.tabwidget.currentWidget().graph
        spectralEstimator = SpectralEmbeddingEstimator()
        nodes_spectral_embeds = spectralEstimator.fit_predict(graph)
        tabEmbeds = TabEmbeds(nodes_spectral_embeds)
        self.tabwidget.addTab(tabEmbeds, "graph_embeds")
        self.tabwidget.setCurrentWidget(tabEmbeds)

    def buttonClsMclClicked(self):
        tab = self.tabwidget.currentWidget()
        if(type(tab) != TabGraph):
            return
        graph = tab.graph
        try:
            mclClusterizer = MCLClusterizer()
            clusterized_nodes_sets = mclClusterizer.fit_predict(graph)
            tab.drawClusterizedGraph(clusterized_nodes_sets)
        except Exception as exception:
            QMessageBox().critical(self, "Error",
                                   f"Something went wrong while fitting and predicting, try again. Exception: {exception}", QMessageBox.Ok)
            return

    def buttonClsMedoidClicked(self):
        tab = self.tabwidget.currentWidget()
        if(type(tab) != TabGraph):
            return
        graph = tab.graph

        configs = None
        kmedoidParamsDialog = KMedoidConfigDialog()
        returned_value = kmedoidParamsDialog.exec_()
        kMedoidsClusterizer = None
        clusterized_nodes_sets = None

        if returned_value:
            configs = returned_value
        else:
            return
        try:
            kMedoidsClusterizer = KMedoidsClusterizer(**configs)
        except Exception as exception:
            QMessageBox().critical(self, "Error",
                                   f"Something went wrong with configs for model, try again. Exception: {exception}", QMessageBox.Ok)
            return
        try:
            clusterized_nodes_sets = kMedoidsClusterizer.fit_predict(graph)
            tab.drawClusterizedGraph(clusterized_nodes_sets)
        except Exception as exception:
            QMessageBox().critical(self, "Error",
                                   f"Something went wrong while fitting and predicting, try again. Exception: {exception}", QMessageBox.Ok)
            return

    def buttonClsSpectralClicked(self):
        tab = self.tabwidget.currentWidget()
        if(type(tab) != TabGraph):
            return
        graph = tab.graph

        configs = None
        spectralParamsDialog = SpectralConfigDialog()
        returned_value = spectralParamsDialog.exec_()

        if returned_value:
            configs = returned_value
        else:
            return

        configurator = SpectralConfigurator(configs)
        estimator_params = configurator.get_params_estimator()
        clusterizer_params = configurator.get_params_clusterizer()

        node_embeds = None
        try:
            spectralEstimator = SpectralEmbeddingEstimator()
            spectralEstimator.fit(graph)
            node_embeds = spectralEstimator.predict(**estimator_params)
        except Exception as exception:
            QMessageBox().critical(self, "Error",
                                   f"Something went wrong with embeddings, try again. Exception: {exception}", QMessageBox.Ok)
            return

        try:
            spectralClusterizer = SpectralClusterizer(**clusterizer_params)
            clusterized_nodes_sets = spectralClusterizer.fit_predict(
                node_embeds)
            tab.drawClusterizedGraph(clusterized_nodes_sets)
        except Exception as exception:
            QMessageBox().critical(self, "Error",
                                   f"Something went wrong with clustering, try again. Exception: {exception}", QMessageBox.Ok)
            return


class TabGraph(QWidget):
    def __init__(self, graph=nx.Graph(graph)):
        super().__init__()
        self.graph = graph

        # Init UI
        self.tabLayout = QHBoxLayout()
        self.graphCanvas = self.createGraphCanvas()
        self.tabLayout.addWidget(self.graphCanvas, 7)
        self.editorMenu = self.initEditorMenu()
        self.tabLayout.addWidget(self.editorMenu, 4)
        self.setLayout(self.tabLayout)

    def createGraphCanvas(self):
        graphCanvas = GraphCanvas()
        graphCanvas.drawGraph(self.graph)
        return graphCanvas

    def drawClusterizedGraph(self, clusterized_nodes_sets):
        self.graphCanvas.drawGraph(self.graph, clusterized_nodes_sets)

    def initEditorMenu(self):
        self.editorMenuLayout = QVBoxLayout(self)
        self.enterGraphButton = QPushButton("Enter graph")
        self.enterGraphButton.clicked.connect(self.changeGraph)
        self.graphEditField = QTextEdit(self)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.graphEditField.setFont(font)
        self.graphEditField.setPlainText(pformat(nx.to_dict_of_lists(self.graph),
                                                 width=40))
        self.graphEditField.setSizePolicy(QSizePolicy.Expanding,
                                          QSizePolicy.Expanding)
        self.editorMenuLayout.addWidget(self.graphEditField, 4)
        self.editorMenuLayout.addWidget(self.enterGraphButton, 1)
        self.widget = QWidget()
        self.widget.setLayout(self.editorMenuLayout)
        return self.widget

    def changeGraph(self):
        try:
            graphText = self.graphEditField.toPlainText()
            self.graph = nx.from_dict_of_lists(ast.literal_eval(graphText))
            self.graphCanvas.drawGraph(self.graph)
        except:
            QMessageBox.critical(self, "Error", "Incorrect graph")

    @staticmethod
    def fromTabGraph(other):
        return TabGraph(other.graph)


class GraphCanvas(FigureCanvas):
    def __init__(self, parent=None, figsize=None,  dpi=100):
        self.fig = Figure(figsize=figsize, dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def drawGraph(self, graph, clustering=None, nodes_colors=None):
        self.fig.clf()
        self.axes = self.fig.add_subplot(111)
        if clustering:
            colors = [(random.random(), random.random(), random.random()) for _
                      in range(len(clustering))]
            nodes_colors = []
            for set_of_nodes, color in zip(clustering, colors):
                for node_index in set_of_nodes:
                    nodes_colors.append((node_index, color))
            nodes_colors = [*map(lambda item: item[1],
                                 sorted(nodes_colors, key=lambda item:item[0]))]
        nx.draw_kamada_kawai(graph,
                             with_labels=True,
                             node_color=nodes_colors,
                             ax=self.axes)
        self.fig.canvas.draw_idle()

    def clear(self):
        self.fig.clf()


class SpectralConfigDialog(QDialog):
    def __init__(self):
        super(SpectralConfigDialog, self).__init__()
        self.setWindowTitle("Config")
        self.initUI()
        self.retrunVal = None

    def initUI(self):
        self.mainWidget = QWidget(self)
        self.mainLayout = QVBoxLayout()
        self.mainWidget.setLayout(self.mainLayout)
        self.configLayout = QGridLayout()
        self.mainLayout.addLayout(self.configLayout, 5)

        # Number of clusters
        self.numOfClustersLabel = QLabel("Set number of clusters")
        self.configLayout.addWidget(self.numOfClustersLabel, 1, 1)
        self.numOfClustersField = QLineEdit()
        self.configLayout.addWidget(self.numOfClustersField, 1, 2)

        # Dimension of embedding
        self.embedDimensionLabel = QLabel("Set dimension of embedding")
        self.configLayout.addWidget(self.embedDimensionLabel, 2, 1)
        self.embedDimensionField = QLineEdit()
        self.configLayout.addWidget(self.embedDimensionField, 2, 2)

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok
                                          | QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.mainLayout.addWidget(self.buttonBox, 1)
        self.mainWidget.adjustSize()

    def accept(self):
        try:
            numberOfClusters = int(self.numOfClustersField.text())
            dimension = int(self.embedDimensionField.text())
        except:
            QMessageBox().critical(self, "Error", "Invalid parameters", QMessageBox.Ok)
            return
        self.retrunVal = {"n_clusters": numberOfClusters,
                          "node_embedding_dimension": dimension}
        self.close()
        return {"n_clusters": numberOfClusters, "node_embedding_dimension": dimension}

    def exec_(self):
        super(SpectralConfigDialog, self).exec_()
        return self.retrunVal


class KMedoidConfigDialog(QDialog):
    def __init__(self):
        super(KMedoidConfigDialog, self).__init__()
        self.setWindowTitle("Config")
        self.initUI()
        self.retrunVal = None

    def initUI(self):
        self.mainWidget = QWidget(self)
        self.mainLayout = QVBoxLayout()
        self.mainWidget.setLayout(self.mainLayout)
        self.configLayout = QGridLayout()
        self.mainLayout.addLayout(self.configLayout, 5)

        # Number of clusters
        self.numOfClustersLabel = QLabel("Enter number of clusters")
        self.configLayout.addWidget(self.numOfClustersLabel, 1, 1)
        self.numOfClustersField = QLineEdit()
        self.configLayout.addWidget(self.numOfClustersField, 1, 2)

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok
                                          | QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.mainLayout.addWidget(self.buttonBox, 1)
        self.mainWidget.adjustSize()

    def accept(self):
        try:
            numberOfClusters = int(self.numOfClustersField.text())
        except:
            QMessageBox().critical(self, "Error", "Invalid parameters", QMessageBox.Ok)
            return
        self.retrunVal = {"n_clusters": numberOfClusters}
        self.close()
        return {"n_clusters": numberOfClusters}

    def exec_(self):
        super(KMedoidConfigDialog, self).exec_()
        return self.retrunVal


class TabEmbeds(QWidget):
    def __init__(self, embeds):
        super().__init__()
        self.embeds = embeds
        self.tabLayout = QHBoxLayout()
        self.canvas = self.createGraphCanvas()
        self.tabLayout.addWidget(self.canvas)
        self.setLayout(self.tabLayout)

    def createGraphCanvas(self):
        self.fig = plt.figure()
        ax = self.fig.add_subplot(111)
        ax.scatter(self.embeds.T[0], self.embeds.T[1])
        ax.grid()
        canvas = FigureCanvas(self.fig)
        canvas.draw_idle()
        return canvas

    @staticmethod
    def fromTabEmbeds(other):
        return TabEmbeds(other.embeds)


if __name__ == "__main__":
    if os.name == "nt":  # if windows
        from PyQt5 import __file__
        pyqt_plugins = os.path.join(os.path.dirname(__file__), "Qt", "plugins")
        QApplication.addLibraryPath(pyqt_plugins)
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = pyqt_plugins
    # QApplication.setAttribute(Qt.AA_EnableHighDpiScaling,True)
    app = QApplication(sys.argv)
    app.processEvents()
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
