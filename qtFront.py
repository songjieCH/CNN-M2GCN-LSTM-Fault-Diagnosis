import datetime
import os
import sys
import threading
import matplotlib.dates as mdates
import torch

from PyQt5.QtCore import Qt, QDateTime, QTimer
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit, \
    QGroupBox, QDateTimeEdit, QSizePolicy, QFileDialog, QComboBox

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from modbus import ModbusService
from InfluxDB import InfluxDBService
from mymodel.train import Trainer, Predictor
import config
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger("pymodbus").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


# Local stand-alone processing
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()


        self.sensor_num = 6
        self.sensor_config = {
            1: {'ip': '192.168.1.164', 'port': 502},
            2: {'ip': '192.168.1.165', 'port': 502},
            3: {'ip': '192.168.1.166', 'port': 502},
            4: {'ip': '192.168.1.167', 'port': 502},
            5: {'ip': '192.168.1.168', 'port': 502},
            6: {'ip': '192.168.1.169', 'port': 502},
        }
        self.rate = 10
        self.stream_running = False
        self.diagnosis_model = None
        self.modbus = {
            1: ModbusService(self.sensor_config[1]['ip'], self.sensor_config[1]['port']),
            2: ModbusService(self.sensor_config[2]['ip'], self.sensor_config[2]['port']),
            3: ModbusService(self.sensor_config[3]['ip'], self.sensor_config[3]['port']),
            4: ModbusService(self.sensor_config[4]['ip'], self.sensor_config[4]['port']),
            5: ModbusService(self.sensor_config[5]['ip'], self.sensor_config[5]['port']),
            6: ModbusService(self.sensor_config[6]['ip'], self.sensor_config[6]['port']),
        }
        self.influxdb = InfluxDBService()
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignTop)
        main_layout.setSpacing(30)

        ##########################  一   数据采集模块  #################
        top_layout = QVBoxLayout()
        top_layout.setSpacing(20)
        top_layout.setAlignment(Qt.AlignTop)
        top_group = QGroupBox("数据采集与传输")
        top_group.setFixedSize(850, 140)

        ip_layout = QHBoxLayout()
        ip_label = QLabel("IP:")
        ip_label.setFixedSize(40, 30)
        ip_layout.addWidget(ip_label)
        self.ip_line = QLineEdit()
        self.ip_line.setReadOnly(True)
        self.ip_line.setFixedSize(200, 30)
        self.ip_line.setPlaceholderText('192.168.1.1')
        ip_layout.addWidget(self.ip_line)

        host_label = QLabel("PORT:")
        host_label.setFixedSize(40, 30)
        ip_layout.addWidget(host_label)
        self.post_line = QLineEdit()
        self.post_line.setReadOnly(True)
        self.post_line.setFixedSize(60, 30)
        self.post_line.setPlaceholderText('502')
        ip_layout.addWidget(self.post_line)

        id_layout = QHBoxLayout()
        id_label = QLabel("ID:")
        id_label.setFixedSize(40, 30)
        id_layout.addWidget(id_label)
        self.id_buttons = []
        for i in range(1, self.sensor_num + 1):
            id_button = QPushButton(str(i))
            id_button.setFixedSize(30, 30)
            id_button.setCheckable(True)
            id_button.setStyleSheet("background-color: red; color: white;")
            id_button.clicked.connect(self.check_sensor_status)
            id_layout.addWidget(id_button)
            self.id_buttons.append(id_button)

        ip_layout.addLayout(id_layout)

        top_layout.addLayout(ip_layout)

        coll_layout = QHBoxLayout()
        data_buttons_layout = QHBoxLayout()
        rate_label = QLabel("采样频率:")
        rate_label.setFixedSize(80, 30)
        data_buttons_layout.addWidget(rate_label)
        self.sample_rate = QLineEdit()
        self.sample_rate.setFixedSize(120, 30)
        self.sample_rate.setPlaceholderText("100 hz")
        self.sample_rate.editingFinished.connect(self.update_rate)
        data_buttons_layout.addWidget(self.sample_rate)
        self.collect_button = QPushButton("采集")
        self.collect_button.clicked.connect(self.collect_data)
        self.collect_button.setFixedSize(50, 30)
        self.stop_button = QPushButton("停止")
        self.stop_button.clicked.connect(self.stop_collect_data)
        self.stop_button.setFixedSize(50, 30)
        data_buttons_layout.addWidget(self.collect_button)
        data_buttons_layout.addWidget(self.stop_button)
        coll_layout.addLayout(data_buttons_layout)

        time_layout = QHBoxLayout()
        current_datetime = QDateTime.currentDateTime()
        self.datetime_edit = QDateTimeEdit()
        self.datetime_edit.setDateTime(current_datetime)
        self.datetime_edit.setFixedSize(160, 30)
        time_layout.addWidget(self.datetime_edit)
        spacer_label = QLabel("-")
        spacer_label.setFixedSize(10, 10)
        time_layout.addWidget(spacer_label)
        current_datetime = QDateTime.currentDateTime()
        self.datetime_edit_2 = QDateTimeEdit()
        self.datetime_edit_2.setDateTime(current_datetime)
        self.datetime_edit_2.setFixedSize(160, 30)
        time_layout.addWidget(self.datetime_edit_2)
        coll_layout.addLayout(time_layout)

        self.history_button = QPushButton("重放")
        self.history_button.clicked.connect(self.reload_data)
        self.history_button.setFixedSize(50, 30)
        coll_layout.addWidget(self.history_button)

        top_layout.addLayout(coll_layout)

        top_group.setLayout(top_layout)
        main_layout.addWidget(top_group)

        ################  二  模型增量学习模块 ####################################
        learn_mk_layout = QHBoxLayout()
        learn_group = QGroupBox("模型增量学习")
        learn_group.setFixedSize(850, 180)

        second_layout = QVBoxLayout()
        second_layout.setAlignment(Qt.AlignTop)
        second_layout.setSpacing(20)

        l1_layout = QHBoxLayout()
        select_model = QPushButton("增量模型")
        self.mode_label = QLineEdit()   # 训练模型路径
        self.mode_label.setFixedSize(300, 30)
        self.mode_label.setPlaceholderText("未选择增量学习模型")
        select_model.setFixedSize(100, 30)
        select_model.clicked.connect(self.current_model)
        l1_layout.addWidget(select_model)
        l1_layout.addWidget(self.mode_label)

        mod_label = QLabel("训练模式:")
        mod_label.setFixedSize(100, 30)
        self.mod_ql = QComboBox()
        self.mod_ql.setFixedSize(150, 30)
        self.mod_ql.addItems(["初始学习", "增量学习"])
        l1_layout.addWidget(mod_label)
        l1_layout.addWidget(self.mod_ql)

        second_layout.addLayout(l1_layout)

        l2_layout = QHBoxLayout()
        select_data = QPushButton("数据集")
        select_data.setFixedSize(100, 30)
        self.data_label = QLineEdit()
        self.data_label.setFixedSize(300, 30)
        self.data_label.setPlaceholderText("未选择数据集")
        self.train_data = []
        select_data.clicked.connect(self.select_data)
        l2_layout.addWidget(select_data)
        l2_layout.addWidget(self.data_label)

        epoch_label = QLabel("迭代次数:")
        epoch_label.setFixedSize(100, 30)
        self.epoch_ql = QLineEdit()
        self.epoch_ql.setFixedSize(150, 30)
        self.epoch_ql.setPlaceholderText("100")
        l2_layout.addWidget(epoch_label)
        l2_layout.addWidget(self.epoch_ql)
        second_layout.addLayout(l2_layout)

        l3_layout = QHBoxLayout()
        par_label = QLabel("输入参数:")
        par_label.setFixedSize(100, 30)
        self.par_line = QLineEdit()
        self.par_line.setFixedSize(600, 30)
        self.par_line.setPlaceholderText("输入超参数")
        l3_layout.addWidget(par_label)
        l3_layout.addWidget(self.par_line)
        second_layout.addLayout(l3_layout)

        learn_mk_layout.addLayout(second_layout)

        train_layout = QHBoxLayout()
        train_button = QPushButton("开始训练")
        train_button.setFixedSize(80, 80)
        train_button.clicked.connect(self.train_model)
        train_layout.addWidget(train_button)
        learn_mk_layout.addLayout(train_layout)

        learn_group.setLayout(learn_mk_layout)
        main_layout.addWidget(learn_group)

        ####################   三   诊断分析  ##################
        det_layout = QHBoxLayout()
        det_group = QGroupBox("诊断分析")
        det_group.setFixedSize(850, 400)

        third_layout = QVBoxLayout()
        third_layout.setAlignment(Qt.AlignTop)
        third_layout.setSpacing(20)

        d_l1_layout = QHBoxLayout()
        d_model = QPushButton("选择模型")
        d_model.setFixedSize(100, 30)
        d_model.clicked.connect(self.select_model)
        self.d_model_label = QLineEdit()
        self.d_model_label.setFixedSize(450, 30)
        self.d_model_label.setPlaceholderText("选择诊断模型")
        d_l1_layout.addWidget(d_model)
        d_l1_layout.addWidget(self.d_model_label)

        self.d_time = QLineEdit()
        self.d_time.setFixedSize(200, 30)
        self.d_time.setPlaceholderText("模型创建时间")
        d_l1_layout.addWidget(self.d_time)
        third_layout.addLayout(d_l1_layout)

        d_l2_layout = QHBoxLayout()
        self.run_da = QPushButton("运行诊断")
        self.run_da.setCheckable(True)  # 设置按钮为可切换状态
        self.run_da.clicked.connect(self.toggle_diagnosis)
        self.run_da.setFixedSize(100, 30)

        d_res_label = QLabel("设备状态:")
        d_res_label.setFixedSize(100, 30)
        self.d_res = QLineEdit()
        self.d_res.setFixedSize(510, 30)
        self.d_res.setPlaceholderText("设备运行正常")
        self.dot_status = QPushButton()
        self.dot_status.setFixedSize(20, 20)
        self.dot_status.setStyleSheet("background-color: green; border-radius: 10px;")

        d_l2_layout.addWidget(self.run_da)
        d_l2_layout.addWidget(d_res_label)
        d_l2_layout.addWidget(self.d_res)
        d_l2_layout.addWidget(self.dot_status)
        third_layout.addLayout(d_l2_layout)

        d_l3_layout = QVBoxLayout()
        d_mes_label = QLabel("告警信息:")
        d_mes_label.setFixedSize(100, 30)
        self.d_mes = QTextEdit()
        self.d_mes.setFixedSize(800, 200)
        self.d_mes.setPlaceholderText("无告警信息")
        d_l3_layout.addWidget(d_mes_label)
        d_l3_layout.addWidget(self.d_mes)
        third_layout.addLayout(d_l3_layout)

        det_layout.addLayout(third_layout)
        det_group.setLayout(det_layout)
        main_layout.addWidget(det_group)

        #################  四   数据显示模块  ########################
        bottom_layout = QHBoxLayout()
        bottom_layout.setAlignment(Qt.AlignTop)
        bottom_group = QGroupBox("数据显示")
        bottom_group.setFixedSize(850, 550)

        time_data_layout = QVBoxLayout()

        acl_label_layout = QHBoxLayout()
        self.x_acc_button = QPushButton("X轴加速度")
        self.x_acc_button.setCheckable(True)
        self.x_acc_button.clicked.connect(self.update_button_states)
        acl_label_layout.addWidget(self.x_acc_button)

        self.y_acc_button = QPushButton("Y轴加速度")
        self.y_acc_button.setCheckable(True)
        self.y_acc_button.clicked.connect(self.update_button_states)
        acl_label_layout.addWidget(self.y_acc_button)

        self.z_acc_button = QPushButton("Z轴加速度")
        self.z_acc_button.setCheckable(True)
        self.z_acc_button.clicked.connect(self.update_button_states)
        acl_label_layout.addWidget(self.z_acc_button)

        # acl_label_layout.addWidget(QPushButton("增加变量"))
        time_data_layout.addLayout(acl_label_layout)

        # todo
        ana_button_layout = QHBoxLayout()
        self.sum_count = QPushButton("故障总数")
        self.percentage_button = QPushButton("故障占比")
        self.log_button = QPushButton("故障日志")
        ana_button_layout.addWidget(self.sum_count)
        ana_button_layout.addWidget(self.percentage_button)
        ana_button_layout.addWidget(self.log_button)
        time_data_layout.addLayout(ana_button_layout)

        tu_layout = QHBoxLayout()
        self.canvas = MplCanvas(self, width=10, height=20, dpi=100)
        tu_layout.addWidget(self.canvas)
        time_data_layout.addLayout(tu_layout)

        bottom_layout.addLayout(time_data_layout)

        bottom_group.setLayout(bottom_layout)
        main_layout.addWidget(bottom_group)

        # 设置主布局
        self.setLayout(main_layout)
        self.setWindowTitle("在线检测系统")
        self.setGeometry(1000, 100, 800, 1400)
        self.setFixedSize(880, 1400)


    ###################################### 第一部分 #####################################################
    def check_sensor_status(self):
        sender = self.sender()
        for id_button in self.id_buttons:
            id_button.setChecked(sender == id_button)
        id = int(sender.text())
        self.ip_line.setText(self.sensor_config[id]['ip'])
        self.post_line.setText(str(self.sensor_config[id]['port']))
        self.modbus[id].connect()
        if self.modbus[id].is_sensor_connected(id):
            self.id_buttons[id - 1].setStyleSheet("background-color: green; color: white;")
            self.d_mes.append(f"{datetime.datetime.now()} ID{id} 连接成功\n")
        else:
            self.id_buttons[id - 1].setStyleSheet("background-color: red; color: white;")
            self.d_mes.append(f"{datetime.datetime.now()} ID{id} 连接失败\n")
    def update_rate(self):
        self.rate = self.sample_rate.text()
    def collect_data(self):
        self.stream_running = True
        self.d_mes.append(f"{datetime.datetime.now()} 主动数据采集开始\n")
        self._schedule_next_collection()
    def _schedule_next_collection(self):
        if self.stream_running:
            self.collect_thread = threading.Timer(1 / self.rate , self._collect_data_thread)
            self.collect_thread.start()
    def _collect_data_thread(self):
        for button in self.id_buttons:
            id = int(button.text())
            if self.modbus[id].is_sensor_connected():
                data = self.modbus[id].read_data()
                if data:
                    # 本地数据存储
                    self.influxdb.insert(measurement=f"sensor{id}", time=data[3], fields={
                        "x_acceleration": data[0],
                        "y_acceleration": data[1],
                        "z_acceleration": data[2]
                    })
        self._schedule_next_collection()  # Schedule the next collection
    def stop_collect_data(self):
        if self.stream_running:
            self.stream_running = False
            if hasattr(self, 'collect_thread'):
                self.collect_thread.cancel()  # Stop the timer
            self.d_mes.append(f"{datetime.datetime.now()} 数据采集停止\n")

    def reload_data(self):
        pass

    ############################ 第二部分 ###################################################
    def train_model(self):
        model_path = self.mode_label.text()
        data_path = self.data_label.text()
        data = self.train_data
        epoch = int(self.epoch_ql.text())
        type = self.mod_ql.currentText()

        def run_training():
            train = Trainer(data_path=data_path, model_path=model_path, class_num=len(data), signal_num=6,
                            pf=self.d_mes)
            if type == "初始学习":
                self.d_mes.append(f"{datetime.datetime.now()} 模型开始训练...\n")
                train.train(epoch)
            elif type == "增量学习":
                self.d_mes.append(f"{datetime.datetime.now()} 模型增量训练...\n")
                train.train_inc(new_classes=len(data), iterations=epoch)
            else:
                self.d_mes.append(f"{datetime.datetime.now()} 未知模式\n")
                return
        training_thread = threading.Thread(target=run_training)
        training_thread.start()

    def current_model(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select File", "", "All Files (*.pt);;",
                                                   options=options)
        if file_name:
            self.mode_label.setText(file_name)
    def select_data(self):
        options = QFileDialog.Options()
        file_names, _ = QFileDialog.getOpenFileNames(self, "Select File", "", "All Files (*.csv);;Text Files (*.txt)",
                                                   options=options)
        if file_names:
            directory = os.path.dirname(file_names[0])
            self.data_label.setText(directory)
            for file_name in file_names:
                self.train_data.append(file_name)

    ########################## 第三部分 ####################################3
    def select_model(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select File", "", "All Files (*.pt);;",
                                                   options=options)
        if file_name:
            modification_time = os.path.getmtime(file_name)
            formatted_time = datetime.datetime.fromtimestamp(modification_time).strftime('%Y-%m-%d %H:%M:%S')
            self.d_time.setText(formatted_time)
            self.d_model_label.setText(file_name)
            self.diagnosis_model = torch.load(file_name, map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            self.pred = Predictor(self.diagnosis_model)
            self.d_mes.append(f"{datetime.datetime.now()} 选择诊断模型成功\n"
                              f"model info: {self.diagnosis_model}\n")

    def toggle_diagnosis(self):
        if self.run_da.isChecked():
            self.run_da.setText("停止诊断")
            self.diagnosis_running = True
            self.run_flink()  # 开始诊断
        else:
            self.run_da.setText("运行诊断")
            self.diagnosis_running = False
            self.stop_flink()  # 停止诊断

    def run_flink(self):
        self.d_mes.append(f"{datetime.datetime.now()} 运行诊断...\n")
        self.timer2 = threading.Timer(1, self._run_flink)
        self.timer2.start()

    def _run_flink(self):
        if not self.diagnosis_running:
            return
        current_time = datetime.datetime.now()
        # data = self._query_diagnosis_data()
        # 模拟数据
        data = torch.randn(1, 2560, 6)
        if self.diagnosis_model is None:
            self.d_mes.append(f"{datetime.datetime.now()} 诊断模型未选择\n")
            return
        if data is None:
            self.d_mes.append(f"{datetime.datetime.now()} 获取数据失败\n")
            return
        self.diagnosis_model.eval()

        predicted = self.pred.predict(data)[0]
        device_status = config.status_label[predicted.item()]
        if self.influxdb is not None:
            self.influxdb.insert(measurement="diagnosis", time=current_time, fields={
                "result": device_status
            })
        if predicted in config.normal_label:
            self.dot_status.setStyleSheet("background-color: green; border-radius: 10px;")
            self.d_res.setText(f"{datetime.datetime.now()} 设备运行正常\n")
        else:
            self.dot_status.setStyleSheet("background-color: red; border-radius: 10px;")
            self.d_res.setText(f"{datetime.datetime.now()} 设备故障, 诊断结果: {device_status}\n")
            self.d_mes.append(f"{datetime.datetime.now()} 故障结果: {device_status}\n")
        if self.diagnosis_running:
            self.timer2 = threading.Timer(1, self._run_flink)
            self.timer2.start()

    def _query_diagnosis_data(self):
        # 根据训练模型输入格式，从数据库或者kafka集群中读取数据
        pass

    def stop_flink(self):
        if hasattr(self, 'timer2') and self.timer2.is_alive():
            self.timer2.cancel()
            self.d_mes.append(f"{datetime.datetime.now()} 停止诊断...\n")
            self.d_res.setText(f"{datetime.datetime.now()} 停止诊断\n")

    ######################## 第四部分 #########################################
    def update_button_states(self):
        sender = self.sender()
        self.x_acc_button.setChecked(sender == self.x_acc_button)
        self.y_acc_button.setChecked(sender == self.y_acc_button)
        self.z_acc_button.setChecked(sender == self.z_acc_button)
        self.timer1 = QTimer(self)
        self.timer1.timeout.connect(self._query_and_update_plot)
        self.timer1.start(1000)


    def _query_and_update_plot(self):
        cur_time = datetime.datetime.now()
        show_time = cur_time - datetime.timedelta(seconds=10)
        sensor_id = self.get_selected_sensor_id()
        variable = self.get_selected_variable()
        data = self._query_database(sensor_id, variable, show_time, cur_time)
        if not data:
            return
        self.update_plot(data)

    def get_selected_sensor_id(self):
        for button in self.id_buttons:
            if button.isChecked():
                return int(button.text())
        return None

    def get_selected_variable(self):
        if self.x_acc_button.isChecked():
            return "x"
        elif self.y_acc_button.isChecked():
            return "y"
        elif self.z_acc_button.isChecked():
            return "z"
        return None

    def _query_database(self, sensor_id, variable, start_time, end_time):

        if sensor_id is None or variable is None:
            self.d_mes.append(f"{datetime.datetime.now()} 请选择传感器和变量\n")
            return None
        try:
            query = f"""
                    SELECT *
                    FROM sensor{sensor_id} 
                    WHERE time >= '{start_time}' AND time <= '{end_time}'
                    """
            tables = self.influxdb.query(query)
            if tables is None:
                return None
            if variable == "x":
                data = [(point['time'], point['x_acceleration']) for point in tables]
            elif variable == "y":
                data = [(point['time'], point['y_acceleration']) for point in tables]
            elif variable == "z":
                data = [(point['time'], point['z_acceleration']) for point in tables]
            else:
                raise ValueError("Invalid variable selected")
            return data
        except Exception as e:
            logging.error(f"Query failed: {e}")
            self.status_edit.setText(f"{datetime.datetime.now()} 查询失败: {e}\n")
            return None

    def update_plot(self, data):
        self.canvas.axes.clear()
        time = [datetime.datetime.strptime(t, '%Y-%m-%dT%H:%M:%S.%fZ') for t, _ in data]
        values = [round(v, 2) for _, v in data]
        time = mdates.date2num(time)
        self.canvas.axes.xaxis.set_major_locator(mdates.SecondLocator(interval=2))
        self.canvas.axes.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        self.canvas.axes.plot(time, values, color='blue')
        # self.canvas.axes.set_ylim(-1, 5)
        self.canvas.axes.grid()
        self.canvas.draw()

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())