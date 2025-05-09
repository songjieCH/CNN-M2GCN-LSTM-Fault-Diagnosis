import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from sensorConfig import SensorConfig

logging.basicConfig()
log = logging.getLogger()
log.setLevel(logging.DEBUG)

class DataAnalysis:
    def __init__(self):
        self.model = torch.load(os.path.join(os.getcwd(), 'model_best.pt'), map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.data_path = os.path.join(os.getcwd(), "data")
        self.data_length = 1000
        self.data_matrix = [[0.0] * self.data_length for _ in range(SensorConfig.sensor_unit_ids * 10)]
        self.ays_time = float('inf')
        self.data_status = [False] * SensorConfig.sensor_unit_ids

        self.running = True
        self.model_thread = ThreadPoolExecutor(max_workers=1)

    def stop(self):
        self.running = False

    def reset_status(self):
        with threading.Lock():
            for i in range(len(self.data_status)):
                self.data_status[i] = False
        self.ays_time = float('inf')

    def is_all_data_ready(self):
        return all(self.data_status)

    def run(self):
        def task():
            print("DataAnalysis task process")
            while self.running:
                if self.is_all_data_ready():
                    with open(self.data_path, 'w') as f:
                        json.dump(self.data_matrix, f)
                    self.reset_status()
                    self.model.eval()
                    with torch.no_grad():
                        input_data = torch.tensor(self.data_matrix, dtype=torch.float32).unsqueeze(0)
                        output = self.model(input_data)
                        log.info(f"output: {output}")
                else:
                    time.sleep(0.01)
        self.model_thread.submit(task)


DataAnalysis().run()