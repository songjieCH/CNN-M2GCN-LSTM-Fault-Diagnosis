import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, ScheduledExecutorService
from queue import Queue, Empty

import sensorConfig
import modbus
import InfluxDB
import RocketProducer

logging.basicConfig()
log = logging.getLogger()
log.setLevel(logging.DEBUG)

class DataCollection:
    try_count = 0

    def __init__(self, modbus, influxdb, producer):
        self.modbus_service = modbus_service
        self.influxdb_service = influxdb_service
        self.producer_service = producer_service
        self.scheduled_futures = [None] * SensorConfig.sensor_unit_ids
        self.executor_service = ThreadPoolExecutor(max_workers=4)
        self.data_queue = [Queue(maxsize=10000) for _ in range(SensorConfig.sensor_unit_ids)]

    def init(self):
        pass  # Initialization is handled in __init__

    def run(self, label):
        def collect_data(index):
            try:
                self.collect_data(index)
            except Exception as e:
                log.warn(f"DataCollection::run - {e}")

        for i in range(SensorConfig.sensor_unit_ids):
            self.scheduled_futures[i] = self.executor_service.submit(collect_data, i)

        while True:
            for i in range(SensorConfig.sensor_unit_ids):
                try:
                    data = self.data_queue[i].get(timeout=0.1)
                    self.process_data(data, i, label)
                except Empty:
                    self.try_count += 1
            if self.try_count > 50:
                break

    def update_interval(self, new_interval):
        SensorConfig.collect_interval = new_interval
        for future in self.scheduled_futures:
            if future:
                future.cancel()

    def collect_data(self, unit_id):
        data = self.modbus_service.read_data(unit_id)
        self.data_queue[unit_id].put(data)

    def process_data(self, data, unit_id, label):
        fields = SensorConfig.to_map(data)
        name = f"sensor-{unit_id} label-{label}"
        if self.influxdb_service:
            self.influxdb_service.insert(name, fields)
        # if self.producer_service:
        #     self.producer_service.send(name, data, 0)

# Example usage
if __name__ == "__main__":
    modbus_service = Modbus()
    influxdb_service = InfluxDB()
    producer_service = RocketProducer()
    data_collection = DataCollection(modbus_service, influxdb_service, producer_service)
    data_collection.run("example_label")