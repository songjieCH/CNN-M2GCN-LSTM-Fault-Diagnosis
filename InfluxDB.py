import logging
import time
from datetime import datetime

from influxdb import InfluxDBClient
import os

logging.basicConfig()
log = logging.getLogger()

class InfluxDBService:
    SUCCESS_CODE = "0x000000"
    EXEC_FAIL_CODE = "0x000002"
    WRITE_FAIL_CODE = "0x000004"

    def __init__(self):
        self.db = None
        self.local_path = "D:\\DevProject\\PythonProject\\FaultDiagnosis\\influxdb-1.7.4\\influxd.exe"
        self.data_path = " C:\\Users\\<YourUser>\\.influxdb\\data"
        self.user_name = os.getenv('INFLUX_USER') or 'admin'
        self.password = os.getenv('INFLUX_PASSWORD') or 'admin'
        self.host = os.getenv('INFLUX_HOST') or 'localhost'
        self.port = os.getenv('INFLUX_PORT') or 8086
        self.database = os.getenv('INFLUX_DATABASE') or 'am2025'
        self.is_running = False
        self.init_influxdb()

    def init_influxdb(self):
        if self.db is None:
            count = 3
            while self.local_path and not self.is_running and count >= 0:
                self.run_influxdb()
                time.sleep(1)
                count -= 1
            if not self.is_running:
                log.error("influxDB start failed")
                return None
            self.db = InfluxDBClient(host= self.host, port=self.port, username=self.user_name, password=self.password, database=self.database)
            if not self.db.switch_database(self.database):
                self.db.create_database(self.database)
            self.db.switch_database(self.database)
            log.info("influxDB init success")

    def run_influxdb(self):
        try:
            command = self.local_path
            import subprocess
            subprocess.Popen(command)
            self.is_running = True
            log.info("influxDB started")
        except Exception as e:
            log.error(f"influxDB start failed: {e}")

    def set_retention_policy(self, policy_name):
        self.db.create_retention_policy(policy_name, 'INF', 1, default=True)
        log.info(f"influxDB setRetentionPolicy: {policy_name}")

    def insert(self, measurement, time, fields):
        point = {
            "measurement": measurement,
            "time": time,
            "fields": fields
        }
        self.db.write_points([point], time_precision='ms')

    def query(self, query):
        try:
            result = self.db.query(query)
            return list(result.get_points())
        except Exception as e:
            log.error(f"influxDB query failed: {e}")
            return None

    def close(self):
        if self.is_running:
            self.db.close()
            self.is_running = False
            log.info("influxDB closed")

    def create_database(self, database_name):
        if self.is_running:
            self.db.create_database(database_name)

if __name__ == "__main__":
    influx_service = InfluxDBService()
    # influx_service.set_retention_policy("autogen")
    influx_service.insert("test_measurement", {"field1": 123, "field2": 456})
    result = influx_service.query("SELECT * FROM test_measurement")
    if result:
        for point in result:
            print(point)
    else:
        print("No data found")
    influx_service.close()