import logging
import struct
from datetime import datetime

from pymodbus.client import ModbusTcpClient
from pymodbus.payload import BinaryPayloadBuilder

logging.basicConfig()
log = logging.getLogger()
log.setLevel(logging.DEBUG)

class ModbusService:
    SUCCESS_CODE = "0x000000"
    EXEC_FAIL_CODE = "0x000002"
    WRITE_FAIL_CODE = "0x000004"

    def __init__(self, ip, port):
        self.client = ModbusTcpClient(host=ip, port=port, timeout=0.5)

    def connect(self):
        self.client.connect()

    def is_connected(self):
        return self.client.connected

    def is_sensor_connected(self, unit_id=1):
        if not self.is_connected():
            return False
        try:
            response = self.client.read_holding_registers(unit_id, 1)
            return response.isError() == False
        except:
            return False

    def read_data(self, unit_id=1):
        data = self.read_holding_registers(0, 7, unit_id)
        if data is not None:
            # acceleration_x, acceleration_y, acceleration_z, time
            return [data[4] / 100.0, data[5] / 100.0, data[6] / 100.0, datetime.now()]

    def release(self):
        self.client.close()

    def write_holding_registers(self, address, value, unit_id):
        try:
            builder = BinaryPayloadBuilder()
            builder.add_16bit_uint(value)
            payload = builder.to_registers()
            response = self.client.write_registers(address, payload, slave=unit_id)
            if response.isError():
                return self.WRITE_FAIL_CODE
            return self.SUCCESS_CODE
        except Exception as e:
            log.error(f"ModbusService::write_holding_registers - {e}")
            return self.EXEC_FAIL_CODE

    def read_holding_registers(self, address, quantity, unit_id):
        try:
            response = self.client.read_holding_registers(address, quantity, slave=unit_id)
            if response.isError():
                return None
            return response.registers
        except Exception as e:
            log.error(f"ModbusService::read_holding_registers - {e}")
            return None

    def reset(self, unit_id):
        self.write_holding_registers(160, 33, unit_id)

    def reset_param(self, unit_id):
        return self.write_holding_registers(160, 22, unit_id)

    def reset_mode(self, unit_id):
        return self.write_holding_registers(160, 44, unit_id)

    def read_slave_address(self, unit_id):
        return self.read_holding_registers(162, 1, unit_id)[0]

    def write_slave_address(self, value, unit_id):
        return self.write_holding_registers(162, value, unit_id)

    def read_run_mode(self, unit_id):
        return self.read_holding_registers(161, 1, unit_id)[0]

    def write_run_mode(self, value, unit_id):
        return self.write_holding_registers(161, value, unit_id)

    def read_baud_rate(self, unit_id):
        return self.read_holding_registers(163, 1, unit_id)[0]

    def write_baud_rate(self, value, unit_id):
        return self.write_holding_registers(163, value, unit_id)

    @staticmethod
    def long_to_bytes(value):
        return value.to_bytes(8, byteorder='big')

    @staticmethod
    def bytes_to_long(bytes_value):
        return int.from_bytes(bytes_value, byteorder='big')


# Example usage
if __name__ == "__main__":
    service = ModbusService("127.0.0.1", 502)
    try:
        print(service.read_data(1))
    finally:
        service.release()
