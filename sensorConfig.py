import struct

class SensorConfig:
    IP = "192.168.1.164"
    PORT = 502
    collect_interval = 100
    sensor_unit_ids = 1  # Sensor unitId
    key = f"{IP}:"

    # Modbus register table
    register_table = {
        0: "Temp",
        1: "X-Vel", 2: "Y-Vel", 3: "Z-Vel",
        4: "X-Acc", 5: "Y-Acc", 6: "Z-Acc",
        7: "X-Vel-Max", 8: "X-VEl-Kur", 9: "X-Acc-Max", 10: "X-Acc-Kur",
        11: "Y-Vel-Max", 12: "Y-VEl-Kur", 13: "Y-Acc-Max", 14: "Y-Acc-Kur",
        15: "Z-Vel-Max", 16: "Z-VEl-Kur", 17: "Z-Acc-Max", 18: "Z-Acc-Kur",
        19: "X-Dis", 20: "Y-Dis", 21: "Z-Dis"
    }

    # Modbus register parameters
    register_param = {
        160: "save param table",
        161: "run mode",
        162: "slave address",
        163: "baud rate"
    }

    @staticmethod
    def to_map(data):
        map = {}
        if data is None:
            return map
        map["Temp"] = struct.unpack('>h', data[0:2])[0] / 100.0
        map["X-Vel"] = struct.unpack('>h', data[2:4])[0] / 100.0
        map["Y-Vel"] = struct.unpack('>h', data[4:6])[0] / 100.0
        map["Z-Vel"] = struct.unpack('>h', data[6:8])[0] / 100.0
        map["X-Acc"] = struct.unpack('>h', data[8:10])[0] / 100.0
        map["Y-Acc"] = struct.unpack('>h', data[10:12])[0] / 100.0
        map["Z-Acc"] = struct.unpack('>h', data[12:14])[0] / 100.0
        map["X-Dis"] = struct.unpack('>h', data[38:40])[0]
        map["Y-Dis"] = struct.unpack('>h', data[40:42])[0]
        map["Z-Dis"] = struct.unpack('>h', data[42:44])[0]
        return map