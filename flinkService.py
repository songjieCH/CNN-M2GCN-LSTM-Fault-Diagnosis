import logging

from pyflink.common import SimpleStringSchema
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.kafka import FlinkKafkaConsumer
from pyflink.datastream.functions import ProcessWindowFunction, ProcessFunction, KeyedProcessFunction
from pyflink.common.typeinfo import Types
from pyflink.datastream.window import GlobalWindows, CountTrigger
import config

logging.basicConfig()
log = logging.getLogger()
log.setLevel(logging.DEBUG)

class FlinkProcess:
    def __init__(self):
        self.env = StreamExecutionEnvironment.get_execution_environment()
        self.env.set_parallelism(config.sensor_unit_ids)

    def run(self):
        # 创建并且合并数据流
        data_stream = None
        for unit_id in range(config.sensor_unit_ids):
            kafka_consumer = FlinkKafkaConsumer(
                topics=config.topic + str(unit_id),
                deserialization_schema=SimpleStringSchema(),
                properties={
                    'bootstrap.servers': config.kafka_bootstrap_servers,
                }
            )
            if data_stream is None:
                data_stream = self.env.add_source(kafka_consumer)
            else:
                data_stream = data_stream.union(self.env.add_source(kafka_consumer))

        assert data_stream is not None

        # 保存数据至数据库
        save_stream = data_stream \
            .process(self.save_func()) \

        # 实时处理
        realtime_stream = data_stream \
            .key_by(self.extract_id, key_type=Types.INT()) \
            .process(self.realtime_func())

        # 预处理
        batch_stream  = data_stream \
            .key_by(self.extract_id, key_type=Types.INT()) \
            .window(GlobalWindows.create()) \
            .trigger(CountTrigger.of(config.data_len)) \
            .process(self.batch_func())

        try:
            self.env.execute("Flink Job")
        except Exception as e:
            log.warning("Flink Job execute failed: %s", e)

    def extract_id(self, data):
        return data[0]

    def save_func(self):
        pass

    def realtime_func(self):
        pass

    def batch_func(self):
        pass