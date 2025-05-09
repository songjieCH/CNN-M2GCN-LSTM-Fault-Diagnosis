import logging
from kafka import KafkaProducer, KafkaConsumer, KafkaAdminClient
from modbus import ModbusService
import config

logging.basicConfig()
log = logging.getLogger()
log.setLevel(logging.DEBUG)

class Producer:
    def __init__(self):
        self.producer = KafkaProducer(bootstrap_servers=config.kafka_bootstrap_servers)
        if not self.producer.bootstrap_connected():
            log.error("Kafka producer connection failed. Link address: %s", config.kafka_bootstrap_servers)

        self.modbus = ModbusService(config.modbus_ip, config.modbus_port)
        self.topic = config.topic

    def send_data(self, unit_id=1):
        data = self.modbus.read_data(unit_id)
        self.send_sync(config.topic + str(unit_id), data)

    def send_sync(self, topic, msg_body):
        future = self.producer.send(topic, value=msg_body.encode('utf-8'))
        try:
            future.get(timeout=10)
        except Exception as e:
            log.error("send message failed", e)

    def send_async(self, topic, msg_body, success_callback=None, exception_callback=None):
        future = self.producer.send(topic, value=msg_body.encode('utf-8'))
        future.add_callback(success_callback)
        future.add_errback(exception_callback)


class Consumer:
    def __init__(self):
        self.consumer = KafkaConsumer(
            bootstrap_servers=config.kafka_bootstrap_servers,
            auto_offset_reset='earliest',
        )
        self.message_listener = None

    def subscribe(self, topic):
        try:
            self.consumer.subscribe([topic])
        except Exception as e:
            log.error("subscribe error", e)

    def register_message_listener(self, listener):
        self.message_listener = listener

    def start(self):
        try:
            for message in self.consumer:
                self.message_listener([message])
        except Exception as e:
            log.error("start error", e)

    def shutdown(self):
        self.consumer.close()

class ConsumerFactory:
    def __init__(self):
        self.consumers = {}

    def create_consumer(self, name):
        consumer_name = f"consumer-{name}"
        consumer = Consumer()
        self.consumers[consumer_name] = consumer
        return consumer

    def get_consumer(self, consumer_name):
        return self.consumers.get(consumer_name)

    def list_consumers(self):
        return list(self.consumers.keys())

    def shutdown(self):
        for consumer in self.consumers.values():
            consumer.shutdown()


