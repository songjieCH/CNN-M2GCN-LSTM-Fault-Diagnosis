import os
import tarfile
import urllib.request
import subprocess

def download_kafka(version='3.0.0', scala_version='2.13', download_dir='.'):
    url = f"https://downloads.apache.org/kafka/{version}/kafka_{scala_version}-{version}.tgz"
    file_name = os.path.join(download_dir, f"kafka_{scala_version}-{version}.tgz")
    urllib.request.urlretrieve(url, file_name)
    with tarfile.open(file_name, "r:gz") as tar:
        tar.extractall(path=download_dir)
    os.remove(file_name)
    return os.path.join(download_dir, f"kafka_{scala_version}-{version}")

def configure_zookeeper(kafka_dir):
    zookeeper_config = os.path.join(kafka_dir, 'config', 'zookeeper.properties')
    with open(zookeeper_config, 'a') as f:
        f.write("\n# Custom Zookeeper Configurations\n")
    return zookeeper_config

def configure_kafka_broker(kafka_dir):
    server_config = os.path.join(kafka_dir, 'config', 'server.properties')
    with open(server_config, 'a') as f:
        f.write("\n# Custom Kafka Broker Configurations\n")
        f.write("broker.id=0\n")
        f.write("log.dirs=/tmp/kafka-logs\n")
        f.write("zookeeper.connect=localhost:2181\n")
    return server_config

def start_zookeeper(kafka_dir, zookeeper_config):
    zookeeper_cmd = os.path.join(kafka_dir, 'bin', 'zookeeper-server-start.sh')
    subprocess.Popen([zookeeper_cmd, zookeeper_config])

def start_kafka_broker(kafka_dir, server_config):
    kafka_cmd = os.path.join(kafka_dir, 'bin', 'kafka-server-start.sh')
    subprocess.Popen([kafka_cmd, server_config])


if __name__ == "__main__":
    kafka_dir = download_kafka(download_dir='./EnvApp')

    zookeeper_config = configure_zookeeper(kafka_dir)
    server_config = configure_kafka_broker(kafka_dir)

    start_zookeeper(kafka_dir, zookeeper_config)
    start_kafka_broker(kafka_dir, server_config)