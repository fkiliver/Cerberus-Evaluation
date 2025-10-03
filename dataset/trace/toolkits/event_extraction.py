import os
import json
from tqdm import tqdm  # 进度条
from datetime import datetime, timezone, timedelta

# 输入和输出目录
INPUT_DIR = "../"  # 假设 JSON 文件在当前目录
OUTPUT_FILE = "../darpa_tc_e3_cleaned.txt"
BATCH_SIZE = 10000  # 每批处理的行数

# 事件数据结构
class Edge:
    def __init__(self, map_index, timestamp, uuid, event_type, subject, predicate_object):
        self.map_index = map_index
        self.timestamp = timestamp
        self.uuid = uuid
        self.event_type = event_type
        self.subject = subject
        self.predicate_object = predicate_object

# 主体（Subject）和客体（Object）数据结构
class Node:
    def __init__(self, map_index, uuid, node_type, path=None, name=None, memory_address=None, filename=None, remote_address=None, remote_port=None):
        self.map_index = map_index
        self.uuid = uuid
        self.node_type = node_type
        self.path = path
        self.name = name
        self.memory_address = memory_address
        self.filename = filename
        self.remote_address = remote_address
        self.remote_port = remote_port

# 读取 JSON 文件，逐行处理
def load_json_files():
    json_files = [f for f in os.listdir(INPUT_DIR) if f.startswith("ta1-trace-e3-official-1.json")]
    return json_files

# 解析单个 JSON 文件并逐行处理
def parse_and_process_file(file_name, output_file, all_nodes):
    with open(os.path.join(INPUT_DIR, file_name), "r", encoding="utf-8") as f:  # 使用 utf-8 编码读取
        edges = []
        line_index = 0

        print(f"开始处理文件: {file_name}")
        for line in tqdm(f, desc=f"处理 {file_name}", unit="line"):
            try:
                entry = json.loads(line)
                datum = entry.get("datum", {})
                
                # 解析 Event
                if "com.bbn.tc.schema.avro.cdm18.Event" in datum:
                    event = datum["com.bbn.tc.schema.avro.cdm18.Event"]
                    event_type = event.get("type", "")
                    timestamp_nanos = event.get("timestampNanos", 0)
                    uuid = event.get("uuid", "")
                    subject = event.get("subject", {}).get("com.bbn.tc.schema.avro.cdm18.UUID", "")
                    predicate_object = event.get("predicateObject", {}).get("com.bbn.tc.schema.avro.cdm18.UUID", "")

                    # 转换时间戳（纳秒 -> 东部时间）
                    timestamp = datetime.fromtimestamp(timestamp_nanos / 1e9, tz=timezone.utc) - timedelta(hours=4)
                    timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")

                    edges.append(Edge(line_index, timestamp, uuid, event_type, subject, predicate_object))
                
                # 解析 Subject
                elif "com.bbn.tc.schema.avro.cdm18.Subject" in datum:
                    subject = datum["com.bbn.tc.schema.avro.cdm18.Subject"]
                    uuid = subject.get("uuid", "")
                    node_type = "Subject"
                    path = subject.get("properties", {}).get("map", {}).get("path", "")
                    name = subject.get("properties", {}).get("map", {}).get("name", uuid) if not path else path
                    all_nodes[uuid] = Node(line_index, uuid, node_type, path=path, name=name)
                
                # 解析 MemoryObject
                elif "com.bbn.tc.schema.avro.cdm18.MemoryObject" in datum:
                    memory_object = datum["com.bbn.tc.schema.avro.cdm18.MemoryObject"]
                    uuid = memory_object.get("uuid", "")
                    memory_address = memory_object.get("memoryAddress", "")
                    all_nodes[uuid] = Node(line_index, uuid, "MemoryObject", memory_address=memory_address)
                
                # 解析 FileObject
                elif "com.bbn.tc.schema.avro.cdm18.FileObject" in datum:
                    file_object = datum["com.bbn.tc.schema.avro.cdm18.FileObject"]
                    uuid = file_object.get("uuid", "")
                    filename = file_object.get("baseObject", {}).get("properties", {}).get("map", {}).get("filename", "")
                    path = file_object.get("baseObject", {}).get("properties", {}).get("map", {}).get("path", filename)
                    all_nodes[uuid] = Node(line_index, uuid, "FileObject", filename=filename, path=path)
                
                # 解析 NetFlowObject
                elif "com.bbn.tc.schema.avro.cdm18.NetFlowObject" in datum:
                    netflow = datum["com.bbn.tc.schema.avro.cdm18.NetFlowObject"]
                    uuid = netflow.get("uuid", "")
                    remote_address = netflow.get("remoteAddress", "")
                    remote_port = netflow.get("remotePort", "")
                    all_nodes[uuid] = Node(line_index, uuid, "NetFlowObject", remote_address=remote_address, remote_port=remote_port)

                # 解析 IpcObject
                elif "com.bbn.tc.schema.avro.cdm18.IpcObject" in datum:
                    ipc_object = datum["com.bbn.tc.schema.avro.cdm18.IpcObject"]
                    uuid = ipc_object.get("uuid", "")
                    path = ipc_object.get("properties", {}).get("map", {}).get("path", "")
                    all_nodes[uuid] = Node(line_index, uuid, "IpcObject", path=path)

                line_index += 1

                # 每当达到批处理大小时，进行写入并清空缓存
                if line_index % BATCH_SIZE == 0:
                    write_batch_to_file(edges, all_nodes, output_file)
                    edges.clear()

            except json.JSONDecodeError:
                continue  # 跳过解析错误的行

        # 写入剩余部分
        write_batch_to_file(edges, all_nodes, output_file)

# 将当前批次的数据写入到文件
def write_batch_to_file(edges, all_nodes, output_file):
    with open(output_file, "a", encoding="utf-8") as f:
        for edge in edges:
            subject_name = all_nodes.get(edge.subject, Node(0, edge.subject, "Unknown")).name
            object_node = all_nodes.get(edge.predicate_object, Node(0, edge.predicate_object, "Unknown"))
            
            # 过滤掉没有意义的UUID
            if subject_name == edge.subject:
                continue
            if not object_node.name and not object_node.path:
                continue
            
            object_name = object_node.name if object_node.name else object_node.path
            
            # 构造输出格式
            record = (
                f"time: {edge.timestamp}; "
                f"type: {edge.event_type}; "
                f"subject: {subject_name}; "
                f"Object_type: {object_node.node_type}; "
                f"object: {object_name}; "
                f"ori_index: {edge.map_index}; "
                f"label: 0;\n"
            )
            f.write(record)

# 主函数
def main():
    json_files = load_json_files()
    
    # 清空输出文件
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("")  # 清空现有内容
    
    # 存储全局的节点
    all_nodes = {}

    # 逐个处理文件
    for json_file in json_files:
        parse_and_process_file(json_file, OUTPUT_FILE, all_nodes)

    print(f"所有文件处理完成，结果已保存至 {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
