import json
import time

def insert_text_into_json(file_path, insert_line, insert_content, repeat=1):
    """
    Insert text content at specified line numbers in a JSON file, can be inserted multiple times.
    And count the time spent reading and writing files.

    :param file_path: str - JSON file path
    :param insert_line: int - line number to insert (from 0)
    :param insert_content: str - text content to insert
    :param repeat: int - number of times to insert (default 1 time)
    """
    # count reading time
    start_read = time.time()
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    end_read = time.time()
    read_duration = end_read - start_read
    print(f"📥 file reading time: {read_duration:.2f} seconds")

    # check if insert line is valid
    if insert_line < 0 or insert_line > len(lines):
        print("❌ insert line out of range")
        return

    # generate insert content and insert it specified times
    insert_text = (insert_content + "\n") * repeat
    lines.insert(insert_line, insert_text)

    # count writing time
    start_write = time.time()
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(lines)
    end_write = time.time()
    write_duration = end_write - start_write
    print(f"📤 file writing time: {write_duration:.2f} seconds")

    print(f"✅ inserted {repeat} times content at line {insert_line}")

# example usage
file_path = "ta1-trace-e3-official-1.json.3p"  # your JSON file path
insert_line = 1422842  # specify the line number to insert (from 0)
insert_content = '''{"datum":{"com.bbn.tc.schema.avro.cdm18.FileObject":{"uuid":"93E0C854-95A4-FE1B-245B-6E217A409764","baseObject":{"hostId":"E621F964-5A66-0F89-30E0-67ADB2A5EC28","permission":{"com.bbn.tc.schema.avro.cdm18.SHORT":"0000"},"epoch":{"int":0},"properties":{"map":{"path":"/tmp/ztmp"}}},"type":"FILE_OBJECT_FILE","fileDescriptor":null,"localPrincipal":null,"size":null,"peInfo":null,"hashes":null}},"CDMVersion":"18","source":"SOURCE_LINUX_SYSCALL_TRACE"}
{"datum":{"com.bbn.tc.schema.avro.cdm18.Event":{"uuid":"E9819DDD-BC4C-0025-DD01-63160D1BD191","sequence":{"long":119044881},"type":"EVENT_CREATE_OBJECT","threadId":{"int":19428},"hostId":"E621F964-5A66-0F89-30E0-67ADB2A5EC28","subject":{"com.bbn.tc.schema.avro.cdm18.UUID":"59169A99-4C73-E5E0-1E25-DB232BA80F32"},"predicateObject":{"com.bbn.tc.schema.avro.cdm18.UUID":"93E0C854-95A4-FE1B-245B-6E217A409764"},"predicateObjectPath":{"string":"/tmp/ztmp"},"predicateObject2":null,"predicateObject2Path":null,"timestampNanos":1523637968165000000,"name":null,"parameters":null,"location":null,"size":null,"programPoint":null,"properties":{"map":{"mode":"0","flags":"O_RDWR|O_CREAT"}}}},"CDMVersion":"18","source":"SOURCE_LINUX_SYSCALL_TRACE"}
{"datum":{"com.bbn.tc.schema.avro.cdm18.Event":{"uuid":"0320414A-E82F-97F0-9496-B4259661CFC6","sequence":{"long":119044882},"type":"EVENT_WRITE","threadId":{"int":19428},"hostId":"E621F964-5A66-0F89-30E0-67ADB2A5EC28","subject":{"com.bbn.tc.schema.avro.cdm18.UUID":"59169A99-4C73-E5E0-1E25-DB232BA80F32"},"predicateObject":{"com.bbn.tc.schema.avro.cdm18.UUID":"93E0C854-95A4-FE1B-245B-6E217A409764"},"predicateObjectPath":{"string":"/tmp/ztmp"},"predicateObject2":null,"predicateObject2Path":null,"timestampNanos":1523637968165000000,"name":null,"parameters":null,"location":null,"size":{"long":95015},"programPoint":null,"properties":{"map":{}}}},"CDMVersion":"18","source":"SOURCE_LINUX_SYSCALL_TRACE"}
{"datum":{"com.bbn.tc.schema.avro.cdm18.FileObject":{"uuid":"96FE4223-D38F-9D49-C00F-D51954FA7DD4","baseObject":{"hostId":"E621F964-5A66-0F89-30E0-67ADB2A5EC28","permission":{"com.bbn.tc.schema.avro.cdm18.SHORT":"01FF"},"epoch":{"int":0},"properties":{"map":{"path":"/tmp/ztmp"}}},"type":"FILE_OBJECT_FILE","fileDescriptor":null,"localPrincipal":null,"size":null,"peInfo":null,"hashes":null}},"CDMVersion":"18","source":"SOURCE_LINUX_SYSCALL_TRACE"}
{"datum":{"com.bbn.tc.schema.avro.cdm18.Event":{"uuid":"98E85C2E-5A67-94F0-3269-01F93D39532B","sequence":{"long":119044883},"type":"EVENT_UPDATE","threadId":{"int":19428},"hostId":"E621F964-5A66-0F89-30E0-67ADB2A5EC28","subject":{"com.bbn.tc.schema.avro.cdm18.UUID":"59169A99-4C73-E5E0-1E25-DB232BA80F32"},"predicateObject":{"com.bbn.tc.schema.avro.cdm18.UUID":"93E0C854-95A4-FE1B-245B-6E217A409764"},"predicateObjectPath":{"string":"/tmp/ztmp"},"predicateObject2":{"com.bbn.tc.schema.avro.cdm18.UUID":"96FE4223-D38F-9D49-C00F-D51954FA7DD4"},"predicateObject2Path":{"string":"/tmp/ztmp"},"timestampNanos":1523637968165000000,"name":null,"parameters":null,"location":null,"size":null,"programPoint":null,"properties":{"map":{}}}},"CDMVersion":"18","source":"SOURCE_LINUX_SYSCALL_TRACE"}'''
repeat = 5  # number of times to insert

insert_text_into_json(file_path, insert_line, insert_content, repeat)
