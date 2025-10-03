import re

# 读取日志文件
def read_log_file(file_path):
    with open(file_path, 'r') as file:
        return file.readlines()

# 提取subject值并统计行数
def extract_subjects_and_count(log_lines):
    subjects = set()  # 使用set自动去重
    count = 0  # 统计符合条件的行数
    for line in log_lines:
        # 检查该行是否包含label=1
        if 'label: 1' in line:
            count += 1  # 增加符合条件的行数
            # 输出每行，帮助调试
            # print(f"Found line with label 1: {line.strip()}")
            # 使用正则表达式提取subject的值
            match = re.search(r'subject:\s*(\S+)', line)
            if match:
                subjects.add(match.group(1))  # 将subject添加到集合中
            else:
                print(f"Failed to find subject in line: {line.strip()}")
    return subjects, count

def main():
    file_path = 'ta1-trace-e3-official-1-04.txt'  # 替换为日志文件的路径
    log_lines = read_log_file(file_path)
    subjects, count = extract_subjects_and_count(log_lines)

    if subjects:
        print("\n不重复的subject值:")
        for subject in subjects:
            print(subject)
    else:
        print("没有找到符合条件的subject值")
    
    print(f"\n符合条件的行数: {count}")

if __name__ == '__main__':
    main()
