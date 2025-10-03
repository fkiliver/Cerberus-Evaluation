def count_keywords_in_file(file_path, keywords):
    keyword_counts = {keyword: 0 for keyword in keywords}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            for keyword in keywords:
                if keyword in line:
                    keyword_counts[keyword] += 1
    total = sum(keyword_counts.values())
    return keyword_counts, total

# Example usage
file_path = "darpa_tc_e3_cleaned.txt"  # Replace with your file path
keywords = ["subject: thunderbird;", "subject: chmod;"]  # Replace with your keywords
# keywords = ["subject: ztmp;", "subject: firefox;", "subject: pass_mgr;", "subject: gtcache;", "subject: sh;"]  # Replace with your keywords

result, total_count = count_keywords_in_file(file_path, keywords)

# Output results
for keyword, count in result.items():
    print(f"Number of lines containing keyword '{keyword}': {count}")
print(f"Total number of lines matching all keywords (may have duplicates): {total_count}")