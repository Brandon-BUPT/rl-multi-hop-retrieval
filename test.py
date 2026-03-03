import json
from collections import Counter


def load_hotpot(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_gold_distribution(data):
    sp_count_dist = Counter()
    title_counter = Counter()
    sent_id_counter = Counter()
    title_per_sample = Counter()

    for item in data:
        sp = item["supporting_facts"]  # [(title, sent_id), ...]

        # 每个样本 supporting fact 数量
        sp_count_dist[len(sp)] += 1

        titles = set()
        for title, sent_id in sp:
            title_counter[title] += 1
            sent_id_counter[sent_id] += 1
            titles.add(title)

        # 每个样本涉及多少个不同 title
        title_per_sample[len(titles)] += 1

    return sp_count_dist, title_counter, sent_id_counter, title_per_sample


def print_stats(data, sp_count_dist, title_counter, sent_id_counter, title_per_sample):
    total = len(data)

    print("========== HotpotQA Gold Distribution ==========")
    print(f"Total samples: {total}\n")

    print("Supporting Facts per Sample:")
    for k in sorted(sp_count_dist):
        v = sp_count_dist[k]
        print(f"  {k} facts: {v} ({v/total:.2%})")

    print("\nDistinct Titles per Sample:")
    for k in sorted(title_per_sample):
        v = title_per_sample[k]
        print(f"  {k} titles: {v} ({v/total:.2%})")

    print("\nTop 20 Most Frequent Gold Titles:")
    for title, count in title_counter.most_common(20):
        print(f"  {title}: {count}")

    print("\nSentence ID Distribution (Top 20):")
    for sid, count in sent_id_counter.most_common(20):
        print(f"  sent_id {sid}: {count}")


if __name__ == "__main__":
    path = "data/hotpotqa/hotpot_dev_distractor.json"  # 改成你的文件路径
    data = load_hotpot(path)

    sp_count_dist, title_counter, sent_id_counter, title_per_sample = \
        analyze_gold_distribution(data)

    print_stats(data, sp_count_dist, title_counter, sent_id_counter, title_per_sample)