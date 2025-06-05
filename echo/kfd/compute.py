import pandas as pd
from io import StringIO
import csv


def extended_check_sequence(sequence, sequence_length):
    sequence = [0] + sequence
    sequence_length += 1

    max_diff_area = 0
    start_ED = 0
    ES = 0
    end_ED = sequence_length - 1

    area_total = sum(sequence)

    for i in range(1, sequence_length - 2):
        area_start_ED = sum(sequence[:i])
        for j in range(i + 1, sequence_length - 1):
            area_ES = sum(sequence[:j])
            for k in range(j + 1, sequence_length):
                area_end_ED = sum(sequence[:k])

                diff_area = abs(2 * area_ES - area_start_ED - area_end_ED + area_total)

                if diff_area > max_diff_area:
                    max_diff_area = diff_area
                    start_ED = i
                    ES = j
                    end_ED = k

    return start_ED, ES, end_ED


def main():
    pred_seq_path = "/home/khmuhammad/Echo-Dream/experiments/CardiacASD.csv"
    ED_start_dict = {}
    ED_end_dict = {}
    ES_dict = {}
    with open(pred_seq_path, newline="") as f:
        reader = csv.reader(f)
        mark = 0
        for row in reader:
            mark += 1
            patient_id = row[0]  # ID
            print(patient_id)
            sequence = row[2:]  # Sequence
            sequence_length = len(sequence)
            sequence = [float(x) for x in sequence]
            print(
                f"Processing Patient ID: {patient_id}, Sequence Length: {sequence_length}"
            )
            start_ED, ES, end_ED = extended_check_sequence(sequence, sequence_length)
            ED_start_dict[patient_id] = start_ED
            ES_dict[patient_id] = ES
            ED_end_dict[patient_id] = end_ED

    # Save the results to CSV files
    df = pd.DataFrame(list(ED_start_dict.items()), columns=["FileName", "Start_ED"])
    df["ES"] = df["FileName"].map(ES_dict)
    df["End_ED"] = df["FileName"].map(ED_end_dict)
    df["Start_ED"] = df["FileName"].astype(int)
    df["ES"] = df["ES"].astype(int)
    df["End_ED"] = df["End_ED"].astype(int)
    df.to_csv("/home/khmuhammad/Echo-Dream/experiments/ED_ES_results.csv", index=False)


main()
