import pandas as pd

data_file_path = "data/chord_data.csv"
df = pd.read_csv(data_file_path, header=None)

column_names = [
    "chord_root_note", "chord_type",
    "Chroma_A", "Chroma_A#", "Chroma_B", "Chroma_C", "Chroma_C#", "Chroma_D", 
    "Chroma_D#", "Chroma_E", "Chroma_F", "Chroma_F#", "Chroma_G", "Chroma_G#"
]
df.columns = column_names

df["combined_label"] = df["chord_root_note"] * 2 + df["chord_type"]

df.drop(["chord_root_note", "chord_type"], axis=1)

output_file = "data/combined_label_chord_data.csv"
df.to_csv(output_file, index=False)

output_file
