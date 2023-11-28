import pandas as pd

data_file_path = "data/chord_data.csv"

df = pd.read_csv(data_file_path, header=None)

column_names = [
    "Chord Root Note", "Chord Type",
    "Chroma A", "Chroma A#", "Chroma B", "Chroma C", "Chroma C#", "Chroma D", 
    "Chroma D#", "Chroma E", "Chroma F", "Chroma F#", "Chroma G", "Chroma G#"
]

df.columns = column_names

output_file = "data/labeled_chord_data.csv"
df.to_csv(output_file, index=False)

output_file
