

                         ~= Yousician =~

      Machine Learning/AI Software Developer (Python)


Use Python programming language to train a method for recognizing major
and minor chords in music recordings. Both chord types have 12 possible
root notes (A, A#, ..., G#), so the total number of target classes is 24.
 
File chord_data.csv contains a dataset for the task. It consists of
12-dimensional input features and the corresponding target classes for
each input vector.  Details of the data are explained below.
 
Your task is to train a model that takes a feature vector as input and
produces an estimate of the target class. You can use neural networks
or any other model of your choice. Please use some existing libraries
and tools instead of reimplementing yourself!

After training your model, calculate average classification rate on
the test set. Use 70% of the data for training, and the remaining 30%
for testing.
 
Your work will not be judged based on the achieved recognition rate,
so you won’t need to seek for an optimal model configuration. Rather,
we'd like to see clean and compact code and solution.
You can also mention why you chose the model you did, and what you
would do next to improve the system.

You shouldn’t spend more than 2-3 hours on the task. Good luck!


                  Description of the data
 
The data is courtesy of the McGill Billboard dataset and the feature
vectors have been computed using the Chordino toolbox.

Each line in the CSV file represents a short segment of music where
one chord is playing. Each line contains 14 values. The first 2 define
the target class, and the remaining 12 contain values of the input
feature vector:
- #1 value: Target chord root note, values from 0 to 11 are
   interpreted as follows: 0 = A, 1 = A#, 2 = B, ..., 11 = G#
- #2 value: Target chord type: 0 = major chord, 1 = minor chord
- #3 - #14: Input features, an array of length 12. The 12 values
   represent so-called chroma features. The first value describes
   the amount of spectral energy falling on pitch class A, the second
   for pitch class A#, and so forth, up to G#.
