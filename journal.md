## How is chords without third handled (power chords, sus2 and sus4)?
- **Outlier analysis and drop these?**
    - *Set some trust boundary to chroma values and drop all where some combination of 1, 3 and 5 tritone's chroma values are not sufficient.*
    - *Not enough data?*
- **Count them in as minor or major chords**
    - *Would not make sense musically but since this is a binary classification, it would make sense.*

## Technical implementation
- **For a binary classification task, a logistic regression or SVM might be a good enough choice, but it is pretty boring**
    - *Still, fancyness is not evaluated*
- **Simple Feed Forward NN**
    - *Might exceed the 2-3 hours*
    - *The data space is quite complex and has a lot of curvature due to the additional chroma noise. NN would be a great approximator.*
- **BERT would be a good choice**
    - *Encoder model would fit into the noise produced by the unwanted notes quite well.*


## During testing
- **Accuracy reduces if the sigmoid in the output layer**
    - *Data is linearly separable*