# Neural-Network-Dependency-Parser
Predicts the syntactic dependency of a sentence using a Keras/Tensorflow enabled Neural Net

# Use:
To evaluate the existing model (data/model.h5):
The evaluation function can be called like this: "python evaluate.py data/model.h5 data/dev.conll". This will show the Labeled and Unlabeled attachment scores. Labeled attachment score is the percentage of correct (parent, relation, child) predictions. Unlabeled attachment score is the percentage of correct (parent, child) predictions.

To generate the Neural Network from the ground up:
1. To extract data into a usable form, use: "python extract_training_data.py data/train.conll data/input_train.npy data/target_train.npy".
2. To train the model, call the program like so: "python train_model.py data/input_train.npy data/target_train.npy data/model.h5". This will take a substantial amount of time as the Neural Net is being retrained.
