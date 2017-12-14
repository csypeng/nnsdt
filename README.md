# nnsdt
Implementation of Hinton's recent paper "Distilling a Neural Network Into a Soft Decision Tree".
This paper is to interprete the result of a neuron network by ONE soft decision tree. The soft decision tree
is trained using the prediction result of the neuron network.


Remark:

1. Current version of the code just supports the binary classification problem.

2. The parameter w of each node in the soft decision tree is clip_by_value in range (1e-10, 1000.0). Otherwise,
w can be nan after about 50 epochs on the dataset I tried. This essientially restricts the search space of w and may produce a local optimal. 