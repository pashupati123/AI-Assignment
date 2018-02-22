* The model is a simple neural network with one hidden layer with the same number of neurons as there are inputs (784).A rectifier activation function is used for the neurons in the hidden layer.

* A softmax activation function is used on the output layer to turn the outputs into probability-like values and allow one class of the 10 to be selected as the model’s output prediction. Logarithmic loss is used as the loss function (called categorical_crossentropy in Keras) and the efficient ADAM gradient descent algorithm is used to learn the weights.

* We can now fit and evaluate the model. The model is fit over 25 epochs with updates every 200 images. The test data is used as the validation dataset, allowing you to see the skill of the model as it trains. A verbose value of 2 is used to reduce the output to one line for each training epoch.

* Finally, the test dataset is used to evaluate the model and a classification error rate along with Plot of model Accuracy vs epoch and model Loss vs epoch is printed.
