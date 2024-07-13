<h1 align="center">LiteNet</h1>

<p align="center">LiteNet is a neural network framework written in C++ with 0 external dependencies. It is designed to be easy to use, with a simple and intuitive API based on the Keras    and PyTorch libraries. All linear algebra operations are implemented from scratch with the C++ standard library.</p>

⚠ **This project is solely for educational purposes. It is not intended to be used in production.**

## Motivation

> Why make a neural network framework? Why in C++? Why no dependencies?

I wanted to learn more about neural networks and how they work under the hood. I also wanted to learn more about C++ and how to write efficient code. I decided to write a neural network framework in C++ with no dependencies to learn more about both of these topics.

## Example Usage (handwritten digit recognition)

For the full example, see [src/example_mnist.cpp](src/example_mnist.cpp).

```cpp
// Build
litenet::Model model;
model.add(std::make_unique<litenet::layers::Dense>(784, 64, "sigmoid", std::make_unique<litenet::initializers::GlorotUniform>()));
model.add(std::make_unique<litenet::layers::Dense>(64, 10, "sigmoid", std::make_unique<litenet::initializers::GlorotUniform>()));
model.compile("mean_squared_error", std::make_unique<litenet::optimizers::Adam>(0.01));

// Train
model.fit(inputs, targets, 30, 64, validationInputs, validationTargets);

// Predict
litenet::Matrix predictions = model.predict(inputs);

// Evaluate
std::vector<double> results = model.evaluate(validationInputs, validationTargets);
std::cout << "Loss: " << results[0] << std::endl;
std::cout << "Accuracy: " << results[1] << std::endl;
```

Results: (on MNIST dataset, 5000 training samples, 100 validation samples)

```
Epoch 1 | loss: 0.256219 | val_loss: 0.278005
Epoch 2 | loss: 0.2069 | val_loss: 0.249769
Epoch 3 | loss: 0.174791 | val_loss: 0.228139
...
Epoch 30 | loss: 0.0256604 | val_loss: 0.0824118

Loss: 0.0824118
Accuracy: 0.94
```

## Features

- [ ] Layers
  - [x] Dense
  - [ ] Conv2D
  - [ ] MaxPooling2D
  - [ ] Flatten
  - [ ] Dropout
- [x] Activation Functions
  - [x] ReLU
  - [x] Leaky ReLU
  - [x] Sigmoid
  - [x] Tanh
  - [x] Softmax
- [x] Loss Functions
  - [x] Mean Squared Error
  - [x] Mean Absolute Error
  - [x] Binary Cross Entropy
  - [x] Categorical Cross Entropy
- [x] Optimizers
  - [x] Stochastic Gradient Descent (mini-batch)
  - [x] Adam
  - [x] AdamW
  - [x] RMSprop
  - [x] Adagrad
- [ ] Metrics
  - [x] Accuracy
  - [ ] Precision
  - [ ] Recall
  - [ ] F1 Score
- [x] Initializers
  - [x] Random Normal
  - [x] Random Uniform
  - [x] Glorot Normal
  - [x] Glorot Uniform
  - [x] He Normal
  - [x] He Uniform
  - [x] Zeros
  - [x] Ones
- [ ] Regularizers
  - [ ] L1
  - [ ] L2
