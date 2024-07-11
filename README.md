<h1 align="center">LiteNet</h1>

<p align="center">LiteNet is a neural network framework written in C++ with 0 external dependencies. It is designed to be easy to use, with a simple and intuitive API based on the Keras and PyTorch libraries. All linear algebra operations are implemented from scratch with the C++ standard library.</p>

⚠ **This project is solely for educational purposes. It is not intended to be used in production.**

## Motivation

> Why make a neural network framework? Why in C++? Why no dependencies?

I wanted to learn more about neural networks and how they work under the hood. I also wanted to learn more about C++ and how to write efficient code. I decided to write a neural network framework in C++ with no dependencies to learn more about both of these topics.

## Example Usage

For the full example, see [src/example_mnist.cpp](src/example_mnist.cpp).

```cpp
// Build
litenet::Model model;
model.add(std::make_unique<litenet::layers::Dense>(784, 32, "relu", std::make_unique<litenet::initializers::HeUniform>()));
model.add(std::make_unique<litenet::layers::Dense>(32, 10, "softmax", std::make_unique<litenet::initializers::GlorotUniform>()));
model.compile("categorical_crossentropy", "sgd", 0.01);

// Train
model.fit(inputs, targets, 500, 1);

// Predict
litenet::Matrix predictions = model.predict(inputs);
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
- [ ] Optimizers
  - [x] Stochastic Gradient Descent
  - [ ] Adam
  - [ ] RMSprop
  - [ ] Adagrad
- [ ] Metrics
  - [ ] Accuracy
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
