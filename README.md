# Multi-Layer Neural Network

## Overview

This project implements a simple multi-layer neural network that can recognize hand-drawn symbols. Users draw symbols using a graphical interface, and the neural network learns to classify them based on their shape.

## How It Works

### Drawing and Input Collection

Users draw symbols on a canvas. Each symbol is represented by a sequence of points (x, y coordinates). These points capture the path of the drawing.

### Preprocessing

Before training, the drawn points are converted into a fixed-size input vector:
- The number of points is reduced to a fixed length.
- Direction vectors are calculated between points.
- Vectors are normalized to ensure consistent input.

This transforms each symbol into a numerical vector that can be used as input for the neural network.

### Labeling

Each symbol is labeled with a character (e.g., "A", "B", "C"). These labels are converted into one-hot encoded vectors, which serve as the expected outputs during training.

## Neural Network Structure

The neural network has:
- An input layer based on the size of the vector (2 * number of directions).
- One hidden layer with a customizable number of neurons.
- An output layer where each neuron represents a possible label/class.

The activation function used is the **sigmoid** function, which outputs values between 0 and 1.

## Training Process

Training uses **forward propagation** and **backpropagation**:

- **Forward propagation**: Calculates predictions based on current weights.
- **Backpropagation**: Adjusts the weights to reduce the error between predictions and actual labels.

The process repeats over multiple **epochs** (training cycles), or until the average error falls below a set **error threshold**.

## Key Concepts

- **Epoch**: One full pass over the entire training dataset.
- **Error threshold**: A limit that defines when training can stop early if the average error becomes low enough.
- **Sigmoid function**: Maps values to the range [0, 1], allowing us to interpret outputs as probabilities.
- **One-hot encoding**: A method of representing class labels as binary vectors.

## User Interface

The graphical interface allows:
- Drawing symbols
- Labeling them
- Starting training
- Testing recognition on new drawings

An error graph can be displayed to visualize training progress.

## Notes

- The use of `.copy()` when saving points ensures that the original drawing is preserved without accidental modification.
- Direction vectors capture the movement between points and are useful for shape-based recognition, regardless of drawing size or speed.
