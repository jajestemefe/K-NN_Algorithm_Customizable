# KNN and Perceptron Classifier (Iris Dataset)

C++ implementation of:

- **K-Nearest Neighbors (KNN)** for multiclass Iris classification
- **Single perceptron** for one-vs-rest binary classification (`Iris-setosa` vs `not-Iris-setosa`)

Built for an Artificial Intelligence course project, with no external ML libraries.

## Features

- Load training and test data from text files
- Min-max normalization based on training-set statistics
- KNN evaluation on test split (accuracy output)
- Custom KNN prediction for user-provided sample
- Perceptron train/test mode for target class
- Custom perceptron prediction for user-provided sample
- Input validation for `k` and interactive user values
- Supports decimal values written with comma (e.g. `5,1`)

## Dataset format

Each line:

`attr1 attr2 attr3 attr4 class_label`

Example:

`5,1 3,5 1,4 0,2 Iris-setosa`

## Build and run

### CLion

- Open project
- Build target: `NAI_Project_1`
- Run target: `NAI_Project_1`

### CMake (terminal)

```bash
cmake -S . -B cmake-build-debug
cmake --build cmake-build-debug --target NAI_Project_1
./cmake-build-debug/NAI_Project_1
```

On Windows executable is `NAI_Project_1.exe`.

## Program commands

After launch, type:

- `help` - show command list
- `test` - run KNN on `iris_test.txt`
- `knn` - predict one custom sample with KNN
- `perc` - train/test perceptron (`Iris-setosa` target)
- `cust` - predict one custom sample with perceptron
- `stop` - exit program

## Notes

- KNN uses Euclidean distance.
- `k` must be positive and cannot exceed training sample count.
- Min-max normalization skips a feature when its range is zero.
- This project currently keeps perceptron target label fixed to `Iris-setosa` in command mode.

## Suggested repository name

If you want a cleaner, portfolio-friendly name:

- `iris-knn-perceptron-cpp`
- `knn-perceptron-from-scratch`
- `iris-classification-cpp`
