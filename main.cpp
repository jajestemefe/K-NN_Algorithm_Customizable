#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <cmath>
#include <valarray>
#include <algorithm>
#include <map>

using namespace std;

struct Sample
{
    vector<double> attributes;
    string label;
};

struct Perceptron
{
    vector<double> weights;
    double threshold;
    double alpha;

    explicit Perceptron(const vector<double> &W ={}, const double threshold = 0, const double alpha = 1)
    {
        weights = W;
        this->threshold = threshold;
        this->alpha = alpha;
    }

    auto deltaRule(const int d, const int y, const vector<double>& inputs)-> void
    {
        for (auto j = 0; j < inputs.size(); j++)
        {
            this->weights.at(j) += (d - y) * this->alpha * inputs.at(j);
        }
    }

    [[nodiscard]] auto compute(const vector<double>& inputs) const -> int
    {
        auto y = 0;

        double wTX = 0;
        for (auto j = 0; j < this->weights.size(); j++)
        {
            wTX += this->weights.at(j) * inputs.at(j);
        }

        y = (wTX >= this->threshold) ? 1 : 0;

        return y;
    }

    auto train(const vector<Sample>& trainingValues, const string& targetLabel)->void
    {
        auto correctPrediction = 0;
        auto counter = 0;

        while(correctPrediction < trainingValues.size() && counter < 500)
        {
            for (const auto & trainingValue : trainingValues)
            {
                const auto d = (trainingValue.label == targetLabel) ? 1 : 0;
                vector<double> inputs = trainingValue.attributes;

                const auto y = this->compute(inputs);

                if (d == y)correctPrediction++;
                else
                {
                    this->deltaRule(d,y,inputs);
                    correctPrediction = 0;
                }
            }
            counter++;
        }
    }

    [[nodiscard]] auto test(const vector<Sample>& testingValues, const string& targetLabel) const-> int
    {
        auto counter = 0;
        for (const auto& values : testingValues)
        {
            const int expected = (values.label == targetLabel) ? 1 : 0;
            const int predicted = this->compute(values.attributes);

            if (predicted == expected)
            {
                counter++;
            }
        }
        return counter;
    }
};

auto getSamples(vector<Sample>& dataset, ifstream& file)-> void
{
    string line;

    while (getline(file, line))
    {
        vector<string> row;
        stringstream ss(line);
        string token;

        while (ss >> token)
        {
            row.push_back(token);
        }

        Sample sample;

        for (auto i = 0; i < row.size()-1; i++)
        {
            sample.attributes.push_back(stod(row.at(i)));
        }
        sample.label = row.back();

        dataset.push_back(sample);
    }
}

auto getDistance(const vector<double>& first, const vector<double>& second) -> double
{
    if (first.size() != second.size()) return -1;

    double distance = 0;

    for (auto i = 0; i < first.size(); i++)
    {
        distance += pow((first[i]) - (second[i]), 2);
    }
    distance = sqrt(distance);
    return distance;
}

auto normalize(vector<Sample>& values, const vector<Sample>& trainingValues)-> void
{
    const auto datasize = trainingValues.at(0).attributes.size();
    for (auto i = 0; i < datasize; i++)
    {
        auto min = trainingValues.front().attributes.at(i);
        auto max = trainingValues.front().attributes.at(i);

        for (const auto &[attributes, label] : trainingValues)
        {
            if (min > attributes.at(i)) min = attributes.at(i);
            if (max < attributes.at(i)) max = attributes.at(i);
        }

        for (auto& [attributes, label] : values)
        {
            attributes.at(i) -= min;
            attributes.at(i) /= (max - min);
        }
    }
}

auto getPrediction(const int& k, const vector<double>& test, const vector<Sample>& train)-> string
{
    vector<vector<double>> distancesWithIndices;

    for (auto i = 0; i < train.size(); i++)
    {
        distancesWithIndices.push_back({getDistance(test, train.at(i).attributes), static_cast<double>(i)});
    }

    ranges::sort(distancesWithIndices, [](const vector<double>& a, const vector<double>& b)
    {
        return a[0] < b[0];
    });

    distancesWithIndices.resize(k);

    map<string, int> decisionMap;
    for (auto i = 0; i < k; i++)
    {
        decisionMap[train.at(static_cast<int>(distancesWithIndices.at(i).at(1))).label]++;
    }

    string prediction;
    auto max = 0;

    for (const auto& [key, value] : decisionMap)
    {
        if (value > max)
        {
            max = value;
            prediction = key;
        }
    }

    return prediction;
}

auto getK()-> int
{
    int k;

    cout << "Please provide the 'k' value\n";
    cout << "\n>>>";
    while (!(cin >> k) || k <= 0)
    {
        cout << "'k' value must be a positive integer\n";
        cout << ">>>";
        cin.clear();
        cin.ignore(numeric_limits<streamsize>::max(),'\n');
    }
    return k;
}

auto computeKNN(vector<Sample> trainingValues, const string& testingPath)-> void
{
    const int k = getK();

    ifstream testingFile(testingPath);
    vector<Sample> testingValues;
    getSamples(testingValues, testingFile);

    normalize(testingValues, trainingValues);
    normalize(trainingValues,trainingValues);

    auto count = 0;

    for (const auto &[attributes, label] : testingValues)
    {
        string prediction = getPrediction(k, attributes, trainingValues);

        if (label == prediction)
        {
            count++;
        }
    }

    const auto accuracy = static_cast<float>(count * 100. / static_cast<float>(testingValues.size()));
    cout << count << "/" << testingValues.size() << " correct result" << endl;
    cout << "ACCURACY: %" << accuracy << endl;
}

auto userKNN(vector<Sample> trainingValues)-> void
{
    vector<Sample> userValues = {{}};
    double input;

    for (auto i = 0; i <= trainingValues.at(0).attributes.size(); i++)
    {
        cout << "User attributes[";
        for (auto j = 0; j < trainingValues.at(0).attributes.size(); j++)
        {
            if (j == i) cout << "( ? )";
            else if (userValues.at(0).attributes.size() > j) cout << "( " << userValues.at(0).attributes.at(j) << " )";
            else cout << "( X )";
        }
        cout << "]" << endl;
        if (userValues.at(0).attributes.size() == trainingValues.at(0).attributes.size())continue;
        cout << ">>>";

        while (!(cin >> input) || input <= 0.)
        {
            cout << "All attributes must be positive numerical values\n";
            cout << ">>>";
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
        }

        userValues.at(0).attributes.push_back(static_cast<double>(input));
    }

    userValues.at(0).label = "unknown";

    normalize(userValues, trainingValues);
    normalize(trainingValues, trainingValues);

    const int k = getK();

    const string prediction = getPrediction(k, userValues.at(0).attributes, trainingValues);
    cout << prediction << endl;
}

auto perceptron(vector<Sample> trainingValues, const string& targetLabel)-> void
{
    ifstream testingFile("../iris_test.txt");
    vector<Sample> testingValues;
    getSamples(testingValues, testingFile);

    normalize(testingValues, trainingValues);
    normalize(trainingValues, trainingValues);

    Perceptron perceptron;
    perceptron.weights.resize(trainingValues.at(0).attributes.size());

    perceptron.train(trainingValues, targetLabel);

    cout << "the perceptron is trained, performing test\n" << endl;

    auto counter = perceptron.test(testingValues, targetLabel);

    const auto accuracy = static_cast<float>(counter * 100. / static_cast<float>(testingValues.size()));
    cout << counter << "/" << testingValues.size() << " correct result" << endl;
    cout << "ACCURACY: %" << accuracy << endl;
}

auto userPerceptron(vector<Sample> trainingValues, const string& targetLabel)-> void
{
    vector<Sample> userValues = {{}};
    double input;

    for (auto i = 0; i <= trainingValues.at(0).attributes.size(); i++)
    {
        cout << "User attributes[";
        for (auto j = 0; j < trainingValues.at(0).attributes.size(); j++)
        {
            if (j == i) cout << "( ? )";
            else if (userValues.at(0).attributes.size() > j) cout << "( " << userValues.at(0).attributes.at(j) << " )";
            else cout << "( X )";
        }
        cout << "]" << endl;
        if (userValues.at(0).attributes.size() == trainingValues.at(0).attributes.size())continue;
        cout << ">>>";

        while (!(cin >> input) || input <= 0.)
        {
            cout << "All attributes must be positive numerical values\n";
            cout << ">>>";
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
        }

        userValues.at(0).attributes.push_back(static_cast<double>(input));
    }

    userValues.at(0).label = "unknown";

    normalize(userValues, trainingValues);
    normalize(trainingValues, trainingValues);

    Perceptron perceptron;
    perceptron.weights.resize(trainingValues.at(0).attributes.size());

    perceptron.train(trainingValues, targetLabel);

    auto output = perceptron.compute(userValues.at(0).attributes);

    if (output == 1)
    {
        cout << targetLabel << endl;
    }
    else cout << "not-" << targetLabel << endl;
    cout << endl;
}

auto readInput()-> void
{
    string input;
    cout << "=========== WELCOME TO THE K-NN PROJECT ===========" << "\n" <<endl;

    const string trainingPath = "../iris_training.txt";
    const string testingPath = "../iris_test.txt";

    ifstream trainingFile(trainingPath);
    vector<Sample> trainingValues;
    getSamples(trainingValues, trainingFile);

    while (true)
    {
        cout << ">>>";

        cin >> input;

        if (input == "test") computeKNN(trainingValues, testingPath);
        else if (input == "stop")return;
        else if (input == "knn")userKNN(trainingValues);
        else if (input == "perc")perceptron(trainingValues, "Iris-setosa");
        else if (input == "cust")userPerceptron(trainingValues, "Iris-setosa");
        else if (input == "help")
        {
            cout << "'test' to perform K-NN on the test file" << endl;
            cout << "'knn' to perform K-NN on your sample" << endl;
            cout << "'stop' to exit the program\n" << endl;
        }
        else cout << "Invalid input, 'help' for commands list" << endl;
    }
}

auto main()-> int
{
    readInput();
}