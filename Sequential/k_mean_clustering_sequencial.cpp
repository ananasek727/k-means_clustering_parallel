#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <limits>
#include <chrono>

using namespace std;

double calculateDistance(const std::vector<double>& point1, const std::vector<double>& point2) {
    double distance = 0.0;
    for (size_t i = 0; i < point1.size(); ++i) {
        distance += pow(point1[i] - point2[i], 2);
    }
    return sqrt(distance);
}

// Function to perform k-means clustering
std::vector<std::vector<double>> kMeans(const std::vector<std::vector<double>>& points, int k, const std::vector<std::vector<double>>& initialCentroids) {
    int maxIterations = 100;
    // Initialize centroids
    std::vector<std::vector<double>> centroids = initialCentroids;

    for (int iteration = 0; iteration < maxIterations; ++iteration) {
        // Assign each point to the nearest centroid
        std::vector<int> labels;
        for (const auto& point : points) {
            double minDistance = std::numeric_limits<double>::max();
            int label = 0;
            for (int i = 0; i < k; ++i) {
                double distance = calculateDistance(point, centroids[i]);
                if (distance < minDistance) {
                    minDistance = distance;
                    label = i;
                }
            }
            labels.push_back(label);
        }

        // Update centroids
        std::vector<std::vector<double>> newCentroids(k, std::vector<double>(points[0].size(), 0.0));
        std::vector<int> clusterSizes(k, 0);

        for (size_t i = 0; i < points.size(); ++i) {
            for (size_t j = 0; j < points[i].size(); ++j) {
                newCentroids[labels[i]][j] += points[i][j];
            }
            clusterSizes[labels[i]]++;
        }

        for (int i = 0; i < k; ++i) {
            if (clusterSizes[i] != 0) {
                for (size_t j = 0; j < newCentroids[i].size(); ++j) {
                    newCentroids[i][j] /= clusterSizes[i];
                }
            }
        }

        centroids = newCentroids;
    }

    return centroids;
}

int main() {
    // Read data from file
    ifstream inputFile("Points.txt");
    if (!inputFile.is_open()) {
        cerr << "Error opening the file!" << endl;
        return 1;
    }
    int size, dimensions, k;
    inputFile.ignore(numeric_limits<streamsize>::max(), ' ');
    inputFile >> size;
    inputFile.ignore(numeric_limits<streamsize>::max(), ' ');
    inputFile.ignore(numeric_limits<streamsize>::max(), ' ');
    inputFile >> dimensions;
    inputFile.ignore(numeric_limits<streamsize>::max(), ' ');
    inputFile.ignore(numeric_limits<streamsize>::max(), ' ');
    inputFile >> k;
    inputFile.ignore(numeric_limits<streamsize>::max(), '\n');
    vector<vector<double>> points(size, vector<double>(dimensions));

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < dimensions; ++j) {
            inputFile >> points[i][j];
        }
    }
    inputFile.close();

    ifstream inputFile2("InitialCentroids.txt");
    
    if (!inputFile2.is_open()) {
        cerr << "Error opening the file!" << endl;
        return 1;
    }
    vector<vector<double>> initialCentroids(k, vector<double>(dimensions));
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < dimensions; ++j) {
			inputFile2 >> initialCentroids[i][j];
		}
	}
    inputFile2.close();
    auto start = std::chrono::high_resolution_clock::now();

    vector<vector<double>> centroids = kMeans(points, k, initialCentroids);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "Time taken: " << duration.count() << " microseconds" << std::endl;


    // Display the centroids

    cout << "Final Centroids:" << endl;
    for (const auto& centroid : centroids) {
        for (double coordinate : centroid) {
            cout << coordinate << " ";
        }
        cout << endl;
    }

    return 0;
}
