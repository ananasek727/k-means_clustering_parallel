#pragma once
#include "utils.h"
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <set>
#include <stdexcept>
#include <ctime>
#include <iostream>
#include <fstream>

// Returns a intiger array with random numbers from 0 to nuberOfAllPoints-1
int* chooseRandomPoints(unsigned int nuberOfAllPoints, unsigned int numberOfPointToChoose)
{
	std::srand(2);
	if (nuberOfAllPoints < numberOfPointToChoose)
	{
		throw std::runtime_error("Number of points to choose is bigger than number of all points.");
	}
	int* points = new int[numberOfPointToChoose]();
	std::set<int>s;
	int stop = 0;
	for (int i = 0; i < numberOfPointToChoose; i++)
	{
		int pointIndex = rand() % nuberOfAllPoints;
		// picked number must be unique
		if (s.find(pointIndex) == s.end()) {
			s.insert(pointIndex);
			points[i] = pointIndex;
		}
		else
		{
			stop++;
			if (stop == 1000)
			{

				printf("stop");
				throw std::runtime_error("Cannot choose random centroids.");
				break;
			}
			// run loop again to select another random number
			i--;
		}
	}
	return points;
}

void saveArrayToFileMembers(int* members, float** points, unsigned int size, unsigned int k, std::string& filename) {
	std::ofstream outputFile(filename);

	if (!outputFile.is_open()) {
		std::cerr << "Error: Unable to open file for writing: " << filename << std::endl;
		return;
	}

	for (unsigned int i = 0; i < size; ++i) {
		for (int j = 0; j < k; ++j) {
			outputFile << points[j][i] << " ";
		}
		outputFile << " Grup: " << members[i] << " ";
		outputFile << std::endl;
	}

	outputFile.close();
	if (outputFile.fail()) {
		std::cerr << "Error: Failed to close file: " << filename << std::endl;
		return;
	}

	std::cout << "Members array saved to file: " << filename << std::endl;
}

void saveArrayToFileCentroids(float** array, unsigned int dimensions, unsigned int k, std::string& filename) {
	std::ofstream outputFile(filename);

	if (!outputFile.is_open()) {
		std::cerr << "Error: Unable to open file for writing: " << filename << std::endl;
		return;
	}

	for (unsigned int i = 0; i < k; ++i) {
		for (unsigned int j = 0; j < dimensions; ++j) {
			outputFile << array[i][j] << " ";
		}
		outputFile << std::endl;
	}

	outputFile.close();
	if (outputFile.fail()) {
		std::cerr << "Error: Failed to close file: " << filename << std::endl;
		return;
	}

	std::cout << "Centroids saved to file: " << filename << std::endl;
}
void saveArrayToFilePoints(float** array, unsigned int dimensions, unsigned int size, unsigned int k, std::string& filename) {
	std::ofstream outputFile(filename);

	if (!outputFile.is_open()) {
		std::cerr << "Error: Unable to open file for writing: " << filename << std::endl;
		return;
	}
	outputFile << "Size: " << size << " Dimensions: " << dimensions <<" K: "<< k<< std::endl;
	for (unsigned int i = 0; i < size; ++i) {
		for (unsigned int j = 0; j < dimensions; ++j) {
			outputFile << array[j][i] << " ";
		}
		outputFile << std::endl;
	}

	outputFile.close();
	if (outputFile.fail()) {
		std::cerr << "Error: Failed to close file: " << filename << std::endl;
		return;
	}

	std::cout << "Points saved to file: " << filename << std::endl;
}