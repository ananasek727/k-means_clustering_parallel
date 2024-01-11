#ifndef Utils_H
#define Utils_H

#include <string>
int* chooseRandomPoints(unsigned int size, unsigned int numberOfPointToChoose);
void saveArrayToFileMembers(int* members, float** points, unsigned int size, unsigned int k, std::string& filename);
void saveArrayToFileCentroids(float** array, unsigned int dimensions, unsigned int k, std::string& filename);
void saveArrayToFilePoints(float** array, unsigned int dimensions, unsigned int size, unsigned int k, std::string& filename);
#endif