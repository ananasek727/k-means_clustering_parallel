#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdint.h>
#include "utils.h"
#include <iostream>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
//#include <GLFW/glfw3.h>

#define THREADS_PER_BLOCK 256

//void drawPoints(float** points, GLFWwindow* window) {
//	glClear(GL_COLOR_BUFFER_BIT);
//
//	glBegin(GL_POINTS);
//	//for (int i = 0; i < numberOfPoints; i++) {
//	//	for (int j = 0; j < points.dimension; j++) {
//	//		glColor3f(1.0, 1.0, 1.0);  // Set point color (white in this example)
//	//		glVertex3f(points.coordinatesArray[j][i], points.coordinatesArray[j + 1][i], points.coordinatesArray[j + 2][i]);
//	//	}
//	//}
//	glEnd();
//
//	glfwSwapBuffers(window);
//	glfwPollEvents();
//}

// constants for random number filling of the points
const int lowerBound = 0;
const int upperBound = 10;
struct pointCoordinates
{
	float** coordinatesArray; // Creating Structure of Arrays with unknow number of dimensions [dimension][numberOfPoints]
	// Coordinates array will look like this:
	// x1, x2, x3, x4, x5, x6, x7, x8, x9, x10
	// y1, y2, y3, y4, y5, y6, y7, y8, y9, y10
	// z1, z2, z3, z4, z5, z6, z7, z8, z9, z10
	// ...


	__host__ pointCoordinates(unsigned int numberOfPoints, unsigned int dimension)
	{
		coordinatesArray = new float* [dimension];
		for (int i = 0; i < dimension; i++)
		{
			coordinatesArray[i] = new float[numberOfPoints];
		}
	}

	__host__ pointCoordinates(unsigned int dimension)
	{
		coordinatesArray = new float* [dimension];
		for (int i = 0; i < dimension; i++)
		{
			coordinatesArray[i] = nullptr;
		}
	}

	__host__ pointCoordinates() {}

	//prints the points in order: x1, y1, z1, x2, y2, z2, ...
	__host__ void printCoordinates(unsigned int numberOfPoints, unsigned int dimension)
	{
		for (int i = 0; i < numberOfPoints; i++)
		{
			for (int j = 0; j < dimension; j++)
			{
				printf("%f ", coordinatesArray[j][i]);
			}
			printf("\n");
		}
	}

	__host__ void fillCoordinatesWithRandomNumbers(unsigned int numberOfPoints, unsigned int dimension)
	{
		// Filling the coordinates with random numbers
		for (int i = 0; i < dimension; i++)
		{
			for (int j = 0; j < numberOfPoints; j++)
			{
				coordinatesArray[i][j] = lowerBound + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (upperBound - lowerBound)));
			}
		}
	}
};

__host__ pointCoordinates randomCentroids(pointCoordinates* points, unsigned int numberOfPoints, unsigned int dimension, unsigned int k)
{
	pointCoordinates centroids(k, dimension);
	int* centroidsIndex = chooseRandomPoints(numberOfPoints, k);
	for (int i = 0; i < k; i++)
	{
		for (int j = 0; j < dimension; j++)
		{
			centroids.coordinatesArray[j][i] = points->coordinatesArray[j][centroidsIndex[i]];
		}
	}
	return centroids;
}

cudaError_t k_mean(pointCoordinates* points, pointCoordinates* clusters, int* members, unsigned int numberOfPoints, unsigned int dimensions, unsigned int k, unsigned int maxiterations);
cudaError_t k_mean_2(pointCoordinates* points, pointCoordinates* clusters, int* members, unsigned int numberOfPoints, unsigned int dimensions, unsigned int k, unsigned int maxiterations);

__device__ float distanceInNDimensions(const float* point1, const float* point2, unsigned int numberOfPoints, unsigned int dimensions, unsigned int k)
{
	float distance = 0;
	for (int i = 0; i < dimensions; i++)
	{
			distance += (point1[i * numberOfPoints] - point2[i * k]) * (point1[i * numberOfPoints] - point2[i * k]);
	}
	return distance;
}
__host__ __device__ float distanceInNDimensionsHost(float** point1, float** point2, unsigned int numberOfPoints, unsigned int dimensions) {
	float distance = 0;
	for (int i = 0; i < dimensions; i++) {
		for (int j = 0; j < numberOfPoints; j++)
		{

			distance += (point1[j][i] - point2[j][i]) * (point1[j][i] - point2[j][i]);

		}
	}
	return sqrt(distance);
}

__global__ void nearestCentroid(float** points, float** centroids, int* members, unsigned int numberOfPoints, unsigned int dimensions, unsigned int k, unsigned int* histogram)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	extern __shared__ float centroidsShared[];
	if (index < numberOfPoints)
	{
		if (index < k)
		{
			for (int i = 0; i < dimensions; i++)
			{
				centroidsShared[i * k + index] = centroids[i][index];
			}
		}
	}
	__syncthreads();
	if (index < numberOfPoints) {
		float minDistance = FLT_MAX;
		//compare distances to every cetroid 
		for (int i = 0; i < k; i++) {
			float distance = distanceInNDimensions(&points[0][index], &centroidsShared[i], numberOfPoints, dimensions, k);
			if (distance < minDistance) {
				minDistance = distance;
				members[index] = i;
			}
		}
		atomicAdd(&histogram[members[index]], 1);
	}
}

// Return an array of k randomly picked Centroids (points)


__global__ void updateLocalCentroid(float** points, float** centroids, int* members, unsigned int numberOfPoints, unsigned int dimensions, unsigned int k)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < numberOfPoints)
	{
		if(index < k)
		{
			for (int i = 0; i < dimensions; i++)
			{
				centroids[i][index] = 0;
			}
		}
		int centroidIndex = members[index];

		if (centroidIndex < k)
		{
			for (int i = 0; i < dimensions; i++) {
				atomicAdd(&centroids[i][centroidIndex], points[i][index]);
			}
		}

	}
}
__global__ void updateGlobalCentroid(float** centroids, unsigned int* histogram, int numberOfPoints, int dimensions, int k)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	extern __shared__ unsigned int histogramShared[];
	if (index < k)
	{
		histogramShared[index] = histogram[index];
	}
	__syncthreads();
	if (index < k) {
		if (histogramShared[index] != 0)
		{
			for (int i = 0; i < dimensions; i++)
			{
				centroids[i][index] = centroids[i][index] / histogramShared[index];
			}
		}
		else {
			for (int i = 0; i < dimensions; i++)
			{
				centroids[i][index] = 0;
			}
		}

		histogram[index] = 0;

	}
}


/// //////////////////////////////////////////////Main function////////////////////////////////////////////////////////////////////////////////////

int main()
{
	unsigned int h_numberOfPoints;
	unsigned int h_dimension;
	unsigned int h_k;
	unsigned int h_iterations;
	printf("Number of threads per block: %d\n", THREADS_PER_BLOCK);
	printf("Number of points (standard notation): ");
	std::cin >> h_numberOfPoints;
	printf("\n");
	printf("Number of dimensions: ");
	std::cin >> h_dimension;
	printf("\n");
	printf("Number of clusters: ");
	std::cin >> h_k;
	printf("\n");
	printf("Number of iterations: ");
	std::cin >> h_iterations;

	// number of clusters
	pointCoordinates h_points(h_numberOfPoints, h_dimension);

	// array of members of each point
	int* h_members = new int[h_numberOfPoints];
	for (int i = 0; i < h_numberOfPoints; i++)
	{
		h_members[i] = -1;
	}

	h_points.fillCoordinatesWithRandomNumbers(h_numberOfPoints, h_dimension);
	//h_points.printCoordinates(h_numberOfPoints, h_dimension);
	//saveArrayToFilePoints(h_points.coordinatesArray, h_dimension, h_numberOfPoints, h_k, (std::string)"Points.txt");


	//Selecting random initial centroids
	pointCoordinates h_centroids = randomCentroids(&h_points, h_numberOfPoints, h_dimension, h_k);
	//h_centroids.printCoordinates(h_k, h_dimension);


	// Main Parrallel K-Mean function with atomic operations
	cudaEvent_t start, start1, stop, stop1;

	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	printf("\n");
	printf("Clustering with atomic operations: \n");

	cudaStatus = k_mean(&h_points, &h_centroids, h_members, h_numberOfPoints, h_dimension, h_k, h_iterations);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Time taken: %f ms\n", milliseconds);


	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}
	// Main Parrallel K-Mean function without atomic operations
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);
	cudaEventRecord(start1, 0);

	printf("\n");
	printf("Clustering without atomic operations: \n");

	cudaStatus = k_mean_2(&h_points, &h_centroids, h_members, h_numberOfPoints, h_dimension, h_k, h_iterations);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}
	cudaEventRecord(stop1, 0);
	cudaEventSynchronize(stop1);
	cudaEventElapsedTime(&milliseconds, start1, stop1);
	printf("Time taken: %f ms\n", milliseconds);

	//cudaStatus = cudaDeviceReset();


//Free memory
	for (int i = 0; i < h_dimension; i++)
	{
		delete[] h_points.coordinatesArray[i];
	}

	delete[] h_points.coordinatesArray;

	for (int i = 0; i < h_dimension; i++)
	{
		delete[] h_centroids.coordinatesArray[i];
	}

	delete[] h_centroids.coordinatesArray;
	free(h_members);
	return 0;
}

/////////////////////////////////////////////////////// k_mean ////////////////////////////////////////////////////////////////////////////////////

// Helper function for using CUDA to perform k-mean in parallel.
cudaError_t k_mean(pointCoordinates* points, pointCoordinates* clusters, int* members, unsigned int numberOfPoints, unsigned int dimensions, unsigned int k, unsigned int maxiterations)
{
	float* d_points;
	float** d_points2D;
	float* d_clusters;
	float** d_clusters2D;

	int* d_members;
	// histogram of the number of members in each cluster
	unsigned int* d_histogram;
	unsigned int* h_histogram = new unsigned int[k];

	// array of clusters befor the update
	float** h_clusters2D_Old = new float* [dimensions];

	for (int i = 0; i < dimensions; i++)
	{
		h_clusters2D_Old[i] = new float[k];
	}

	cudaError_t cudaStatus;

	// Allocate GPU buffers -----------------------------------------------------------------------------------------------------------------------

	cudaMallocManaged(&d_points, numberOfPoints * dimensions * sizeof(float));

	// Initialize the array elements
	for (int i = 0; i < dimensions; i++) {
		for (int j = 0; j < numberOfPoints; j++) {
			d_points[i * numberOfPoints + j] = points->coordinatesArray[i][j];
		}
	}

	// Cast the linear array to a pointer-to-pointer type
	cudaMallocManaged(&d_points2D, dimensions * sizeof(float*));
	for (int i = 0; i < dimensions; i++) {
		d_points2D[i] = &d_points[i * numberOfPoints];
	}

	cudaMallocManaged(&d_clusters, k * dimensions * sizeof(float));

	// Initialize the array elements
	for (int i = 0; i < dimensions; i++) {
		for (int j = 0; j < k; j++) {
			d_clusters[i * k + j] = clusters->coordinatesArray[i][j];
		}
	}

	// Cast the linear array to a pointer-to-pointer type
	cudaMallocManaged(&d_clusters2D, dimensions * sizeof(float*));
	for (int i = 0; i < dimensions; i++) {
		d_clusters2D[i] = &d_clusters[i * k];
	}


	cudaStatus = cudaMalloc((void**)&d_members, numberOfPoints * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&d_histogram, k * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	// Copy input vectors from host memory to GPU buffers -----------------------------------------------------------------------------------------

	cudaStatus = cudaMemcpy(d_members, members, numberOfPoints * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// --------------------------------------------------------------------------------------------------------------------------------------------
	int blocks = (numberOfPoints + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	int threads = THREADS_PER_BLOCK;
	int blocks2 = (k + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	int threads2 = THREADS_PER_BLOCK;
	int iterations = 0;
	printf("\n");

	////////////////////////////////////////////////////////Main part where magic happen//////////////////////////////////////////////////////////////
	do
	{

		// find and assign the nearest centroid to each point -------------------------------------------------------------------------------------
		nearestCentroid << <blocks, threads, k* dimensions * sizeof(float) >> > (d_points2D, d_clusters2D, d_members, numberOfPoints, dimensions, k, d_histogram);
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize() failed: %s\n", cudaGetErrorString(cudaStatus));
		}

		// remember the old centroids
		for (int i = 0; i < dimensions; i++)
		{
			for (int j = 0; j < k; j++)
			{
				h_clusters2D_Old[i][j] = d_clusters2D[i][j];
			}
		}

		// update the centroids per block ---------------------------------------------------------------------------------------------------------
		updateLocalCentroid << <blocks, threads>> > (d_points2D, d_clusters2D, d_members, numberOfPoints, dimensions, k);
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			goto Error;
		}
		updateGlobalCentroid << <blocks2, threads2, k * sizeof(int) >> > (d_clusters2D, d_histogram, numberOfPoints, dimensions, k);
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			goto Error;
		}
		++iterations;
	} while (iterations < maxiterations);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "kuku k-mean launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	printf("\nNumber of iterations: %d\n", iterations);
	// print the final centroids
	printf("\nNew Centroids: \n");
	for (int i = 0; i < k; i++)
	{
		for (int j = 0; j < dimensions; j++)
		{
			printf("%f ", d_clusters2D[j][i]);
		}
		printf("\n");
	}

	// Copy output vector from GPU buffer to host memory. -----------------------------------------------------------------------------------------
	cudaStatus = cudaMemcpy(members, d_members, numberOfPoints * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Save the final centroids to file -----------------------------------------------------------------------------------------------------------



Error:
	std::string filename1 = "outputMembers_1.txt";
	std::string filename2 = "outputCentroids_1.txt";
	//saveArrayToFileMembers(members, d_points2D, numberOfPoints, k, filename1);
	//saveArrayToFileCentroids(d_clusters2D, dimensions, k, filename2);
	size_t free, total;
	cudaFree(d_points);
	cudaFree(d_points2D);
	cudaFree(d_clusters);
	cudaFree(d_clusters2D);
	cudaFree(d_members);
	cudaFree(d_histogram);
	delete[] h_histogram;
	for (int i = 0; i < dimensions; i++)
	{
		delete[] h_clusters2D_Old[i];
	}
	delete[] h_clusters2D_Old;
	printf("\n");
	cudaMemGetInfo(&free, &total);
	return cudaStatus;
}
__device__ float distanceInNDimensions_2(const float* point1, const float* point2, unsigned int numberOfPoints, unsigned int dimensions, unsigned int k)
{
	float distance = 0;
	for (int i = 0; i < dimensions; i++)
	{
			distance += (point1[i] - point2[i * k]) * (point1[i] - point2[i * k]);
	}
	return distance;
}
__global__ void nearestCentroid_2(float** points, float** centroids, int* members, unsigned int numberOfPoints, unsigned int dimensions, unsigned int k, unsigned int* histogram)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	extern __shared__ float centroidsShared[];
	if (index < numberOfPoints)
	{
		if (index < k)
		{
			histogram[index] = 0;
			for (int i = 0; i < dimensions; i++)
			{
				centroidsShared[i * k + index] = centroids[i][index];
			}
		}
	}
	__syncthreads();
	if (index < numberOfPoints) {
		float minDistance = FLT_MAX;
		//compare distances to every cetroid 
		for (int i = 0; i < k; i++) {
			float distance = distanceInNDimensions_2(points[index], &centroidsShared[i], numberOfPoints, dimensions, k);
			if (distance < minDistance) {
				minDistance = distance;
				members[index] = i;
			}
		}
		atomicAdd(&histogram[members[index]], 1);
	}
}
__global__ void sumParallel(float** points, unsigned int start, unsigned int dimension, unsigned int size, float* result)
{
	//split data into shared memory by dimension
	extern __shared__ float sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (start <= i && i < size) {
		sdata[tid] = points[i][dimension];
	}
	else {
		sdata[tid] = 0;
	}
	__syncthreads();
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	if (tid == 0) {
		result[blockIdx.x] = sdata[0];
	}

}


__host__ void updateCentroids(float** points, float** centroids, unsigned int* histogram, unsigned int numberOfPoints, unsigned int dimensions, unsigned int k)
{
	int blocks = (numberOfPoints + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	int threads = THREADS_PER_BLOCK;
	float* result = new float[dimensions];

	float* d_sum_output;
	float* h_sum_output = new float[numberOfPoints];
	cudaMalloc((void**)&d_sum_output, numberOfPoints * sizeof(float));

	int histogram_counter = 0;

	for (int i = 0; i < dimensions; i++)
	{
		result[i] = 0;
	}
	for (int i = 0; i < k; i++)
	{
		if (histogram[i] != 0)
		{
			for (int j = 0; j < dimensions; ++j)
			{
				result[j] = 0;
				sumParallel << < blocks, threads, threads * sizeof(float) >> > (points, histogram_counter, j, histogram_counter + histogram[i], d_sum_output);
				cudaMemcpy(h_sum_output, d_sum_output, numberOfPoints * sizeof(float), cudaMemcpyDeviceToHost);
				for (int k = 0; k < blocks; k++)
				{
					result[j] += h_sum_output[k];
				}
			}
			histogram_counter += histogram[i];

			for (int j = 0; j < dimensions; ++j)
			{
				centroids[j][i] = result[j] / histogram[i];

			}
		}
		else
		{
			++histogram_counter;
			for (int j = 0; j < dimensions; ++j)
			{
				centroids[j][i] = 0;
			}
		}
	}
}


__host__ void sortByMembers(float** points, int* members, unsigned int numberOfPoints, unsigned int dimensions)
{
	thrust::device_ptr<int> indices(members);
	thrust::device_ptr<float*> dev_ptr(points);
	thrust::sort_by_key(indices, indices + numberOfPoints, dev_ptr);
}

/////////////////////////////////////////////////////// k_mean_2 ////////////////////////////////////////////////////////////////////////////////////
cudaError_t k_mean_2(pointCoordinates* points, pointCoordinates* clusters, int* members, unsigned int numberOfPoints, unsigned int dimensions, unsigned int k, unsigned int maxiterations)
{
	float* d_points;
	float** d_points2D;
	float* d_clusters;
	float** d_clusters2D;

	int* d_members;
	// histogram of the number of members in each cluster
	unsigned int* d_histogram;
	unsigned int* h_histogram = new unsigned int[k];

	// array of clusters befor the update
	float** h_clusters2D_Old = new float* [dimensions];

	for (int i = 0; i < dimensions; i++)
	{
		h_clusters2D_Old[i] = new float[k];
	}

	cudaError_t cudaStatus;

	// Allocate GPU buffers -----------------------------------------------------------------------------------------------------------------------

	cudaMallocManaged(&d_points, numberOfPoints * dimensions * sizeof(float));

	// Initialize the array elements
	for (int i = 0; i < numberOfPoints; i++) {
		for (int j = 0; j < dimensions; j++) {
			d_points[i * dimensions + j] = points->coordinatesArray[j][i];
		}
	}

	// Cast the linear array to a pointer-to-pointer type
	cudaMallocManaged(&d_points2D, numberOfPoints * sizeof(float*));
	for (int i = 0; i < numberOfPoints; i++) {
		d_points2D[i] = &d_points[i * dimensions];
	}

	cudaMallocManaged(&d_clusters, k * dimensions * sizeof(float));

	// Initialize the array elements
	for (int i = 0; i < dimensions; i++) {
		for (int j = 0; j < k; j++) {
			d_clusters[i * k + j] = clusters->coordinatesArray[i][j];
		}
	}

	// Cast the linear array to a pointer-to-pointer type
	cudaMallocManaged(&d_clusters2D, dimensions * sizeof(float*));
	for (int i = 0; i < dimensions; i++) {
		d_clusters2D[i] = &d_clusters[i * k];
	}


	cudaStatus = cudaMalloc((void**)&d_members, numberOfPoints * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&d_histogram, k * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	// Copy input vectors from host memory to GPU buffers -----------------------------------------------------------------------------------------

	cudaStatus = cudaMemcpy(d_members, members, numberOfPoints * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// --------------------------------------------------------------------------------------------------------------------------------------------
	int blocks = (numberOfPoints + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	int threads = THREADS_PER_BLOCK;
	int blocks_k = (k + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	int threads_k = THREADS_PER_BLOCK;
	int iterations = 0;

	////////////////////////////////////////////////////////Main part where magic happen//////////////////////////////////////////////////////////////
	do
	{

		// find and assign the nearest centroid to each point -------------------------------------------------------------------------------------
		nearestCentroid_2 << <blocks, threads, k* dimensions * sizeof(float) >> > (d_points2D, d_clusters2D, d_members, numberOfPoints, dimensions, k, d_histogram);
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize() failed: %s\n", cudaGetErrorString(cudaStatus));
		}

		//remember the old centroids
		for (int i = 0; i < dimensions; i++)
		{
			for (int j = 0; j < k; j++)
			{
				h_clusters2D_Old[i][j] = d_clusters2D[i][j];
			}
		}

		// update the centroids per block ---------------------------------------------------------------------------------------------------------
		sortByMembers(d_points2D, d_members, numberOfPoints, dimensions);
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize() failed: %s\n", cudaGetErrorString(cudaStatus));
		}

		cudaMemcpy(h_histogram, d_histogram, k * sizeof(int), cudaMemcpyDeviceToHost);
		updateCentroids(d_points2D, d_clusters2D, h_histogram, numberOfPoints, dimensions, k);
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize() failed: %s\n", cudaGetErrorString(cudaStatus));
		}
		++iterations;
	} while (iterations < maxiterations);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "kuku k-mean launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	printf("\nNumber of iterations: %d\n", iterations);
	// print the final centroids
	printf("\nNew New Centroids: \n");
	for (int i = 0; i < k; i++)
	{
		for (int j = 0; j < dimensions; j++)
		{
			printf("%f ", d_clusters2D[j][i]);
		}
		printf("\n");
	}

	// Copy output vector from GPU buffer to host memory. -----------------------------------------------------------------------------------------
	cudaStatus = cudaMemcpy(members, d_members, numberOfPoints * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Save the final centroids to file -----------------------------------------------------------------------------------------------------------



Error:
	std::string filename1 = "outputMembers_1.txt";
	std::string filename2 = "outputCentroids_1.txt";
	//saveArrayToFileMembers(members, d_points2D, numberOfPoints, k, filename1);
	//saveArrayToFileCentroids(d_clusters2D, dimensions, k, filename2);
	size_t free, total;
	cudaFree(d_points);
	cudaFree(d_points2D);
	cudaFree(d_clusters);
	cudaFree(d_clusters2D);
	cudaFree(d_members);
	cudaFree(d_histogram);
	delete[] h_histogram;
	for (int i = 0; i < dimensions; i++)
	{
		delete[] h_clusters2D_Old[i];
	}
	delete[] h_clusters2D_Old;
	printf("\n");
	cudaMemGetInfo(&free, &total);
	return cudaStatus;
}