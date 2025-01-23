#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <string>
#include <algorithm>
#include <random>
#include <set>
#include <chrono>


__global__ void computeEuclideanDistance(float *data, float *output, int numRows, int numCols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < numRows && j < numRows) {
        float sum = 0;
        for (int k = 0; k < numCols; ++k) {
            float diff = data[j * numCols + k] - data[i * numCols + k];
            sum += diff * diff;
        }
        output[i * numRows + j] = sqrt(sum);
    }
}

__global__ void computeDegreeMatrix(float *affinityMatrix, float *degreeMatrix, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        float sum = 0.0;
        for (int j = 0; j < size; j++) {
            sum += affinityMatrix[index * size + j];
        }
        degreeMatrix[index] = sum;
    }
}

__global__ void computeLaplacianMatrix(float *degreeMatrix, float *affinityMatrix, float *laplacianMatrix, int size) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < size && col < size) {
        if (row == col){
            laplacianMatrix[row * size + col] = degreeMatrix[row] - affinityMatrix[row * size + col];
        }
        else{
            laplacianMatrix[row * size + col] = -affinityMatrix[row * size + col];
        }
    }
}

void initialize_centroids(float* centroids, float* data, int k, int dimensions, std::mt19937& gen) {
    std::uniform_int_distribution<> distrib(0, dimensions - 1);
    std::set<int> uniqueNumbers;
    
    while (uniqueNumbers.size() < k) {
        int number = distrib(gen);
        uniqueNumbers.insert(number);
    }

    int i=0;

    for (int idx : uniqueNumbers) {
        for (int j = 0; j < k; ++j) {
            centroids[i]=data[k*idx + j];
            i++;
        }
    }
}


__global__ void compute_centroids(int* assignments, float* centroids, float* data, float* sum, float* count, int k, int dimensions, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        int cluster = assignments[i];
        for (int j = 0; j < dimensions; ++j) {
            sum[cluster * dimensions + j] += data[i * dimensions + j];
        }
        count[cluster]++;
    }
}

__global__ void assign_clusters(const float* data, const float* centroids, int* cluster_assignment, int n, int k, int dimensions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float min_dist = INFINITY;
        int closest_centroid = 0;
        for (int i = 0; i < k; ++i) {
            float dist = 0.0;
            for (int d = 0; d < dimensions; ++d) {
                float diff = data[idx * dimensions + d] - centroids[i * dimensions + d];
                dist += diff * diff;
            }
            if (dist < min_dist) {
                min_dist = dist;
                closest_centroid = i;
            }
        }
        cluster_assignment[idx] = closest_centroid;
    }
}

__global__ void markKNearestNeighbors(float *distances, float *neighbors, int n, int k) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n) {
        for (int i = 0; i < n; i++) {
            neighbors[row * n + i] = 0.0f;
        }

        for (int i = 0; i < k; i++) {
            float minDist = INFINITY;
            int minIndex = -1;

            for (int j = 0; j < n; j++) {
                if (j == row) continue;

                if (distances[row * n + j] < minDist && neighbors[row * n + j] == 0.0f) {
                    minDist = distances[row * n + j];
                    minIndex = j;
                }
            }

            if (minIndex != -1) {
                neighbors[row * n + minIndex] = 1.0f;
            }
        }
    }
}

__global__ void averageWithTranspose(float *A, float *result, int size) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < size) {
        for (int i = 0; i < size; i++) {
            float element = A[row * size + i];
            float transposedElement = A[i * size + row];
            result[row * size + i] = 0.5f * (element + transposedElement);
        }
    }
}

bool areEqual(const int* arr1, const int* arr2, int size) {
    for (int i = 0; i < size; ++i) {
        if (arr1[i] != arr2[i]) {
            return false;
        }
    }
    return true;
}

std::vector<std::vector<float>> loadCSV(const char* filename) {
    std::vector<std::vector<float>> data;
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        std::vector<float> row;

        while (std::getline(lineStream, cell, ',')) {
            float val;
            std::stringstream cellStream(cell);
            if (cellStream >> val) {
                row.push_back(val);
            }
        }

        if (!row.empty()) {
            data.push_back(row);
        }
    }

    return data;
}

int main(int argc, char* argv[]) {

    int k = 3;
    int iter = 0;
    int maxIter = 20;
    int kNearest = 12;
    const char* inputFile;
    const char* outputFile;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-k") {
            if (i + 1 < argc) {
                std::istringstream ss(argv[++i]);
                ss >> k;
            }
        } else if (arg == "-i") {
            if (i + 1 < argc) {
                std::istringstream ss(argv[++i]);
                ss >> maxIter;
            }
        } else if (arg == "-n") {
            if (i + 1 < argc) {
                std::istringstream ss(argv[++i]);
                ss >> kNearest;
            }
        } else if (arg == "-f") {
            if (i + 1 < argc) {
                inputFile = argv[++i];
            }
        } else if (arg == "-o") {
            if (i + 1 < argc) {
                outputFile = argv[++i];
            }
        }
    }
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<float>> data = loadCSV(inputFile);
    int numRows = data.size(); 
    int numCols = numRows > 0 ? data[0].size() : 0;
    int size = numRows;

    std::vector<float> flatData;
    for (const auto& row : data) {
        flatData.insert(flatData.end(), row.begin(), row.end());
    }

    float *d_output, *d;
    cudaMalloc(&d, size * numCols * sizeof(float));
    cudaMalloc(&d_output, size * size * sizeof(float));
    cudaMemcpy(d, flatData.data(), size * numCols* sizeof(float), cudaMemcpyHostToDevice);

  
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(ceil(size / 16.0), ceil(size / 16.0));
    int blockiSize = 256;
    int numBlocks2 = (size + blockiSize - 1) / blockiSize;

    computeEuclideanDistance<<<numBlocks, threadsPerBlock>>>(d, d_output, numRows, numCols);

    std::vector<float> h_output(size * size);
    cudaMemcpy(h_output.data(), d_output, size * size * sizeof(float), cudaMemcpyDeviceToHost);

    float *d_neighbors;
    cudaMalloc(&d_neighbors, sizeof(float) * size * size);

    markKNearestNeighbors<<<numBlocks2, blockiSize>>>(d_output, d_neighbors, size, kNearest);

    float *h_neighbors = new float[size * size];
    cudaMemcpy(h_neighbors, d_neighbors, sizeof(float) * size * size, cudaMemcpyDeviceToHost);
    cudaFree(d_output);

    float *d_result;
    cudaMalloc(&d_result, sizeof(float) * size * size);
    averageWithTranspose<<<numBlocks2, blockiSize>>>(d_neighbors, d_result, size);
    cudaDeviceSynchronize();
    cudaFree(d_neighbors);

    float *h_result = new float[size * size];
    cudaMemcpy(h_result, d_result, sizeof(float) * size * size, cudaMemcpyDeviceToHost);
 
    float *d_degreeMatrix, *d_laplacianMatrix;
    cudaMalloc(&d_degreeMatrix, size * sizeof(float));
    cudaMalloc(&d_laplacianMatrix, size * size * sizeof(float));

    computeDegreeMatrix<<<numBlocks, threadsPerBlock>>>(d_result, d_degreeMatrix, size);

    computeLaplacianMatrix<<<numBlocks, threadsPerBlock>>>(d_degreeMatrix, d_result, d_laplacianMatrix, size);
    std::vector<float> h_laplacianMatrix(size * size);
    cudaMemcpy(h_laplacianMatrix.data(), d_laplacianMatrix, size * size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_degreeMatrix);
    cudaFree(d_result);

   
    cusolverDnHandle_t cusolver_handle;
    cusolverDnCreate(&cusolver_handle);


    float *dev_eigenvalues;
    cudaMalloc(&dev_eigenvalues, size * sizeof(float));

    
    int lwork = 0;
    cusolverDnSsyevd_bufferSize(
        cusolver_handle,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_LOWER,
        size,
        d_laplacianMatrix,
        size,
        dev_eigenvalues,
        &lwork
    );

    float *work;
    cudaMalloc(&work, lwork * sizeof(float));

    int *dev_info;
    cudaMalloc(&dev_info, sizeof(int));

    cusolverDnSsyevd(
        cusolver_handle,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_LOWER,
        size,
        d_laplacianMatrix,
        size,
        dev_eigenvalues,
        work,
        lwork,
        dev_info
    );

    cudaFree(work);
   
    std::vector<float> host_eigenvalues(size);
    cudaMemcpy(host_eigenvalues.data(), dev_eigenvalues, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dev_eigenvalues);

    float *h_V = new float[size * size]; 
    cudaMemcpy(h_V, d_laplacianMatrix, sizeof(float) * size * size, cudaMemcpyDeviceToHost);
    cudaFree(d_laplacianMatrix);
    
    int info = 0;
    cudaMemcpy(&info, dev_info, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_info);

    float *h_selectedEigenvectors = new float[size * k];
    
     for (int i = 0; i < k; i++) {
        for (int j = 0; j < size; j++) {
            h_selectedEigenvectors[j*k+i] = h_V[i * size + j];
        }
    }
    
    float *host_centroids = new float[k * k];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, size*k / k - 1);
    
    initialize_centroids(host_centroids, h_selectedEigenvectors, k, size, gen);
    
    int *assignments= new int[size]();
    int *new_assignments= new int[size];
    new_assignments[0] = 1;
    float* dev_centroids;

    cudaMalloc(&dev_centroids, k * k * sizeof(float));
    cudaMemcpy(dev_centroids, host_centroids, k * k * sizeof(float), cudaMemcpyHostToDevice);
    float* dev_eigenvectors;
    cudaMalloc(&dev_eigenvectors, k * size * sizeof(float));
    cudaMemcpy(dev_eigenvectors, h_selectedEigenvectors, k * size * sizeof(float), cudaMemcpyHostToDevice);
    int* dev_assignments;
    cudaMalloc(&dev_assignments, size * sizeof(int));
    float* sum = new float[k * k]();
    float* count = new float[k]();
    float* dev_sum;
    float* dev_count;
    cudaMalloc(&dev_sum, k * k * sizeof(float));
    cudaMalloc(&dev_count, k * sizeof(float));
    cudaMemcpy(dev_sum, sum, k * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_count, count, k * sizeof(float), cudaMemcpyHostToDevice);

    while (!(areEqual(new_assignments, assignments, size)) && (iter<maxIter)) {
        int *assignments = new_assignments;
        assign_clusters<<<numBlocks, threadsPerBlock>>>(dev_eigenvectors, dev_centroids, dev_assignments, size, k, k);
        compute_centroids<<<numBlocks, threadsPerBlock>>>(dev_assignments, dev_centroids, dev_eigenvectors, dev_sum, dev_count, k, k, size);
        cudaMemcpy(host_centroids, dev_centroids, sizeof(float) * k * k, cudaMemcpyDeviceToHost);
        cudaMemcpy(sum, dev_sum, sizeof(float) * k * k, cudaMemcpyDeviceToHost);
        cudaMemcpy(count, dev_count, sizeof(float) * k, cudaMemcpyDeviceToHost);
        cudaMemcpy(new_assignments, dev_assignments, sizeof(float) * size, cudaMemcpyDeviceToHost);
        for (int i = 0; i < k; ++i) {
            if (count[i] > 0) {
                for (int j = 0; j < k; ++j) {
                    host_centroids[i * k + j] = sum[i * k + j] / count[i];
                }
            } else {
            
                int idx = distrib(gen) * k;
                for (int j = 0; j < k; ++j) {
                    host_centroids[i * k + j] = h_selectedEigenvectors[idx + j];
                }
            }
        }
        for (int i = 0; i < k; ++i) {
            count[i] = 0;
            for (int j = 0; j < k; ++j){
                sum[i*k+j] = 0;
            }
        }
        cudaMemcpy(dev_sum, sum, k * k * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_count, count, k * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_centroids, host_centroids, k * k * sizeof(float), cudaMemcpyHostToDevice);
        iter++;
    }
    cudaMemcpy(assignments, dev_assignments, sizeof(int) * size, cudaMemcpyDeviceToHost);

    std::ofstream file(outputFile);
    for (size_t i = 0; i < size; ++i) {
   
        file << assignments[i] << "\n";
    }
    file.close();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    cusolverDnDestroy(cusolver_handle);
    cudaFree(dev_centroids);
    cudaFree(dev_eigenvectors);
    cudaFree(dev_assignments);
    cudaFree(dev_sum);
    cudaFree(dev_count);
    
    return 0;
}