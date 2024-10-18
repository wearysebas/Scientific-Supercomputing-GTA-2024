/* Practical 1.4
Using the timing class, time the write and read operations and
compare the bandwidth to disk (GB/s). Can you explain your observations? What
happens as a function of the number of elements in the array?
*/


#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>  // For timing
#include <cmath>   // For bandwidth calculations

// Function to write the elements of a vector to a file
void writeVectorToFile(const std::vector<int>& vec, const std::string& filename) {
    std::ofstream outFile(filename);

    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open the file for writing." << std::endl;
        return;
    }

    for (int element : vec) {
        outFile << element << std::endl;
    }

    outFile.close();
}

// Function to read the numbers from the file and store them in a vector
void readVectorFromFile(const std::string& filename, std::vector<int>& vec) {
    std::ifstream inFile(filename);

    if (!inFile.is_open()) {
        std::cerr << "Error: Could not open the file for reading." << std::endl;
        return;
    }

    int number;
    vec.clear();
    while (inFile >> number) {
        vec.push_back(number);
    }

    inFile.close();
}

// Function to calculate the size of the vector in bytes
size_t vectorSizeInBytes(const std::vector<int>& vec) {
    return vec.size() * sizeof(int);  // Each int is typically 4 bytes
}

int main() {
    const size_t numElements = 1e7;  // 10 million elements (adjustable)

    // Initialize a vector with `numElements` integers
    std::vector<int> array(numElements);
    for (size_t i = 0; i < numElements; ++i) {
        array[i] = i + 1;  // Initialize vector with values 1 to numElements
    }

    // Timing write operation
    auto startWrite = std::chrono::high_resolution_clock::now();
    writeVectorToFile(array, "output.txt");
    auto endWrite = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> writeTime = endWrite - startWrite;
    size_t dataSize = vectorSizeInBytes(array);  // Size of data written in bytes

    // Timing read operation
    std::vector<int> readArray;
    auto startRead = std::chrono::high_resolution_clock::now();
    readVectorFromFile("output.txt", readArray);
    auto endRead = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> readTime = endRead - startRead;

    // Bandwidth calculations (in GB/s)
    double writeBandwidth = (dataSize / writeTime.count()) / (1e9);  // GB/s
    double readBandwidth = (dataSize / readTime.count()) / (1e9);    // GB/s

    // Output results
    std::cout << "Write time: " << writeTime.count() << " seconds\n";
    std::cout << "Read time: " << readTime.count() << " seconds\n";
    std::cout << "Write bandwidth: " << writeBandwidth << " GB/s\n";
    std::cout << "Read bandwidth: " << readBandwidth << " GB/s\n";

    return 0;
}
