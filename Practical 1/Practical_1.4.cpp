/*
Write a function called read(file, vector) that reads in the numbers for the file created in last query and stores them in a new vector. 
Verify that the numbers written are read correctly.
*/


#include <iostream>
#include <vector>
#include <fstream>  // Include fstream for file handling

// Function to read the numbers from the file and store them in a vector
void readVectorFromFile(const std::string& filename, std::vector<int>& vec) {
    std::ifstream inFile(filename);  // Open the file for reading

    // Check if the file opened successfully
    if (!inFile.is_open()) {
        std::cerr << "Error: Could not open the file." << std::endl;
        return;
    }

    int number;
    vec.clear();  // Clear the vector before reading new values
    while (inFile >> number) {
        vec.push_back(number);  // Read each number and store it in the vector
    }

    inFile.close();  // Close the file when done
}

// Function to print the elements of the vector to verify the data
void printVector(const std::vector<int>& vec) {
    for (int element : vec) {
        std::cout << element << " ";
    }
    std::cout << std::endl;
}

int main() {
    std::vector<int> readArray;  // Vector to store the numbers read from the file

    // Call the function to read the vector from the file
    readVectorFromFile("output.txt", readArray);

    // Verify by printing the contents of the vector
    std::cout << "Numbers read from file: " << std::endl;
    printVector(readArray);  // Should match the numbers written to the file

    return 0;
}
