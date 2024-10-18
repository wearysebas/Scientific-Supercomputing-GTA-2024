/*
Use C++ streams to implement a function that writes the elements of a std::vector to a
file.
*/

#include <iostream>
#include <vector>
#include <fstream>  // Include fstream for file handling

// Function to write the elements of a vector to a file
void writeVectorToFile(const std::vector<int>& vec, const std::string& filename) {
    std::ofstream outFile(filename);  // Open a file for writing

    // Check if the file opened successfully
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open the file." << std::endl;
        return;
    }

    // Write each element of the vector to the file
    for (int element : vec) {
        outFile << element << std::endl;  // Write each element on a new line
    }

    outFile.close();  // Close the file when done
}

int main() {
    // Declare and initialize a vector with some values
    std::vector<int> array(100);
    for (int i = 0; i < 100; ++i) {
        array[i] = i + 1;  // Assign values 1 to 100
    }

    // Call the function to write the vector to a file
    writeVectorToFile(array, "output.txt");

    std::cout << "Vector written to output.txt" << std::endl;

    return 0;
}
