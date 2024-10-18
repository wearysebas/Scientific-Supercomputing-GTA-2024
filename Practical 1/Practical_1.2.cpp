/*
Use standard library vector containers to declare an array of 100 elements. Write a C++ 
function, taking a vector as its argument, to print all the elements of the array to the
screen. Make sure that you pass the array by reference, and not by value
*/

#include <iostream>
#include <vector>  // Include vector header

// Function to print the elements of the vector, passed by reference
void printVector(const std::vector<int>& vec) {
    for (int element : vec) {
        std::cout << element << " ";
    }
    std::cout << std::endl;
}

int main() {
    // Declare a vector of 100 elements initialized to 0
    std::vector<int> array(100, 0);

    // You can modify elements here if needed, e.g., initializing with different values
    for (int i = 0; i < 100; ++i) {
        array[i] = i + 1;  // Assign values 1 to 100
    }

    // Call the function to print the vector
    printVector(array);

    return 0;
}
