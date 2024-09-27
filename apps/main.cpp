#include <iostream>
#include <vector>
#include <array>
#include "../src/Convolution.hpp"

int main(int argc, char** argv)
{   
    std::vector<int> data = {1,2,3,4,5,6,7,8,9};
    std::array<int,3> kernel = {1,1,1};
    convolution1D(data,kernel);
}
