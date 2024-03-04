#pragma once
#include <fstream>
#include <iostream>
#include <stdint.h>
#include <vector>
#include <sstream>
#include "network.h"
#include "reader.h"

class UI{

public:
    UI(){}
    void read_file(Reader& images, Reader &lables, Reader &testImages, Reader &testLabels);

    int read_valid_int_input(int small_than, int bigger_than);
    Network *createNetwork(int x);
    Network * wich_construktor(int construktor_num, int hidden_num);

    void input_training_parameters(int& mini_batch_size,int& number_of_batches, float& learning_rate);
    void training(int test_num, int mini_batch_size, float learning_rate, Network& myNetwork, Reader images, Reader labels, int time_ortest, int which_file = 0);

    void netzwerk_result_check(Reader testImages, Reader testLabels, Network &myNetwork, int time_ortest, int whichfile =0);


};


//at can cast a string (std::string) to any type (T). The function uses a std::istringstream to parse the string as T. If the parsing succeeds, the resulting value is returned
template<class T> T from_string(const std::string& s)
{
    std::istringstream stream (s);
    T t;
    stream >> t;
    return t;
}




#ifndef PROJECT_UI_H
#define PROJECT_UI_H

#endif //PROJECT_UI_H