#pragma once
#include "network.h"



class simd_and_parallel: public Network{
public:


   
    void think() override;
    simd_and_parallel(int N_Layers, int N_Inputs, int N_Hiddens, int N_Outputs);
    void learn(float learn_Rate, int batch_Size);




};



#ifndef SIMD_AND_PARALLEL_H
#define SIMD_AND_PARALLEL_H
#endif