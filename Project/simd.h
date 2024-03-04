#pragma once

#include "network.h"



class simd: public Network{
public:
    simd(int N_Layers, int N_Inputs, int N_Hiddens, int N_Outputs);

   
     void think() override;
   
    void learn(float learn_Rate, int batch_Size) override;



};


#ifndef SIMD_H
#define SIMD_H
#endif
