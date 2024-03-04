#pragma once

#include "network.h"




class parallel: public Network{
public:



    void think() override;
   parallel(int N_Layers, int N_Inputs, int N_Hiddens, int N_Outputs);
    void learn(float learn_Rate, int batch_Size) override;




};


#ifndef PARALLEL_H
#define PARALLEL_H
#endif
