#pragma once

#include "network.h"


class sisd: public Network{
public:




    void think() override;
    sisd(int N_Layers, int N_Inputs, int N_Hiddens, int N_Outputs);
    void learn(float learn_Rate, int batch_Size) override;



};



#ifndef SISD_H
#define SISD_H
#endif
