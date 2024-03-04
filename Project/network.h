#pragma once
#include <cmath>
#include <vector>
#include "layer.h"
using namespace std;
using std::cout;
using std::endl;
using std::boolalpha;
class Network
{
public:
    Network()= default;
    Network(int N_Layers, int N_Inputs, int N_Hiddens, int N_Outputs);
    ~Network();

    void set_Input_Value(float input_Value, int index);
    virtual void think()=0;

    float sigmod(float x);
    float dSigmoddX(float x);
    void calculate_Cost(int real_Answer);
    virtual void learn(float learn_Rate, int batch_Size)=0;
    void apply_Learned();
    float getCost()
    {
        return cost;
    }
    int isCorrect(){
        return Correct;
    }
    int getAnswer(){
        return Answer;
    }
    int get_Hidden_Count(){
        return n_Hiddens;
    }
	float transform_Prime(float inputValue);
    //Network& operator=(const Network &other);

private:
    int n_Inputs{};
    float cost{};
    int Correct{};
    int Answer{};
protected:
    int n_Layers{};
   // std::vector<Layer*> layer_List;
    Layer **layer_List;
    int n_Outputs{};
    int n_Hiddens{};
	//std::vector<float> correct_Answer;
    float *correct_Answer;
};

#ifndef PROJECT_NETWORK_H
#define PROJECT_NETWORK_H

#endif //PROJECT_NETWORK_H
