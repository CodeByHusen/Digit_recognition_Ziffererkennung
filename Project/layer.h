#pragma once

#include <vector>
#include <cmath>
#include <cstdlib>
#include <iostream>

using namespace std;

class Layer
{
public:
	/** 
	 * @brief implementation a construct
	 * 
	*/
	Layer(int n, int w);

	//============================================================
	/**
	 * @brief Reset values of these arrays to 0
	 * to update the neural network weights and biases.
	 * to ensure that the updated values of the weights and biases are calculated correctly.
	*/
	void reset();


	//============================================================

	void Apply_Learned_And_Reset();


	/**
	 * initialize the weights and biases of the "Layer" object to random values
	*/
	void randomize();


	//============================================================


	/**
	 * @brief Forwardpropagation
	*/
	void fire(Layer const *prevLayer); // The function takes in a pointer to a previous layer (prevLayer) 


	//============================================================

	/**
	 * @brief activation function
	 * Sigmoid function:This function is often called the sigmoid function,
	 * and it maps values from the range (-infinity, infinity) to the range (0, 1).
	*/
	//float sigmoid_func(float inputValue);
	




	//============================================================

	/**
	 * @brief the training process for a neural network, to adjust the changes to the weights based on some learning rule
	 * or optimization algorithm.
	*/
	void movedCdW(float value, int neuron, int weight);



	/**
	 * @brief the training process for a neural network, to adjust the changes to the biases based on some learning rule 
	 * or optimization algorithm
	*/
	void movedCdB(float value, int neuron);
	

	/**
	 * @brief the changes to the activations based on some learning rule or optimization algorithm
	*/
	void movedCdA(float value, int neuron);

	//============================================================

	/**
	 * @brief This function may be used to provide input to the neural network,
	 * by setting the activation values for the input neurons to the input values for the network.
	*/
	void inputActivation(float value, int neuron);
	


	/**
	 * @brief the output of the neural network, by getting the activation values for the output neurons
	 * after the network has been activated with some input.
	*/
	float getActivation(int neuron) const;
	


	/**
	 * @brief the size of the layer, for example, to loop over the neurons in the layer or
	 * to allocate storage for the activations or weights of the layer
	*/
	int getNumNeurons() const;
	
	/**
	 * @brief the error gradient for the activations of the neurons in the layer, as part of
	 * the backpropagation algorithm for training a neural network
	*/
	void inputdCdA(float value, int neuron);


	/**
	 * @brief retrieve the error gradient for the activations of the neurons in the layer, as part of
	 * the backpropagation algorithm for training a neural network
	*/
	float getdCdA(int neuron) const;
	

	float getZ(int neuron) const;


	/**
	 * @brief the current values of the weights in the neural network, for example, to inspect the 
	 * learned weights or to save the weights to a file
	*/
	float getWeight(int neuronIndex, int weightIndex) const;


	/**
	 * @brief adjust the weights of the neural network as part of the training process
	*/
	float moveWeight(float value, int neuronIndex, int weightIndex);

	
//private:

	int neuron_Count;// The number of neurons passed.

	int weight_Count;// The weight of the passed.

	std::vector<float> a; //The activation vector

	std::vector<float> z; // The hidden layer behind the activation layer


	std::vector<std::vector<float>> weight;

	std::vector<float> bias;

	std::vector<float> dCdA;// derivation of cost function to activation function.

	std::vector<std::vector<float>> dCdW; // derivation of cost function to activation function.

	std::vector<float> dCdB; // Derivation of the cost function according to the B layers.
};


#ifndef PROJECT_LAYER_H
#define PROJECT_LAYER_H

#endif //PROJECT_LAYER_H

