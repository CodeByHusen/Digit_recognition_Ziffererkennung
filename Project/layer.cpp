#include <cmath>
#include "layer.h"



/** 
 * @brief implementation a construct
 * 
*/
Layer::Layer(int n, int w){
	
    neuron_Count = n;
	weight_Count = w;
	/**
	* @brief Assignment of vectorsSize the number of neurons
	* The size of the array is determined by the "neuronCount" variable, which is the number of neurons in the network
	* Within the loop, a list of weights is created for each neuron, also specified by the weightCount variable.
	* The size of the list of weights thus corresponds to the number of inputs for the neuron.
	* */	
	
	weight.resize(neuron_Count);
	for (int j = 0; j < neuron_Count; j++)
	{
		weight[j].resize(weight_Count);
	}
	bias.resize(neuron_Count);
	a.resize(neuron_Count);
	z.resize(neuron_Count);
	dCdA.resize(neuron_Count);
	dCdW.resize(neuron_Count);
	for (int i = 0; i < neuron_Count; i++)
	{
		dCdW[i].resize(weight_Count);
	}
		dCdB.resize(neuron_Count);
	reset();
	randomize();
	
    }

//============================================================
/**
 * @brief Reset values of these arrays to 0
 * to update the neural network weights and biases.
 * to ensure that the updated values of the weights and biases are calculated correctly.
*/
void Layer::reset()
{
	//uses the std::fill function from the C++ standard library to reset the values of
    // all elements in the dCdA and dCdB arrays to 0.0 in a single line of code.

	for (int j = 0; j < neuron_Count; j++)
	{
		dCdA[j] = 0.0f;
		dCdB[j] = 0.0f;
		for (int k = 0; k < weight_Count; k++)
		{
			dCdW[j][k] = 0.0f;
		}
	}
}

//============================================================

void Layer::Apply_Learned_And_Reset()
{
	for (int j = 0; j < neuron_Count; j++)
	{
		for (int k = 0; k < weight_Count; k++)
		{
			weight[j][k] += dCdW[j][k];
			dCdW[j][k] = 0.0f;
		}
		bias[j] += dCdB[j];
		dCdB[j] = 0.0f;
	}
}

/**
* initialize the weights and biases of the "Layer" object to random values
*/
void Layer::randomize()
{
	for (int j = 0; j < neuron_Count; j++)
	// We divide rand_num by rand_mak to get a number between 0 and 1 then multiply that by
    // 2 to fix errors, and then subtract -1 to get the correct number, which is between 0 and 1 or 0 and -1

	{
		bias[j] = 2 * ((static_cast <float> (rand())) / (static_cast <float> (RAND_MAX))) - 1;

		for (int k = 0; k < weight_Count; k++)
		{
			weight[j][k] = 2 * ((static_cast <float> (rand())) / (static_cast <float> (RAND_MAX))) - 1;
		}
	}
}

//============================================================


/**
* @brief Forwardpropagation
*/
void Layer::fire(Layer const *prevLayer) // The function takes in a pointer to a previous layer (prevLayer) 
{
	for (int j = 0; j < neuron_Count; j++)
	{
		z[j] = 0.0f;
		for (int k = 0; k < weight_Count; k++)
		{
		//output from the previous layer and the weights connecting the previous layer 
        //to the courrent layer

			z[j] += prevLayer->getActivation(k) * weight[j][k];
		}
			
			
		//followed by the addition of a bias term (bias[j]) and the application of an
       	// "activation function" 
		z[j] += bias[j];
		//a[j] = sigmoid_func(z[j]);
		a[j] = 1.0f / (1 + exp(-z[j]));
	}
}

//============================================================

/**
 * @brief activation function
 * Sigmoid function:This function is often called the sigmoid function,
 * and it maps values from the range (-infinity, infinity) to the range (0, 1).
*/
//float Layer::sigmoid_func(float inputValue)
//{
//	return 1.0f / (1 + exp(-inputValue));
//}



//============================================================

/**
 * @brief the training process for a neural network, to adjust the changes to the weights based on some learning rule
 * or optimization algorithm.
*/
void Layer::movedCdW(float value, int neuron, int weight)
{
	dCdW[neuron][weight] += value; ///the change value for a particular weight.
}


/**
 * @brief the training process for a neural network, to adjust the changes to the biases based on some learning rule 
 * or optimization algorithm
*/
void Layer::movedCdB(float value, int neuron)
{
	dCdB[neuron] += value;
}

/**
 * @brief the changes to the activations based on some learning rule or optimization algorithm
 */
void Layer::movedCdA(float value, int neuron)
{
	dCdA[neuron] += value;
}

//============================================================

/**
 * @brief This function may be used to provide input to the neural network,
 * by setting the activation values for the input neurons to the input values for the network.
*/
void Layer::inputActivation(float value, int neuron)
{
	a[neuron] = value;
}


/**
 * @brief the output of the neural network, by getting the activation values for the output neurons
 * after the network has been activated with some input.
*/
float Layer::getActivation(int neuron) const
{
	return a[neuron]; 
}


/**
 * @brief the size of the layer, for example, to loop over the neurons in the layer or
 * to allocate storage for the activations or weights of the layer
*/
int Layer::getNumNeurons() const
{
	return neuron_Count;
}

/**
 * @brief the error gradient for the activations of the neurons in the layer, as part of
 * the backpropagation algorithm for training a neural network
*/
void Layer::inputdCdA(float value, int neuron)
{
	dCdA[neuron] = value;
}


/**
 * @brief retrieve the error gradient for the activations of the neurons in the layer, as part of
 * the backpropagation algorithm for training a neural network
*/
float Layer::getdCdA(int neuron) const
{
	return dCdA[neuron];
}

float Layer::getZ(int neuron) const
{
	return z[neuron];
}


/**
 * @brief the current values of the weights in the neural network, for example, to inspect the 
 * learned weights or to save the weights to a file
*/
float Layer::getWeight(int neuronIndex, int weightIndex) const
{
	return weight[neuronIndex][weightIndex]; //test
}


/**
 * @brief adjust the weights of the neural network as part of the training process
*/
float Layer::moveWeight(float value, int neuronIndex, int weightIndex)
{
	return weight[neuronIndex][weightIndex] += value;
}


