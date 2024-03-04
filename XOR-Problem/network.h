#pragma once
#include <cmath>
#include <vector>
#include "layer.h"

class Network
{
public:
	/// N_Layers:indicates the number of layers in the network.
    /// N_Input: indicates the number of input neurons.
    /// N_Hiddens:indicates the number of hidden neurons.
    /// N_Outputs:specifies the number of output neurons

	Network(int N_Layers, int N_Inputs, int N_Hiddens, int N_Outputs): 
		n_Layers(N_Layers),
		n_Inputs(N_Inputs),
		n_Hiddens(N_Hiddens),
		n_Outputs(N_Outputs)
	{
		if (n_Layers < 3 || n_Inputs < 1 || n_Hiddens < 1 || n_Outputs < 1)
		{
			throw std::runtime_error("Invalid Network Construction");
			return;
		}

		correct_Answer.resize(n_Outputs);
		layer_List.resize(n_Layers);

		layer_List[0] = new Layer(n_Inputs, 0);
		layer_List[1] = new Layer(n_Hiddens, n_Inputs);
		for (int i = 1; i < (n_Layers - 2); i++)
		{
			layer_List[i + 1] = new Layer(n_Hiddens, n_Hiddens);
		}
		layer_List[n_Layers-1] = new Layer(n_Outputs, n_Hiddens);
	}
	/// all the elements of the layerList array. 
	/// This is necessary because the layerList array is dynamically allocated and contains pointers 
	/// to dynamically allocated objects. If these objects are not deleted when they 
	/// are no longer needed, it will cause a memory leak, which can lead to performance problems
	///  and other issues.
	~Network()
	{
		for (int i = 0; i < n_Layers; i++)
		{
			delete layer_List[i];
		}
	/** OR //eliminates the need for an index variable and 
    simply iterates over all elements of layerList, deleting each one in turn
    for (Layer* layer : Layer_List)
    {
        delete layer;
    }*/

	}
	//============================================================
	/**
	 * @brief set the input value for a specific neuron in the network before the network is activated
	*/
	void set_Input_Value(float input_Value, int index)
	{
		layer_List[0]->inputActivation(input_Value, index);
	}

	//============================================================
	/**
	 * @brief activate the neural network and calculate the outputs for all neurons
	*/
	void think()
	{
		/// The loop calls the "fire" method for each layer,
	    /// passing the previous layer as a parameter.
		for (int i = 1; i < n_Layers; i++)
		{
			///store the previous layer in a pointer p
			Layer const * p = layer_List[i - 1];
			layer_List[i]->fire(p);
		}
	}

	//============================================================

	/**
	 * @brief  calculates the "cost" (in the sense of a measure of error)
     * to check the network for its performance and to determine the error it makes when solving a task.
     * This information could then be used to adjust the network and improve its performance.
	*/
	void calculate_Cost(int real_Answer)
	{
		cost = 0.0f;

		int max(0);
		/// check up to how much output layer we have
		for (int i = 0; i < n_Outputs; i++)
		{
			/// iterates through the outputs of the final layer of the neural network and keeps track of 
	        /// the index of the output with the highest activation value.
    	    /// if activation[max] < activation[i] then increase max by 1
			if (layer_List[n_Layers -1]->getActivation(max) < layer_List[n_Layers - 1]->getActivation(i))
			{
				max = i;
			}
			/// if i = correct answer then Correct_Answer = 1 this 1 stands for the confirmation that the answer is correct
			if (i == real_Answer)
			{
				correct_Answer[i] = 1.0f;
			}
			/// if i != correct answer then Correct_Answer = 0 this 0 stands for the confirmation that the answer is wrong
			else
			{
				correct_Answer[i] = 0.0f;
			}
			/// For each output, the function sets the corresponding element of an array m_richtige_antwort 
     		/// to 1.0 if the output index is equal to the correct answer, 
     		/// and 0.0 otherwise. The function then updates the m_kosten variable 
      		/// by adding the square of the difference between the corresponding element
      		/// of m_richtige_antwort and the activation value for the output, multiplied by 0.5.
      		/// cost = ((Correct_Answer[i] - Layer_List[n_Layers -1]. a[i])^2)
       		/// Y =  1 / (1 +  e^-(∑ ((weight * input) + bias)) ) = sigmod(∑(weight * input)+bias))
			cost += pow((correct_Answer[i] - layer_List[n_Layers-1]->getActivation(i)), 2) * 0.5f;

			Correct = 0;
			if (correct_Answer[max] == 1.0f)
			{
				Correct = 1;
			}
			Answer = max;
		}
	}

	float getCost() const
	{
		return cost;
	}

	int isCorrect() const
	{
		return Correct;
	}

	int getAnswer() const
	{
		return Answer;
	}

	//============================================================
	/**
	 * implementing backpropagation, a standard algorithm for training neural network
	*/
	void learn(float learn_Rate, int batch_Size) // stochastic gradient descent + back propegation
	{

		//Calculate Output Layer

		for (int j = 0; j < n_Outputs; j++)
		{
			layer_List[n_Layers-1]->inputdCdA((layer_List[n_Layers-1]->getActivation(j) - correct_Answer[j]) * transform_Prime(layer_List[n_Layers-1]->getZ(j)), j);

			for (int k = 0; k < n_Hiddens; k++)
			{
				layer_List[n_Layers-1]->movedCdW(-(learn_Rate / (float)batch_Size) * layer_List[n_Layers-1]->getdCdA(j) * layer_List[n_Layers-2]->getActivation(k), j, k);
			}

			layer_List[n_Layers-1]->movedCdB(-(learn_Rate / (float)batch_Size) *  layer_List[n_Layers-1]->getdCdA(j), j);
		}

		//Calculate Hidden Layers

		for (int i = (n_Layers - 2); i > 0; i--)
		{
			for (int j = 0; j < layer_List[i]->getNumNeurons(); j++)
			{
				layer_List[i]->inputdCdA(0.0f, j);

				for (int k = 0; k < layer_List[i + 1]->getNumNeurons(); k++)
				{
				    layer_List[i]->movedCdA(layer_List[i + 1]->getWeight(k,j) * layer_List[i + 1]->getdCdA(k) * transform_Prime(layer_List[i]->getZ(j)), j);
				}

				for (int k = 0; k < layer_List[i - 1]->getNumNeurons(); k++)
				{
					layer_List[i]->movedCdW(-(learn_Rate / (float)batch_Size) * layer_List[i]->getdCdA(j) * layer_List[i - 1]->getActivation(k), j, k);
				}

				layer_List[i]->movedCdB(-(learn_Rate / (float)batch_Size) * layer_List[i-1]->getdCdA(j), j);
			}
		}
	}
	/**
	 * @brief reset all the layers in the Layer_List data member of the Network object
	 * starting from the second layer (index 1) and going up to the last layer (n_Layers - 1)
	*/
	void apply_Learned()
	{
		for (int i = 1; i < n_Layers; i++)
		{
			layer_List[i]->Apply_Learned_And_Reset();
		}
	}

	//============================================================

	float transform_Prime(float inputValue) //Sigmoid
	{
		return Sigmoid(inputValue) * (1 - Sigmoid(inputValue));
	}

	float Sigmoid(float inputValue) // Sigmoid
	{
		return 1.0f / (1 + exp(-inputValue));
	}

private:


    int Answer;
    int Correct;
    float cost;
    int n_Layers;
    int n_Outputs;
    int n_Hiddens;
	int n_Inputs;

	

	

	

	

	
    std::vector<Layer*> layer_List;
    std::vector<float> correct_Answer;
};