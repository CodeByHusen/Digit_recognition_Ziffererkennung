#include <stdexcept>
#include <iostream>
#include "sisd.h"

sisd::sisd(int N_Layers, int N_Inputs, int N_Hiddens, int N_Outputs) : Network(N_Layers,  N_Inputs,  N_Hiddens,  N_Outputs) {}






void sisd::learn(float learn_Rate, int batch_Size) {

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
    void sisd::think() {
	    /// The loop calls the "fire" method for each layer,
        /// passing the previous layer as a parameter.
	    for (int i = 1; i < n_Layers; i++)
	    {
		    ///store the previous layer in a pointer p
		    Layer const * p = layer_List[i - 1];
		    layer_List[i]->fire(p);
	    }

    }
