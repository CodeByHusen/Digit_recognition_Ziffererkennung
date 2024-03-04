#include "reader.h"
#include "network.h"
#include <chrono>
#include <thread>





int main()
{


	//Start Training

	

	int mini_BatchSize = 1;
	float my_BatchCost = 0.3f;
	mini_BatchSize = 4;
	float batch_Accuracy = 0.0f;
	float learn_Rate = 10.1;
	int batchsize = 1;

	Network myNetWork(3, 2, 9, 2);

	for (int i = 0; i < 35; i++) {
		my_BatchCost = 0.0f;
    		batch_Accuracy = 0.0f;

    		// Train on input (0, 0) and expect output 0
    		myNetWork.set_Input_Value(0, 0);
    		myNetWork.set_Input_Value(0, 1);
	    	myNetWork.think();
	    	myNetWork.calculate_Cost(0);
	    	myNetWork.learn(learn_Rate, batchsize);
		my_BatchCost += myNetWork.getCost() / mini_BatchSize;
		batch_Accuracy += ((float)myNetWork.isCorrect() / (float)mini_BatchSize);
		myNetWork.apply_Learned();

	   	// Train on input (1, 0) and expect output 1
	    	myNetWork.set_Input_Value(1, 0);
	    	myNetWork.set_Input_Value(0, 1);	
	    	myNetWork.think();
	    	myNetWork.calculate_Cost(1);
		myNetWork.learn(learn_Rate, batchsize);
		my_BatchCost += myNetWork.getCost() / mini_BatchSize;
		batch_Accuracy += ((float)myNetWork.isCorrect() / (float)mini_BatchSize);
		myNetWork.apply_Learned();
   

	    	// Train on input (0, 1) and expect output 1
	   	myNetWork.set_Input_Value(0, 0);
	   	myNetWork.set_Input_Value(1, 1);	
	   	myNetWork.think();
	   	myNetWork.calculate_Cost(1);
	   	myNetWork.learn(learn_Rate, batchsize);
		my_BatchCost += myNetWork.getCost() / mini_BatchSize;
		batch_Accuracy += ((float)myNetWork.isCorrect() / (float)mini_BatchSize);
		myNetWork.apply_Learned();

	    	// Train on input (1, 1) and expect output 0
	    	myNetWork.set_Input_Value(1, 0);
	    	myNetWork.set_Input_Value(1, 1);
	    	myNetWork.think();
	   	myNetWork.calculate_Cost(0);
	    	myNetWork.learn(learn_Rate, batchsize);
		my_BatchCost += myNetWork.getCost() / mini_BatchSize;
		batch_Accuracy += ((float)myNetWork.isCorrect() / (float)mini_BatchSize);
		myNetWork.apply_Learned();


	}

	// Test the trained network
	myNetWork.set_Input_Value(0, 0);
	myNetWork.set_Input_Value(0, 1);
	myNetWork.think();
	myNetWork.calculate_Cost(1);
	std::cout << "Machine's answer for input (0, 0): " << myNetWork.getAnswer() << endl;

	myNetWork.set_Input_Value(1, 0);
	myNetWork.set_Input_Value(0, 1);
	myNetWork.think();
	myNetWork.calculate_Cost(0);
	std::cout << "Machine's answer for input (1, 0): " << myNetWork.getAnswer() << endl;

	myNetWork.set_Input_Value(0, 0);
	myNetWork.set_Input_Value(1, 1);
	myNetWork.think();
	myNetWork.calculate_Cost(0);
	std::cout << "Machine's answer for input (0, 1): " << myNetWork.getAnswer() << endl;

	myNetWork.set_Input_Value(1, 0);
	myNetWork.set_Input_Value(1, 1);

	myNetWork.think();
	myNetWork.calculate_Cost(1);
	std::cout << "Machine's answer for input (1, 1): " << myNetWork.getAnswer() << endl;


	
	return 0;
}
