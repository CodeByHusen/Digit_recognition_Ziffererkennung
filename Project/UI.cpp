#include <chrono>
#include <thread>
#include "UI.h"
#include "simd.h"
#include "sisd.h"
#include "parallel.h"
#include "simd_and_parallel.h"


/**
 * @brief read four files of images and labels for training and testing a machine learning model
 * The file paths are defined as hardcoded strings with a specific directory.
 * 
*/
void UI::read_file(Reader& images, Reader &labels, Reader & testImages, Reader &testLabels){
   
    string filePfad= "/home/husen/Schreibtisch/semester 5/micro.tech/DigitRecognition-master/Minist_Database/";




    string trainimagesFileString = filePfad + "train_images_idx3_ubyte";
    char * trainImagesFile = const_cast<char *>(trainimagesFileString.c_str());

    string trainLabelFileString = filePfad + "train_labels_idx1_ubyte";
    char * trainLabelFile = const_cast<char *>(trainLabelFileString.c_str());

    string testImagesFileString = filePfad + "t10k_images_idx3_ubyte";
    char * testImagesFile = const_cast<char *>(testImagesFileString.c_str());

    string testLabelsFileString = filePfad + "t10k_labels_idx1_ubyte";
    char * testLabelsFile = const_cast<char *>(testLabelsFileString.c_str());

    Reader file1(trainImagesFile),
            file2(trainLabelFile),
            file3(testImagesFile),
            file4(testLabelsFile);
    images = file1;
    labels = file2;
    testImages = file3;
    testLabels = file4;

}


/*
 * Creat a function for  check over the positive integer input  
 * Valid number between small_than number and biger_than number
 * If the entry is invalid, an error message is issued and the user is prompted to enter a new entry
 * If the input is valid, it is returned
*/
int UI::read_valid_int_input(int small_than, int bigger_than)
{
    int input;
    int x = 0;
    string inputstring;
    while (x == 0)
    {
        std::getline(cin, inputstring, '\n');


        try {
            if (inputstring.empty()) {
                throw invalid_argument("Please enter a number and do not leave the line empty");
            }
            for (char i : inputstring) {
                x = 1;
                if (!isdigit(i)) {
                    string throwausgabe = "Wrong input: ";
                    throwausgabe += "\nOnly positive numbers may be entered\n";
                    throw invalid_argument(throwausgabe);
                }
            }
            if (from_string<int>(inputstring) > bigger_than || from_string<int>(inputstring) < small_than) {
                string throwausgabe = "Wrong entry: Please only enter numbers between " + to_string(small_than) + " and " +
                                      to_string(bigger_than);
                throw invalid_argument(throwausgabe);
            }
            if (inputstring[0] == '0') {
                string throwausgabe = "Incorrect input: Please only enter numbers greater than 0 ";
                throw invalid_argument(throwausgabe);
            }
        }
        catch (invalid_argument &e) {
            cerr << e.what() << endl;
            x = 0;
            inputstring = "";
        }
    }
    input = from_string<int>(inputstring);
    return input;
}


/**
 * @brief defines the dimensions of the neural network.
 * The function asks the user for the number of layers and the number of neurons per hidden layer
 * Then, depending on the argument x passed, a specific type of network is created and returned
*/
Network *UI::createNetwork(int x)
{
    int arg1, arg2;

    std::cout << "Input number of layers (we recomended 4 layers): ";
    arg1 = read_valid_int_input( 3, 9);   //layer anzahl
    std::cout << "\nInput number of neurons per hidden layer (we recomended 30 neurons): ";
    arg2 = read_valid_int_input(1, 999);
    std::cout << endl;
    switch (x){
        case 1: return new sisd(arg1, 784, arg2, 10);
        case 2: return new simd(arg1, 784, arg2, 10);
        case 3: return new parallel(arg1, 784, arg2, 10);
        default: return new simd_and_parallel(arg1, 784, arg2, 10);
    }
}

/**
 * @brief prompt the user to enter three arguments that correspond to the neural network training data
 * 1-Mini-Batch Size: Number of training images to process per batch
 * 2-Test Count: Number of batches to train
 * 3-Lernrate: Faktor, um den die Gewichte nach jeder Trainingsiteration angepasst werden sollen.
*/
void UI::input_training_parameters(int& mini_batch_size,int& number_of_batches, float& learning_rate)
{
    std::cout << "Input number of training images per batch (we recomended: 10 images per batch): ";
    mini_batch_size = read_valid_int_input(1, 999);
    std::cout << "\nInput number of training batches you want to run (we recomended 1000 ttaining batches): ";
    number_of_batches = read_valid_int_input(1, 60000);
    std::cout << "\nInput learning rate (we recomended 1 learning rate): ";
    cin >> learning_rate;
    std::cout << endl;
}

/**
 * @brief trains a neural network on a set of data by dividing it into mini-batches
 * For each mini-batch, it retrieves the input data (images) and the corresponding target data (labels)
 * and feeds the input data into the network to get the network's output
 * t then calculates the cost (error) of the output with respect to the target data and updates the network's weights 
 * using backpropagation with a specified learning rate
 * This process is repeated for each mini-batch in the set of data.
 * 
 * The function also calculates the time taken for each step of the process and outputs the
 *  accuracy of each mini-batch and the overall cost
 * If the time_ortest argument is 1, the function saves the time taken and other parameters to a specified file.
 * If the time_ortest argument is 2, the function outputs the accuracy and cost of each mini-batch.
*/
void UI::training(int test_num, int mini_batch_size, float learning_rate, Network& myNetwork, Reader images, Reader labels, int time_ortest, int which_file)
{
    float myBatchCost;
    float batchAccuracy;

    int imageIndex = 0;
    float thinkTime = 0.0f;
    float learnTime = 0.0f;

    for (int b = 0; b < test_num; b++) {
        myBatchCost = 0.0f;
        batchAccuracy = 0.0f;

        imageIndex++;
        int imageIndexBatch = 0;
        imageIndexBatch = imageIndex;
        for (int m = 0; m < mini_batch_size; m++) {
            /// Dadurch stzen wir den imageindex auf die nächste Bild in der For Loop
            imageIndexBatch++;
            /// Wir speichern Ein Bild in den ersten Element in den Layers index und vorallem in den vector a von den Layer Layer_list[0]. Bei jedes Schleifen Durchlauf der obige Schlife wird ein neues Bild in den Index a gespeichert. jedes Bild besteht aus 784 Ziffern wobei die Höhe und Breite 28 ist und damit 28 * 28 = 784 pixel
            /// Es ist wichtig zu wissen, dass die Zahlen in den aus 0 und eine zahl zwischen 0 und 1 steht. die Ziffern werden durch Zahlen die nicht 0 sind dar gestellt
            for (int i = 0; i < 784; i++) {
                myNetwork.set_Input_Value(images.getPixel(imageIndexBatch * 784 + i), i);
            }
            auto startThink = std::chrono::system_clock::now();
            myNetwork.think();
            auto endThink = std::chrono::system_clock::now();
            std::chrono::duration<float> thinkElapsed_seconds = endThink - startThink;
            thinkTime += thinkElapsed_seconds.count();

            myNetwork.calculate_Cost(labels.getLabel(imageIndexBatch));

            auto startLearn = std::chrono::system_clock::now();
            myNetwork.learn(learning_rate, mini_batch_size);
            auto endLearn = std::chrono::system_clock::now();
            std::chrono::duration<float> learnElapsed_seconds = endLearn - startLearn;
            learnTime += learnElapsed_seconds.count();

            myBatchCost += myNetwork.getCost() / mini_batch_size;
            batchAccuracy += ((float)myNetwork.isCorrect() / (float)mini_batch_size);
        }
        myNetwork.apply_Learned();
      if(time_ortest == 2){ // alone if we test to save time
          std::cout << "Batch: " << b + 1 << " / " << test_num << "--> Accuracy in %: " << batchAccuracy * 100 <<  "% Cost: " << myBatchCost << endl;
      }
    }
    if(time_ortest == 2){
        std::cout << endl << "Done Training!" << endl << endl;
    }
    if(time_ortest == 1) {  
        string whichfilename[4] = {"sisd_time.txt", "simd_time.txt", "parallel_time.txt", "simd_parallel_time.txt"};
        string filePfad = "/home/husen/Schreibtisch/semester 5/micro.tech/ziffererkennung_husen_cynthia_2022_wise/duration of the learning process/";
        switch (time_ortest) {
     
            case 1:
                filePfad += whichfilename[0];
                break;
            case 2:
                filePfad += whichfilename[1];
                break;
            case 3:
                filePfad += whichfilename[2];
                break;
            default:
                filePfad += whichfilename[3];
                break;
        }
        fstream file;
        file.open(filePfad, std::ios_base::app | std::ios_base::in);
        if (file.is_open()) {
            // file << thinkTime << " " << learnTime << " "<< myNetwork.getHiddenCount() << " " << learningRate << " "<< miniBatchSize;
            file << thinkTime << " " << learnTime << " " << myNetwork.get_Hidden_Count() << " ";
        }
        file.close();
    }
}



/**
 * @brief check the accuracy of a given neural network using set of test images
 * and corresponding labels 
 *  The function can be run in three modes depending on the value of time_ortest
 * 1-If time_ortest is 1, the function calculates the overall accuracy of the network and saves 
 * it to a file (specified by whichfile) in the format "X%"
 * 
 * 2-If time_ortest is 2, the function displays each test image and the corresponding network 
 * output, as well as the accumulated accuracy after each test.
 * 
 * 3-If time_ortest is 3, the same as mode 2, but with a 2 second delay before the first
 *  test and a 1/10 second delay between subsequent tests.
 * 
*/
void UI::netzwerk_result_check(Reader testImages, Reader testLabels, Network &myNetwork, int time_ortest, int whichfile)
{
	using namespace std::this_thread;
	using namespace std::chrono;
	
	if (time_ortest == 2 || time_ortest == 3) {
	sleep_for(seconds(2));
	}

	float sumResultPercent = 0.0f;
	int numTest = 0;
	int correct = 0;
	if (time_ortest == 3) {
	    numTest = 1000;
	} else {
	    numTest = 10000;
	}
	for (int m = 0; m < numTest; m++) {
	    int addNewLine = 0;
	    for (int i = 0; i < 784; i++) {
		myNetwork.set_Input_Value(testImages.getPixel(m * 784 + i), i);
		if (time_ortest == 2 || time_ortest == 3) {
		    addNewLine++;
		    if (testImages.getPixel(m * 784 + i) == 0) {
		        cout << ". ";
		    } else if (testImages.getPixel(m * 784 + i) != 0) {
		        cout << "0 ";
		    }
		    if (addNewLine == 28) {
		        cout << endl;
		        addNewLine = 0;
		    }
		}
	    }
	    myNetwork.think();
	    myNetwork.calculate_Cost(testLabels.getLabel(m));
	    correct += myNetwork.isCorrect();
	    if (time_ortest == 1) {
		sumResultPercent = (100 * ((float)(correct) / (float)(m + 1)));
	    }
	    if (time_ortest == 2 || time_ortest == 3) {
		std::cout << endl << "Machine's Answer is: " << myNetwork.getAnswer() << endl;
		std::cout << endl << "Accumulated Accuracy in % : " << 100 * ((float)(correct) / (float)(m + 1)) << "% " << endl;
		if (time_ortest == 2) {
		    sleep_for(seconds(1) / 10);
		}
	    }
	}


	if (time_ortest == 1) {
	    std::string file_names[4] = {"sisd_time.txt", "simd_time.txt", "parallel_time.txt", "simd_parallel_time.txt"};
	    std::string file_path = "/home/husen/Schreibtisch/semester 5/micro.tech/ziffererkennung_husen_cynthia_2022_wise/duration of the learning process/";
	    switch (time_ortest) {
		case 1:
		    file_path += file_names[0];
		    break;
		case 2:
		    file_path += file_names[1];
		    break;
		case 3:
		    file_path += file_names[2];
		    break;
		default:
		    file_path += file_names[3];
		    break;
	    }
	    std::fstream file;
	    file.open(file_path, std::ios_base::app | std::ios_base::in);
	    if (file.is_open())
	    {
	    	file << (sumResultPercent) <<"%" << endl;
	    
	    }
	    file.close();
}
}

/**
 * @brief  create an object of one of four classes
 * all derived from a common class Network: sisd, simd, parallel, and parallel_and_simd
 * : sisd, simd, parallel, and parallel_and_simd. The number of layers (4), the number of 
 * input neurons (784), the number of neurons in a hidden layer (hiddenNum) and the number 
 * of output neurons (10) are passed as arguments.
*/
Network *UI::wich_construktor(int construktorNum, int hiddenNum) {
    switch(construktorNum){
        case 1: return new sisd(4, 784, hiddenNum, 10);
        case 2: return new simd(4, 784, hiddenNum, 10);
        case 3: return new parallel(4, 784, hiddenNum, 10);
        default: return new simd_and_parallel(4, 784, hiddenNum, 10);
    }
 }
