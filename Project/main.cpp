#include "reader.h"
#include "network.h"
#include "UI.h"
#include "simd.h"
#include "sisd.h"
#include "parallel.h"
#include "simd_and_parallel.h"


int main() {
    cout << "-------------Welcome--------------. \n"
            "The goal of our program is to recognize the handwritten digits and continue to measure the execution time of the various algorithms. \n*\n*\n"
            "Please enter a number:\n"
            "1- Measure the execution time of the algorithms.\n\n"
            "2- Costume setting. \n\n"
            "3- Default setting. \n\n" 
            "4- Exit\n\n"
            "Your input number is: ";
    UI digit_recognition;
    int runTimeordetect =  digit_recognition.read_valid_int_input(1, 4);
    cout << endl;
    if(runTimeordetect == 1) {
        Reader images, lables, testImages, testLabels;
        digit_recognition.read_file(images, lables, testImages, testLabels);
        for (int j = 1; j < 5; j++) {
            switch(j){
                case 1:
                    cout << "1- Using sisd\n!!please wait it takes time!!\n";
                    break;
                case 2:
                    cout << "2- Using simd\n!!please wait it takes time!!\n";
                    break;
                case 3:
                    cout << "3- Using parallel\n!!!please wait it takes time!!\n";
                    break;
                default:
                    cout << "4- Using parallel and simd\n!!please wait it takes time!!\n";
            }
            int hiddenNum = 0;
            for (int i = 0; i < 6; i++) {
                hiddenNum += 10;
                Network *network = digit_recognition.wich_construktor(j, hiddenNum);
                int miniBatchSize = 10, numTests = 60000;
                float learningRate1 = 1.0F;
                digit_recognition.training(numTests, miniBatchSize, learningRate1, *network, images, lables, runTimeordetect, j);
                cout << "-Learning phase " << i <<" is Finisched :)" << endl;
                digit_recognition.netzwerk_result_check(testImages, testLabels, *network, runTimeordetect, j);
                cout << "-Test phase " <<i << " is Finisched :)\nPlease wait, it takes until the next calculation result!!" << endl << endl;
            }
        }
    }
    if(runTimeordetect == 2) {
        cout << "-------------------------------------\n";
        Reader images, labels, testImages, testLabels;
        digit_recognition.read_file(images, labels, testImages, testLabels);
        cout << "1- With sisd\n\n2- With simd\n\n3- With parallel\n\n4- With parallel and simd\n" << "Your input was: ";
        int x = digit_recognition.read_valid_int_input(1, 3);
        cout << endl;
        Network *network1 = digit_recognition.createNetwork(x);
        int miniBatchSize1, numTests1;
        float learningRate1;
        digit_recognition.input_training_parameters(miniBatchSize1, numTests1, learningRate1);
        digit_recognition.training(numTests1, miniBatchSize1, learningRate1, *network1, images, labels, runTimeordetect);
        std::cout << endl << "Learning phase is over! :-)" << endl << endl;
        digit_recognition.netzwerk_result_check(testImages, testLabels, *network1, runTimeordetect);
    }
    //automatic without user input
    if(runTimeordetect == 3){
        Reader images, labels, testImages, testLabels;
        digit_recognition.read_file(images, labels, testImages, testLabels);
        Network *network1 = new parallel(4, 784, 60, 10);
        int miniBatchSize = 10, numTests = 2000;
        float learningRate1 = 2.0F;
        digit_recognition.training(numTests, miniBatchSize, learningRate1, *network1, images, labels, 2);
        digit_recognition.netzwerk_result_check(testImages, testLabels, *network1, 3);

        if(runTimeordetect == 4)
        {
            cout << "Exiting program..."<<endl;
            exit(0);
        }
    }

    return 0;
}


