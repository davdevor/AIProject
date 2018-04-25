# AIProject - Long Short Term Memory Recurrent Neural Network Poetry Generation

## About
This project is to Generate Poetyr. We use a Recurrent Neural Network to generate poetry. RNNs are able to utilize a prior sequence of inputs to interpret a current situation similar to humans. The RNN we have uses Long Short Term Memory cells which are capable of making connections further back in the history sequence than a typical RNN.

## Oracle 
The oracle folder contains a python program of what the best case of our project could be. The best case would be that our project generates coherent poems that relate to a topic given. The program looks at a file of poems that have a topic associtated with them. So the user inputs a topic and the program randomly chooses a poem with that topic. To run the program use python3.

## Baseline
The baseline folder contains a python program of the baseline of what our project should meet. The baseline is that the project generates grammatically correct sentences that may have some semantic meaning. The program has a defined sentence structure and access to various words with different parts of speech and randomly generates a sentence. To run the program use python3. The basseline folder also contains a python program that uses oxford dictonary's api to retrieve the words.

## RNN
The rnn folder contains our recurrent neural network to generate poems. It contains a program called poetryformatter that took the poems from the json file in the oracle folder and converts them to a text file. The rnn file contains our neural network. The neural network is built using tensor flow. To run this file you need a 64-bit version of python3 with tensorflow and numpy installed. The program looks in the saved folder for the checkpoint files. Those files are contain the trained state of the neural network. When the program starts up it restores it to its trained state. To run the program to generate poems you can run this command python3 rnn.py. If you want to train it further you can use python3 rnn.py -t .

## Semantics
The semantics folder contains a program to meaure and plot the performance of our neural network based on the poems in the generatedpoems.txt file. The semantics.py program needs to be ran with python 2.7 and needs sematch installed. When ran it will tell you which datasets it needs installed. To run plot.py you need python3 and matplotlib installed.
