# Anti-Jamming
Classify the jamming pattern and predict the action of channel selection in the future time slots.
\\
The model is described as follow. A radio system operates in a pool of frequencies, for example 8 frequencies from f1 to f8. A sequence of spectrum information in the past is collected every time slot. To get the spectrum information in each time slot, a radio device needs to sense all frequencies. Spectrum information at each frequency can get the value 0 or 1. Here, 0 means that frequency was detected as a free frequency and 1 means that frequency was detected as a jamming frequency. However, in practice the radio device could not sense all spectrum at each time slot. In our problem, it can sense only one frequency at a time slot. Thus, our spectrum information has missing 7 values out of 8 frequencies in each time slot. In order to predict the jamming pattern in the future, we first need to reconstruct the spectrum information. This will be done in step 1.


Step 1: Run Unet_TF2.py to reconstruct the jamming sequence with missing values. Input is spectrum information with missing values. Output is a complete spectrum information.


Step 2: Run lstm_pytoch.py to predict the future jamming pattern. Input is a reconstruct spectrum information and output is a spectrum information in multiple time slots ahead.


Step 3: Run DQL_AntiJamming_withData.py to select one frequency from the frequency pool to use in the future which can avoid the jamming frequencies. Input state is the complete spectrum information at the current time slot which can get from the output of step 2. Action is one of frequency in the pool. 


Step 4: Run Evaluate.py to evaluate the DQL_AntiJamming_withData scheme where the input state comes from the test set.


Step 5: We can use the environment gym to generate the spectrum state in the case we do not have the data in hand. The gym environment can generate the complete spectrum information which can have some type of jamming pattern such as comb, sweep and random. So the Step 3 and 4 can be runned by using Jamming_env.py without the data provided from Step 1 and 2.
