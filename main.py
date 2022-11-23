#from IPython import get_ipython
#import os
#get_ipython().magic('reset -f')
#os.system("clear")

import time
import threading

import numpy as np
import pandas as pd


#from ClassifiersModelsEEG import EEGModels, inception, resnet
#from ClassifierEEG import ClassifierEEG



DEBUG = False

#PROCESS = "TEST"
PROCESS = "ADC-DAC"

if PROCESS == "ADC-DAC":
    import busio
    import digitalio
    import board

    import adafruit_mcp3xxx.mcp3008 as MCP
    from adafruit_mcp3xxx.analog_in import AnalogIn

    # create the spi bus
    spi = busio.SPI(clock=board.SCK, MISO=board.MISO, MOSI=board.MOSI)

    # create the cs (chip select)
    cs = digitalio.DigitalInOut(board.D5)

    # create the mcp object
    mcp = MCP.MCP3008(spi, cs)

    import adafruit_mcp4725

    #i2c = busio.I2C(board.SCL, board.SDA)
    #dac = adafruit_mcp4725.MCP4725(i2c)



buffer = []
buffer_prep = []
buffer_pred = []
buffer_time = []

sampling_rate = 250
sample_duration = 4  #1000 samples = 4 sec (1000/250)
test_duration = 8

classifier_name  = "eegnet"
directory = './Classifiers_Trained_Models/' + classifier_name + '/'



def stream():
    global PROCESS, mcp, buffer

    if PROCESS == "TEST":
        header = "F3 FC5 AF3 F7 T7 P7 O1 O2 P8 T8 F8 AF4 FC6 F4".split()
        test = pd.read_csv("/Users/MohamedBenomar/Desktop/ETSETB/MEE/2B/TFM/RasPi-Files/RAW_REC/s1_s1.csv")
        test = test[header].iloc[0:(sampling_rate*test_duration)].values.tolist()
        for x in test:
            buffer.append(x)
            time.sleep(1/sampling_rate)

    elif PROCESS == "ADC-DAC":
        # create an analog input channel on pin 0
        chan = AnalogIn(mcp, MCP.P0)
        buffer.append(chan.voltage)



def dac():
    global PROCESS, dac

    if PROCESS == "ADC-DAC":
        header = "F3 FC5 AF3 F7 T7 P7 O1 O2 P8 T8 F8 AF4 FC6 F4".split()
        test = pd.read_csv("./s1_s1.csv")
        test = test[header].iloc[0:(sampling_rate*test_duration)].values.tolist()
        for x in test:
            for y in x:
                dac.raw_value = y
                time.sleep(1/sampling_rate)


def main():
    global buffer, sample_duration, buffer_prep, DEBUG, classifier_name, directory

    s = time.localtime()
    current_time = time.strftime("%H:%M:%S", s)
    print("\n\nStarting Time: {} \n".format(current_time))

    i = 1
    while True:

        samples, buffer = (np.asarray(buffer[0:(sample_duration*sampling_rate)]).transpose(), buffer[(sample_duration*sampling_rate)::]) if len(buffer) >= (sample_duration*sampling_rate) else (np.empty((0,0)), buffer)
        sample_start = time.localtime()

        if samples.shape[1] >= (sample_duration*sampling_rate):

            if DEBUG: print("\n\nIteration: {}\n\n".format(i))
            if DEBUG: print("Sample: {}\n".format(samples.shape))

            x_test, y_pred = ClassifierEEG.fitted_classifier(EEG_cleaned, classifier_name, directory)
            buffer_pred.append(y_pred)

            buffer_time.append(sample_start-time.localtime())

            i += 1

        if i > (test_duration/sample_duration):
            e = time.localtime()
            d=time.mktime(e)-time.mktime(s)
            current_time = time.strftime("%H:%M:%S", e)
            print("\n\n\nEnding Time: {} \nTotal Time: {} sec ({} min)".format(current_time, d, d/60))

            buffer_time.append(sum(buffer_time)/len(buffer_time))
            with open('EEG-Time-Report.csv', 'w') as file:
                for x in buffer_time:
                    file.write(x + '\n')
            break



if __name__ == "__main__":
    thread_stream  = threading.Thread(target=stream)
    thread_dac  = threading.Thread(target=dac)
    thread_main   = threading.Thread(target=main)

    thread_stream.start()
    thread_dac.start()
    thread_main.start()

    thread_stream.join()
    thread_dac.join()
    thread_main.join()
    if DEBUG: print("\nSTOP")
