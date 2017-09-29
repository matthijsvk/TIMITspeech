from __future__ import print_function

import warnings
from pprint import pprint  # printing properties of networkToRun objects
from time import gmtime, strftime

# pprint(vars(a))

warnings.simplefilter("ignore", UserWarning)  # cuDNN warning

import logging
import formatting
from tqdm import tqdm

logger_combined = logging.getLogger('combined')
logger_combined.setLevel(logging.DEBUG)
FORMAT = '[$BOLD%(filename)s$RESET:%(lineno)d][%(levelname)-5s]: %(message)s '
formatter = logging.Formatter(formatting.formatter_message(FORMAT, False))

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger_combined.addHandler(ch)

# File logger: see below META VARIABLES

import time

program_start_time = time.time()

print("\n * Importing libraries...")
from combinedNN_tools import *
from general_tools import *
import traceback

###################### Script settings #######################################
root = os.path.expanduser('~/TCDTIMIT/')
resultsPath = root + 'combinedSR/TCDTIMIT/results/allEvalResults.pkl'

logToFile = True;
overwriteResults = False

# if you wish to force retrain of networks, set justTest to False, forceTrain in main() to True, and overwriteSubnets to True.
# if True, and justTest=False, even if a network exists it will continue training. If False, it will just be evaluated
forceTrain = False

# JustTest: If True, mainGetResults just runs over the trained networks. If a network doesn't exist, it's skipped
#           If False, ask user to train networks, then start training networks that don't exist.
justTest = False

getConfusionMatrix = True  # if True, stores confusionMatrix where the .npz and train_info.pkl are stored
# use this for testing with reduced precision. It converts the network weights to float16, then back to float32 for execution.
# This amounts to rounding. Performance should hardly be impacted.
ROUND_PARAMS = False

# use this to TEST trained networks on the test dataset with noise added.
# This data is generated using audioSR/fixDataset/mergeAudiofiles.py + audioToPkl_perVideo.py and combinedSR/dataToPkl_lipspeakers.py
# It is loaded in in combinedNN_tools/finalEvaluation (also just before training in 'train'. You could also generate noisy training data and train on that, but I haven't tried that
withNoise = False
noiseTypes = ['white', 'voices']
ratio_dBs = [0, -3, -5, -10]


###################### Script code #######################################

# quickly generate many networks
def createNetworkList(dataset, networkArchs):
    networkList = []
    for networkArchi in networkArchs:
        networkList.append(networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=networkArch,
                     audio_dataset=dataset, test_dataset=dataset))
    return networkList

networkList = [
        # # # # # ### AUDIO ###  -> see audioSR/RNN.py, there it can run in batch mode which is much faster
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[64,64],
        #                audio_dataset="TCDTIMIT", test_dataset="TCDTIMIT"),#,forceTrain=True),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[256, 256],
        #              audio_dataset="combined", test_dataset="TCDTIMIT"),#, forceTrain=True)
        networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[256, 256],
                     audio_dataset="TCDTIMIT", test_dataset="TCDTIMIT"),  # , forceTrain=True)
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[512,512],
        #                audio_dataset="TCDTIMIT", test_dataset="TCDTIMIT"),#,forceTrain=True),

        # run TCDTIMIT-trained network on lipspeakers, and on the real TCDTIMIT test set (volunteers)
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[256, 256],
        #              audio_dataset="TCDTIMIT", test_dataset="TIMIT"),  # , forceTrain=True)  % 66.75 / 89.19
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[256, 256],
        #                audio_dataset="TCDTIMIT", test_dataset="TCDTIMITvolunteers"),  # ,forceTrain=True),

        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[64,64],
        #                audio_dataset="combined", test_dataset="TCDTIMIT"),  # ,forceTrain=True),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[256, 256],
        #                audio_dataset="combined", test_dataset="TCDTIMIT"),  # ,forceTrain=True),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[512, 512],
        #                audio_dataset="combined", test_dataset="TCDTIMIT"),  # ,forceTrain=True),

        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[256, 256],
        #                audio_dataset="combined", test_dataset="TCDTIMIT"),  # ,forceTrain=True),


        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[8], audio_dataset="TIMIT", test_dataset="TIMIT"),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[32], audio_dataset="TIMIT", test_dataset="TIMIT"),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[64], audio_dataset="TIMIT", test_dataset="TIMIT"),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[256], dataset="TIMIT", audio_dataset="TIMIT", test_dataset="TIMIT"),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[512], audio_dataset="TIMIT", test_dataset="TIMIT"),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[1024], audio_dataset="TIMIT", test_dataset="TIMIT"),
        #
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[8, 8], audio_dataset="TIMIT", test_dataset="TIMIT"),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[32, 32], audio_dataset="TIMIT", test_dataset="TIMIT"),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[64, 64], audio_dataset="TIMIT", test_dataset="TIMIT"),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[256, 256], audio_dataset="TIMIT", test_dataset="TIMIT"),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[512, 512], audio_dataset="TIMIT", test_dataset="TIMIT"),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[1024,1024], audio_dataset="TIMIT", test_dataset="TIMIT"),
        #
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[8,8,8], audio_dataset="TIMIT", test_dataset="TIMIT"),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[32,32,32], audio_dataset="TIMIT", test_dataset="TIMIT"),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[64,64,64], audio_dataset="TIMIT", test_dataset="TIMIT"),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[256,256,256], audio_dataset="TIMIT", test_dataset="TIMIT"),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[512,512,512], audio_dataset="TIMIT", test_dataset="TIMIT"),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[1024, 1024, 1024], audio_dataset="TIMIT", test_dataset="TIMIT"),
        #
        #
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[8,8,8,8], audio_dataset="TIMIT", test_dataset="TIMIT"),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[32,32,32,32], audio_dataset="TIMIT", test_dataset="TIMIT"),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[64,64,64,64], audio_dataset="TIMIT", test_dataset="TIMIT"),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[256,256,256,256], audio_dataset="TIMIT", test_dataset="TIMIT"),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[512,512,512,512], audio_dataset="TIMIT", test_dataset="TIMIT"),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[1024,1024,1024,1024], audio_dataset="TIMIT", test_dataset="TIMIT"),

        # #get the MFCC results
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[64, 64], audio_dataset="TIMIT", test_dataset="TIMIT",nbMFCCs=13),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[64, 64], audio_dataset="TIMIT", test_dataset="TIMIT",nbMFCCs=26),
        # networkToRun(runType="audio", AUDIO_LSTM_HIDDEN_LIST=[64, 64], audio_dataset="TIMIT", test_dataset="TIMIT",nbMFCCs=39),
    ]

def main():

    # Use this if you want only want to start training if the network doesn't exist
    if withNoise:
        allResults = []
        for noiseType in noiseTypes:
            for ratio_dB in ratio_dBs:
                results, resultsPath = mainGetResults(networkList, withNoise, noiseType, ratio_dB)
                allResults.append(results)
                # print(allResults)
                # import pdb;pdb.set_trace()

        allNoisePath = root + 'resultsNoisy.pkl'
        exportResultsToExcelManyNoise(allResults, allNoisePath)
    else:
        results, resultsPath = mainGetResults(networkList)
        print("\n got all results")
        exportResultsToExcel(results, resultsPath)

    # Use this if you want to force run the network on train sets. If justTest==True, it will only evaluate performance on the test set
    runManyNetworks(networkList, withNoise=withNoise)

class networkToRun:
    def __init__(self,
                 AUDIO_LSTM_HIDDEN_LIST=[256, 256], audio_dataset="TCDTIMIT", nbMFCCs=39, audio_bidirectional=True,
                 LR_start=0.001,
                 forceTrain=False, runType='audio',
                 dataset="TCDTIMIT", test_dataset=None):
        # Audio
        self.AUDIO_LSTM_HIDDEN_LIST = AUDIO_LSTM_HIDDEN_LIST  # LSTM architecture for audio part
        self.audio_dataset = audio_dataset  # training here only works for TCDTIMIT at the moment; for that go to audioSR/RNN.py. This variable is used to get the stored results from that python script
        self.nbMFCCs = nbMFCCs
        self.audio_bidirectional = audio_bidirectional

        # Others
        self.runType = runType
        self.LR_start = LR_start
        self.forceTrain = forceTrain  # If False, just test the network outputs when the network already exists.
        # If forceTrain == True, train it anyway before testing
        # If True, set the LR_start low enough so you don't move too far out of the objective minimum

        self.dataset = dataset
        if test_dataset == None:   self.test_dataset = self.dataset
        else:                      self.test_dataset = test_dataset




def exportResultsToExcel(results, path):
    storePath = path.replace(".pkl", ".xlsx")
    import xlsxwriter
    workbook = xlsxwriter.Workbook(storePath)

    for runType in results.keys()[1:]:  # audio, lipreading, combined:
        worksheet = workbook.add_worksheet(runType)  # one worksheet per runType, but then everything is spread out...
        row = 0

        allNets = results[runType]

        # get and write the column titles

        # get the number of parameters. #for audio, only 1 value. For combined/lipreadin: lots of values in a dictionary
        try:
            nb_paramNames = allNets.items()[0][1][
                'nb_params'].keys()  # first key-value pair, get the value ([1]), then get names of nbParams (=the keys)
        except:
            nb_paramNames = ['nb_params']
        startVals = 4 + len(nb_paramNames)  # column number of first value

        colNames = ['Network Full Name', 'Network Name', 'Dataset', 'Test Dataset'] + nb_paramNames + ['Test Cost',
                                                                                                       'Test Accuracy',
                                                                                                       'Test Top 3 Accuracy',
                                                                                                       'Validation accuracy']
        for i in range(len(colNames)):
            worksheet.write(0, i, colNames[i])

        # write the data for each network
        for netName in allNets.keys():
            row += 1

            thisNet = allNets[netName]
            # write the path and name
            worksheet.write(row, 0, os.path.basename(netName))  # netName)
            worksheet.write(row, 1, thisNet['niceName'])
            if runType == 'audio':
                worksheet.write(row, 2, thisNet['audio_dataset'])
                worksheet.write(row, 3, thisNet['test_dataset'])
            else:
                worksheet.write(row, 2, thisNet['dataset'])
                worksheet.write(row, 3, thisNet['test_dataset'])

            # now write the params
            try:
                vals = thisNet['nb_params'].values()  # vals is list of [test_cost, test_acc, test_top3_acc]
            except:
                vals = [thisNet['nb_params']]
            for i in range(len(vals)):
                worksheet.write(row, 4 + i, vals[i])

            # now write the values
            vals = thisNet['values']  # vals is list of [test_cost, test_acc, test_top3_acc]
            for i in range(len(vals)):
                worksheet.write(row, startVals + i, vals[i])

    workbook.close()

    logger_combined.info("Excel file stored in %s", storePath)

def exportResultsToExcelManyNoise(resultsList, path):
    storePath = path.replace(".pkl", ".xlsx")
    import xlsxwriter
    workbook = xlsxwriter.Workbook(storePath)

    storePath = path.replace(".pkl", ".xlsx")
    import xlsxwriter
    workbook = xlsxwriter.Workbook(storePath)

    row = 0

    if len(resultsList[0]['audio'].keys()) > 0: thisRunType = 'audio'
    if len(resultsList[0]['lipreading'].keys()) > 0: thisRunType = 'lipreading'
    if len(resultsList[0]['combined'].keys()) > 0: thisRunType = 'combined'
    worksheetAudio = workbook.add_worksheet('audio');
    audioRow = 0
    worksheetLipreading = workbook.add_worksheet('lipreading');
    lipreadingRow = 0
    worksheetCombined = workbook.add_worksheet('combined');
    combinedRow = 0

    for r in range(len(resultsList)):
        results = resultsList[r]
        noiseType = results['resultsType']

        for runType in results.keys()[1:]:
            if len(results[runType]) == 0: continue
            if runType == 'audio': worksheet = worksheetAudio; row = audioRow
            if runType == 'lipreading': worksheet = worksheetLipreading; row = lipreadingRow
            if runType == 'combined': worksheet = worksheetCombined; row = combinedRow

            allNets = results[runType]

            # write the column titles
            startVals = 5
            colNames = ['Network Full Name', 'Network Name', 'Dataset', 'Test Dataset', 'Noise Type', 'Test Cost',
                        'Test Accuracy', 'Test Top 3 Accuracy']
            for i in range(len(colNames)):
                worksheet.write(0, i, colNames[i])

            # write the data for each network
            for netName in allNets.keys():
                row += 1

                thisNet = allNets[netName]
                # write the path and name
                worksheet.write(row, 0, os.path.basename(netName))  # netName)
                worksheet.write(row, 1, thisNet['niceName'])
                worksheet.write(row, 2, thisNet['dataset'])
                worksheet.write(row, 3, thisNet['test_dataset'])
                worksheet.write(row, 4, noiseType)

                # now write the values
                vals = thisNet['values']  # vals is list of [test_cost, test_acc, test_top3_acc]
                for i in range(len(vals)):
                    worksheet.write(row, startVals + i, vals[i])

            if runType == 'audio': audioRow = row
            if runType == 'lipreading': lipreadingRow = row
            if runType == 'combined': combinedRow = row

        row += 1

    workbook.close()

    logger_combined.info("Excel file stored in %s", storePath)

def getManyNetworkResults(networks, resultsType="unknownResults", roundParams=False, withNoise=False,
                          noiseType='white', ratio_dB=0):
    results = {'resultsType': resultsType}
    results['audio'] = {}
    results['lipreading'] = {}
    results['combined'] = {}

    failures = []

    for networkParams in tqdm(networks, total=len(networks)):
        logger_combined.info("\n\n\n\n ################################")
        logger_combined.info("Getting results from network...")
        logger_combined.info("Network properties: ")
        # pprint(vars(networkParams))
        try:
            if networkParams.forceTrain == True:
                runManyNetworks([networkParams])
            thisResults = {}
            thisResults['values'] = []
            thisResults['dataset'] = networkParams.dataset
            thisResults['test_dataset'] = networkParams.test_dataset
            thisResults['audio_dataset'] = networkParams.audio_dataset

            model_name, nice_name = getModelName(networkParams.AUDIO_LSTM_HIDDEN_LIST, networkParams.dataset)

            logger_combined.info("Getting results for %s", model_name + '.npz')
            network_train_info = getNetworkResults(model_name)
            if network_train_info == -1:
                raise IOError("this model doesn't have any stored results")
            # import pdb;pdb.set_trace()

            # audio networks can be run on TIMIT or combined as well
            if networkParams.runType != 'audio' and networkParams.test_dataset != networkParams.dataset:
                testType = "_" + networkParams.test_dataset
            else:
                testType = ""

            if roundParams:
                testType = "_roundParams" + testType

            if networkParams.runType != 'lipreading' and withNoise:
                thisResults['values'] = [
                    network_train_info['final_test_cost_' + noiseType + "_" + "ratio" + str(ratio_dB) + testType],
                    network_train_info['final_test_acc_' + noiseType + "_" + "ratio" + str(ratio_dB) + testType],
                    network_train_info['final_test_top3_acc_' + noiseType + "_" + "ratio" + str(ratio_dB) + testType]]
            else:
                try:
                    val_acc = max(network_train_info['val_acc'])
                except:
                    try:
                        val_acc = max(network_train_info['test_acc'])
                    except:
                        val_acc = network_train_info['final_test_acc']
                thisResults['values'] = [network_train_info['final_test_cost' + testType],
                                         network_train_info['final_test_acc' + testType],
                                         network_train_info['final_test_top3_acc' + testType], val_acc]

            thisResults['nb_params'] = network_train_info['nb_params']
            thisResults['niceName'] = nice_name

            # eg results['audio']['2Layer_256_256_TIMIT'] = [0.8, 79.5, 92,6]  #test cost, test acc, test top3 acc
            results[networkParams.runType][model_name] = thisResults

        except:
            logger_combined.info('caught this error: ' + traceback.format_exc());
            # import pdb;pdb.set_trace()
            failures.append(networkParams)

    logger_combined.info("\n\nDONE getting stored results from networks")
    logger_combined.info("####################################################")

    if len(failures) > 0:
        logger_combined.info("Couldn't get %s results from %s networks...", resultsType, len(failures))
        for failure in failures:
            pprint(vars(failure))
        if autoTrain or query_yes_no("\nWould you like to evalute the networks now?\n\n"):
            logger_combined.info("Running networks...")
            runManyNetworks(failures, withNoise=withNoise, noiseType=noiseType, ratio_dB=ratio_dB)
            mainGetResults(failures, withNoise=withNoise, noiseType=noiseType, ratio_dB=ratio_dB)

        logger_combined.info("Done training.\n\n")
        # import pdb; pdb.set_trace()
    return results


def getNetworkResults(save_name, logger=logger_combined):  # copy-pasted from loadPreviousResults
    if os.path.exists(save_name + ".npz") and os.path.exists(save_name + "_trainInfo.pkl"):
        old_train_info = unpickle(save_name + '_trainInfo.pkl')
        # import pdb;pdb.set_trace()
        if type(old_train_info) == dict:  # normal case
            network_train_info = old_train_info  # load old train info so it won't get lost on retrain

            if not 'final_test_cost' in network_train_info.keys():
                network_train_info['final_test_cost'] = min(network_train_info['test_cost'])
            if not 'final_test_acc' in network_train_info.keys():
                network_train_info['final_test_acc'] = max(network_train_info['test_acc'])
            if not 'final_test_top3_acc' in network_train_info.keys():
                network_train_info['final_test_top3_acc'] = max(network_train_info['test_topk_acc'])
        else:
            logger.warning("old trainInfo found, but wrong format: %s", save_name + "_trainInfo.pkl")
            # do nothing
    else:
        return -1
    return network_train_info


# networks is a list of dictionaries, where each dictionary contains the needed parameters for training
def runManyNetworks(networks, withNoise=False, noiseType='white', ratio_dB=0):
    results = {}
    failures = []
    if justTest:
        logger_combined.warning("\n\n!!!!!!!!! WARNING !!!!!!!!!!   \n justTest = True")
        if not query_yes_no("\nAre you sure you want to continue?\n\n"):
            return -1
    for network in tqdm(networks, total=len(networks)):
        print("\n\n\n\n ################################")
        print("Training new network...")
        print("Network properties: ")
        pprint(vars(network))
        try:
            model_save, test_results = runNetwork(AUDIO_LSTM_HIDDEN_LIST=network.AUDIO_LSTM_HIDDEN_LIST,
                                                  audio_features=network.audio_features,
                                                  audio_bidirectional=network.audio_bidirectional,
                                                  CNN_NETWORK=network.CNN_NETWORK,
                                                  cnn_features=network.cnn_features,
                                                  LIP_RNN_HIDDEN_LIST=network.LIP_RNN_HIDDEN_LIST,
                                                  lipRNN_bidirectional=network.lipRNN_bidirectional,
                                                  lipRNN_features=network.lipRNN_features,
                                                  DENSE_HIDDEN_LIST=network.DENSE_HIDDEN_LIST,
                                                  combinationType=network.combinationType,
                                                  dataset=network.dataset, datasetType=network.datasetType,
                                                  test_dataset=network.test_dataset,
                                                  addNoisyAudio=network.addNoisyAudio,
                                                  runType=network.runType,
                                                  LR_start=network.LR_start,
                                                  allowSubnetTraining=network.allowSubnetTraining,
                                                  forceTrain=network.forceTrain,
                                                  overwriteSubnets=network.overwriteSubnets,
                                                  audio_dataset=network.audio_dataset,
                                                  withNoise=withNoise, noiseType=noiseType, ratio_dB=ratio_dB)
            print(model_save)
            name = model_save + ("_Noise" + noiseType + "_" + str(ratio_dB) if withNoise else "")
            results[name] = test_results  # should be test_cost, test_acc, test_topk_acc

        except:
            print('caught this error: ' + traceback.format_exc());
            # import pdb;            pdb.set_trace()

            failures.append(network)
    print("#########################################################")
    print("\n\n\n DONE running all networks")

    if len(failures) > 0:
        print("Some networks failed to run...")
        # import pdb;pdb.set_trace()
    return results



if __name__ == "__main__":
    main()
