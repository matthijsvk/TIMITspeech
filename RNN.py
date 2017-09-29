from __future__ import print_function

import warnings
from time import gmtime, strftime

warnings.simplefilter("ignore", UserWarning)  # cuDNN warning

import logging
import tools.formatting as formatting

import time
program_start_time = time.time()

print("\n * Importing libraries...")
from RNN_implementation import *
from pprint import pprint   # printing properties of networkToRun objects
from tqdm import tqdm       # % done bar


##### SCRIPT META VARIABLES #####
VERBOSE = True
num_epochs = 20
nbMFCCs = 39  # num of features to use -> see 'utils.py' in convertToPkl under processDatabase
nbPhonemes = 39  # number output neurons

# Root path:  data is stored here, models and results will be stored here as well
# see README.md for info about where you should store your data
root = os.path.expanduser("~"+os.sep+"TCDTIMIT"+os.sep)

# Choose which datasets to run
datasets = ["TIMIT"]  # "TCDTIMIT", combined"

# for each dataset type as key this dictionary contains as value a list of all the network architectures that need to be trained for this dataset
many_n_hidden_lists = {}
many_n_hidden_lists['TIMIT'] = [[50]]#[8,8], [32], [64], [256], [512],[1024]]
                                #[8, 8], [32, 32], [64, 64], [256, 256], [512, 512]]#, [1024,1024]]
                                #[8,8,8],[32,32,32],[64,64,64],[256,256,256],[512,512,512],
                                #[8,8,8,8],[32,32,32,32],[64,64,64,64],[256,256,256,256],[512,512,512,512]]
                                #[512,512,512,512],
                                #[1024,1024,1024],[1024,1024,1024,1024]]
many_n_hidden_lists['TCDTIMIT'] = [[256, 256], [512, 512], [256, 256, 256, 256]]
# combined is TIMIT and TCDTIMIT put together
many_n_hidden_lists['combined'] = [[256, 256]]  # [32, 32], [64, 64], [256, 256]]#, [512, 512]]

#######################

bidirectional = True
add_dense_layers = False

run_test = True     # if network exists, just test, don't retrain but just evaluate on test set
autoTrain = False   # if network doesn't exist, train a new one automatically
round_params = False

# Decaying LR: each epoch LR = LR * LR_decay
LR_start = 0.01 # will be divided by 10 if retraining existing model
LR_fin = 0.0000001
# LR_decay = (LR_fin / LR_start) ** (1. / num_epochs)
LR_decay = 0.5

print("LR_start = %s", str(LR_start))
print("LR_fin = %s", str(LR_fin))
print("LR_decay = %s", str(LR_decay))


# quickly generate many networks
def create_network_list(dataset, networkArchs):
    network_list = []
    for networkArch in networkArchs:
        network_list.append(NetworkToRun(run_type="audio", n_hidden_list=networkArch,
                                          dataset=dataset, test_dataset=dataset, run_test=True))
    return network_list
    
def main():
    # Use this if you want only want to start training if the network doesn't exist
    network_list = create_network_list(dataset="TIMIT", networkArchs=many_n_hidden_lists['TIMIT'])

    net_runner = NetworkRunner(network_list)
    net_runner.get_network_results()
    print("\n got all results")
    net_runner.exportResultsToExcel()


class NetworkToRun:
    def __init__(self,
                 n_hidden_list=(256, 256,), nbMFCCs=39, audio_bidirectional=True,
                 LR_start=0.001, round_params=False, run_test=False, force_train=False,
                 run_type='audio', dataset="TCDTIMIT", test_dataset=None,
                 with_noise=False, noise_types=('white',), ratio_dBs=('0, -3, -5, -10',)):
        # Audio
        self.n_hidden_list = n_hidden_list  # LSTM architecture for audio part
        self.nbMFCCs = nbMFCCs
        self.audio_bidirectional = audio_bidirectional

        # Others
        self.run_type = run_type
        self.LR_start = LR_start
        self.run_test = run_test
        self.force_train = force_train  # If False, just test the network outputs when the network already exists.
        # If force_train == True, train it anyway before testing
        # If True, set the LR_start low enough so you don't move too far out of the objective minimum

        self.dataset = dataset
        if test_dataset == None:
            self.test_dataset = self.dataset
        else:
            self.test_dataset = test_dataset

        self.round_params = round_params
        self.with_noise = with_noise
        self.noise_types = noise_types
        self.ratio_dBs = ratio_dBs
        
        self.model_name, self.nice_name = self.get_model_name()
        self.model_path = self.get_model_path()
        self.model_path_noNPZ = self.model_path.replace('.npz','')

    # this generates the correct path based on the chosen parameters, and gets the train/val/test data
    def load_data(self, noise_type='white', ratio_dB='0'):
        dataset = self.dataset
        test_dataset = self.test_dataset
        with_noise = self.with_noise

        data_dir = os.path.join(root, self.run_type+"SR", dataset, "binary" + str(nbPhonemes),
                               dataset)  # output dir from datasetToPkl.py
        data_path = os.path.join(data_dir, dataset + '_' + str(nbMFCCs) + '_ch.pkl')
        if run_test:
            test_data_dir = os.path.join(root, self.run_type+"SR", test_dataset, "binary" + str(nbPhonemes) + \
                           ('_'.join([noise_type, os.sep, "ratio", str(ratio_dB)]) if with_noise else ""),test_dataset)
            test_data_path = os.path.join(test_data_dir, test_dataset + '_' + str(nbMFCCs) + '_ch.pkl')

        self.logger.info('  data source: %s', data_path)

        dataset = unpickle(data_path)
        x_train, y_train, valid_frames_train, x_val, y_val, valid_frames_val, x_test, y_test, valid_frames_test = dataset

        # if run_test, you can use another dataset than the one used for training for evaluation
        if run_test:
            self.logger.info("  test data source: %s", test_data_path)
            if with_noise:
                x_test, y_test, valid_frames_test = unpickle(test_data_path)
            else:
                _, _, _, _, _, _, x_test, y_test, valid_frames_test = unpickle(test_data_path)

        datasetFiles = [x_train, y_train, valid_frames_train, x_val, y_val, valid_frames_val, x_test, y_test,
                        valid_frames_test]
        # Print some information
        debug = False
        if debug:
            self.logger.info("\n* Data information")
            self.logger.info('X train')
            self.logger.info('  %s %s', type(x_train), len(x_train))
            self.logger.info('  %s %s', type(x_train[0]), x_train[0].shape)
            self.logger.info('  %s %s', type(x_train[0][0]), x_train[0][0].shape)
            self.logger.info('  %s', type(x_train[0][0][0]))

            self.logger.info('y train')
            self.logger.info('  %s %s', type(y_train), len(y_train))
            self.logger.info('  %s %s', type(y_train[0]), y_train[0].shape)
            self.logger.info('  %s %s', type(y_train[0][0]), y_train[0][0].shape)

            self.logger.info('valid_frames train')
            self.logger.info('  %s %s', type(valid_frames_train), len(valid_frames_train))
            self.logger.info('  %s %s', type(valid_frames_train[0]), valid_frames_train[0].shape)
            self.logger.info('  %s %s', type(valid_frames_train[0][0]), valid_frames_train[0][0].shape)

        return datasetFiles

    # this builds the chosen network architecture, loads network weights and compiles the functions
    def setup_network(self, batch_size):
        dataset = self.dataset
        test_dataset = self.test_dataset
        n_hidden_list = self.n_hidden_list
        round_params = self.round_params
        
        store_dir = root + dataset + "/results"
        if not os.path.exists(store_dir): os.makedirs(store_dir)

        # log file
        fh = self.setupLogging(store_dir)
        #############################################################


        self.logger.info("\n\n\n\n STARTING NEW TRAINING SESSION AT " + strftime("%Y-%m-%d %H:%M:%S", gmtime()))

        ##### IMPORTING DATA #####

        self.logger.info('  model target: %s', self.model_name)

        self.logger.info('\n* Building network using batch size: %s...', batch_size)
        RNN_network = NeuralNetwork('RNN', None, batch_size=batch_size,
                                    num_features=nbMFCCs, n_hidden_list=n_hidden_list,
                                    num_output_units=nbPhonemes,
                                    bidirectional=bidirectional, addDenseLayers=add_dense_layers,
                                    debug=False,
                                    dataset=dataset, test_dataset=test_dataset, logger=self.logger)

        # print number of parameters
        nb_params = lasagne.layers.count_params(RNN_network.network_lout_batch)
        self.logger.info(" Number of parameters of this network: %s", nb_params)

        # Try to load stored model
        self.logger.info(' Network built. \nTrying to load stored model: %s', self.model_name + '.npz')
        success = RNN_network.load_model(self.model_path_noNPZ, round_params=round_params)

        RNN_network.loadPreviousResults(self.model_path_noNPZ)

        ##### COMPILING FUNCTIONS #####
        self.logger.info("\n* Compiling functions ...")
        RNN_network.build_functions(train=True, debug=False)

        return RNN_network, success, fh

    def setupLogging(self, store_dir):
        self.logger = logging.getLogger(self.model_name)
        self.logger.setLevel(logging.DEBUG)
        FORMAT = '[$BOLD%(filename)s$RESET:%(lineno)d][%(levelname)-5s]: %(message)s '
        formatter = logging.Formatter(formatting.formatter_message(FORMAT, False))
        # formatter2 = logging.Formatter(
        #     '%(asctime)s - %(name)-5s - %(levelname)-10s - (%(filename)s:%(lineno)d): %(message)s')

        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        logFile = store_dir + os.sep + self.model_name + '.log'
        if os.path.exists(logFile):
            self.fh = logging.FileHandler(logFile)  # append to existing log
        else:
            self.fh = logging.FileHandler(logFile, 'w')  # create new logFile
        self.fh.setLevel(logging.DEBUG)
        self.fh.setFormatter(formatter)
        self.logger.addHandler(self.fh)

    def get_model_name(self):
        n_hidden_list = self.n_hidden_list
        dataset = self.dataset
        model_name = str(len(n_hidden_list)) + "_LSTMLayer" + '_'.join([str(layer) for layer in n_hidden_list]) \
                     + "_nbMFCC" + str(nbMFCCs) + ("_bidirectional" if bidirectional else "_unidirectional") + \
                     ("_withDenseLayers" if add_dense_layers else "") + "_" + dataset

        nice_name = "Audio:" + ' '.join(
            ["LSTM", str(n_hidden_list[0]), "/", str(len(n_hidden_list))])

        return model_name, nice_name

    def get_model_path(self):
        model_name, nice_name = self.get_model_name()
        model_path = os.path.join(root, self.run_type + 'SR', self.dataset, 'results', model_name + '.npz')
        return model_path

    # this takes the prepared data, built network and some parameters, and trains/evaluates the network
    def executeNetwork(self, RNN_network, load_params_success, batch_size, datasetFiles, 
                       noiseType='white', ratio_dB=0, fh=None):
        with_noise = self.with_noise
        run_test = self.run_test
        
        LR = LR_start
        if load_params_success == 0: LR = LR_start / 10.0

        ##### TRAINING #####
        self.logger.info("\n* Training ...")
        results = RNN_network.train(datasetFiles, self.model_path_noNPZ, num_epochs=num_epochs,
                                    batch_size=batch_size, LR_start=LR, LR_decay=LR_decay,
                                    compute_confusion=True, justTest=run_test, debug=False,
                                    withNoise=with_noise, noiseType=noiseType, ratio_dB=ratio_dB)

        self.fh.close()
        self.logger.removeHandler(self.fh)

        return results

    # estimate a good batchsize based on the size of the network
    def getBatchSizes(self):
        n_hidden_list = self.n_hidden_list
        if n_hidden_list[0] > 128:
            batch_sizes = [64, 32, 16, 8, 4]
        elif n_hidden_list[0] > 64:
            batch_sizes = [128, 64, 32, 16, 8, 4]
        else:
            batch_sizes = [256, 128, 64, 32, 16, 8, 4]
        return batch_sizes

    # run the network at the maximum batch size
    def runNetwork(self):

        batch_sizes = self.getBatchSizes()
        results = [0, 0, 0]
        for batch_size in batch_sizes:
            try:
                RNN_network, load_params_success, fh = self.setup_network(batch_size)
                datasetFiles = self.load_data()
                results = self.executeNetwork(RNN_network, load_params_success, batch_size=batch_size,
                                         datasetFiles=datasetFiles, fh=fh)
                break
            except:
                print('caught this error: ' + traceback.format_exc());
                self.logger.info("batch size too large; trying again with lower batch size")
                pass  # just try again with the next batch_size

        return results

    # run a network for all noise types
    def testAudio(self, batch_size, fh, load_params_success, RNN_network, 
                  with_noise=False, noiseTypes=('white',), ratio_dBs=(0, -3, -5, -10,)):
        for noiseType in noiseTypes:
            for ratio_dB in ratio_dBs:
                datasetFiles = self.load_data()
                self.executeNetwork(RNN_network=RNN_network, load_params_success=load_params_success, 
                                    batch_size=batch_size, datasetFiles=datasetFiles,
                               noiseType=noiseType, ratio_dB=ratio_dB, fh=fh)
                if not with_noise:  # only need to run once, not for all noise types as we're not testing on noisy audio anyway
                    return 0
        return 0

    # try different batch sizes for testing a network
    def testNetwork(self, with_noise, noiseTypes, ratio_dBs):

        batch_sizes = self.getBatchSizes()
        for batch_size in batch_sizes:
            try:
                RNN_network, load_params_success, fh = self.setup_network(batch_size)
                # evaluate on test dataset for all noise types
                self.testAudio(batch_size, fh, load_params_success, RNN_network,
                          with_noise, noiseTypes, ratio_dBs)
            except:
                print('caught this error: ' + traceback.format_exc());
                self.logger.info("batch size too large; trying again with lower batch size")
                pass  # just try again with the next batch_size

    def get_clean_results(self, network_train_info, nice_name, noise_type='white', ratio_dB='0'):
        with_noise = self.with_noise
        results_type = ("round_params" if self.round_params else "") + (
            "_Noise" + noise_type + "_" + str(ratio_dB) if with_noise else "")

        this_results = {'results_type': results_type}
        this_results['values'] = []
        this_results['dataset'] = self.dataset
        this_results['test_dataset'] = self.test_dataset
        this_results['audio_dataset'] = self.dataset

        # audio networks can be run on TIMIT or combined as well
        if self.run_type != 'audio' and self.test_dataset != self.dataset:
            test_type = "_" + self.test_dataset
        else:
            test_type = ""
        if self.round_params:
            test_type = "_round_params" + test_type
        if self.run_type != 'lipreading' and with_noise:
            this_results['values'] = [
                network_train_info['final_test_cost_' + noise_type + "_" + "ratio" + str(ratio_dB) + test_type],
                network_train_info['final_test_acc_' + noise_type + "_" + "ratio" + str(ratio_dB) + test_type],
                network_train_info['final_test_top3_acc_' + noise_type + "_" + "ratio" + str(ratio_dB) + test_type]]
        else:
            try:
                val_acc = max(network_train_info['val_acc'])
            except:
                try:
                    val_acc = max(network_train_info['test_acc'])
                except:
                    val_acc = network_train_info['final_test_acc']
            this_results['values'] = [network_train_info['final_test_cost' + test_type],
                                      network_train_info['final_test_acc' + test_type],
                                      network_train_info['final_test_top3_acc' + test_type], val_acc]
        this_results['nb_params'] = network_train_info['nb_params']
        this_results['niceName'] = nice_name

        return this_results

    def get_network_train_info(self, save_path,):
        save_name = save_path.replace('.npz','')
        if os.path.exists(save_path) and os.path.exists(save_name + "_trainInfo.pkl"):
            network_train_info = unpickle(save_name + '_trainInfo.pkl')

            if not 'final_test_cost' in network_train_info.keys():
                network_train_info['final_test_cost'] = min(network_train_info['test_cost'])
            if not 'final_test_acc' in network_train_info.keys():
                network_train_info['final_test_acc'] = max(network_train_info['test_acc'])
            if not 'final_test_top3_acc' in network_train_info.keys():
                network_train_info['final_test_top3_acc'] = max(network_train_info['test_topk_acc'])
            return network_train_info
        else:
            return -1
        
        
# This class runs many networks, storing the weight files, training data and log in the appropriate location
# It can also just evaluate a network on a test set, or simply load the results from previous runs
# All results are stored in an Excel file.
# Audio networks can also be evaluated on audio data which has been polluted with noise (set the appropriate parameters in the declaration of the networks you want to test)
# It receives a network_list as input, containing NetworkToRun objects that specify all the relevant parameters to define a network and run it.
class NetworkRunner:
    def __init__(self, network_list):
        self.network_list = network_list
        self.results = {}
        self.setup_logging(root)

    def setup_logging(self,store_dir):
        self.logger = logging.getLogger('audioNetworkRunner')
        self.logger.setLevel(logging.DEBUG)
        FORMAT = '[$BOLD%(filename)s$RESET:%(lineno)d][%(levelname)-5s]: %(message)s '
        formatter = logging.Formatter(formatting.formatter_message(FORMAT, False))
        # formatter2 = logging.Formatter(
        #     '%(asctime)s - %(name)-5s - %(levelname)-10s - (%(filename)s:%(lineno)d): %(message)s')

        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        logFile = store_dir + os.sep + 'audioNetworkRunner' + '.log'
        if os.path.exists(logFile):
            self.fh = logging.FileHandler(logFile)  # append to existing log
        else:
            self.fh = logging.FileHandler(logFile, 'w')  # create new logFile
        self.fh.setLevel(logging.DEBUG)
        self.fh.setFormatter(formatter)
        self.logger.addHandler(self.fh)

    # this loads the specified results from networks in network_list
    # more efficient if we don't have to reload each network file; just retrieve the data you found earlier
    def get_network_results(self):
        self.results_path = root + 'storedResults' + ".pkl"
        try:
            prev_results = unpickle(self.results_path)
        except:
            prev_results = {}

        # get results for networks that were trained before. If a network has run_test=True, run it on the test set
        results, to_retrain, failures = self.get_trained_network_results(self.network_list)

        # failures mean that the network does not exist yet. We have to generate and train it.
        if len(failures) > 0:
            results2 = self.train_networks(failures)
            results.update(results2)

        # to_retrain are networks that had forceTrain==True. We have to train them.
        if len(to_retrain) > 0:
            results3 = self.train_networks(to_retrain)
            results.update(results3)

        # update and store the results
        prev_results.update(results)
        saveToPkl(self.results_path, prev_results)
        self.results = prev_results

    # train the networks
    def train_networks(self, networks):
        results = []
        self.logger.info("Couldn't get results from %s networks...", len(networks))
        for network in networks:
            pprint(vars(network))
        if autoTrain or query_yes_no("\nWould you like to train the networks now?\n\n"):
            self.logger.info("Running networks...")

            failures = []
            for network in tqdm(networks, total=len(networks)):
                print("\n\n\n\n ################################")
                print("Training new network...")
                print("Network properties: ")
                pprint(vars(network))
                try:
                    network.runNetwork()
                except:
                    print('caught this error: ' + traceback.format_exc());
                    pprint(vars(network))
                    failures.append(network)

            if len(failures) > 0:
                print("Some networks failed to train...")
                import pdb; pdb.set_trace()

            # now the networks are trained, we can load their results
            results, _, failures = self.get_trained_network_results(networks)

        return results

    # get the stored results from networks that were trained previously
    # for networks that have run_test=True, run the network on the test set before getting the results
    # for networks that have forceTrain=True, add to 'to_retrain' list
    # for networks that fail to give results (most probably because they haven't been trained yet), add to 'failures' list
    def get_trained_network_results(self, networks):
        results = {}
        results['audio'] = {}
        results['lipreading'] = {}
        results['combined'] = {}

        failures = []
        to_retrain = []

        for network in tqdm(networks, total=len(networks)):
            self.logger.info("\n\n\n\n ################################")
            # pprint(vars(network_params))
            try:
                # if forced test evaluation, test the network before trying to load the results
                if network.run_test == True:
                    network.testNetwork()

                # if forced training, we don't need to get the results. That will only be done after retraining
                if network.force_train == True:
                    to_retrain.append(network)
                    # by leaving here we make sure the network is not appended to the failures list 
                    # (that happens when we try to get its stored results while it doesn't exist, and we haven't done that yet here.
                    continue

                model_name, nice_name = network.get_model_name()
                model_path = network.get_model_path()

                self.logger.info("Getting results for %s", model_path)
                network_train_info = network.get_network_train_info(model_path)
                if network_train_info == -1:
                    raise IOError("this model doesn't have any stored results")

                if network.with_noise:
                    for noise_type in network.noise_types:
                        for ratio_dB in network.ratio_dBs:
                            this_results = network.get_clean_results(network_train_info=network_train_info, 
                                                                     nice_name=nice_name, 
                                                                     noise_type=noise_type, 
                                                                     ratio_dB=ratio_dB)

                            results[network.run_type][model_name] = this_results
                else:
                    this_results = network.get_clean_results(network_train_info=network_train_info, 
                                                            nice_name=nice_name)
                                                             
                    # eg results['audio']['2Layer_256_256_TIMIT']['values'] = [0.8, 79.5, 92,6]  #test cost, acc, top3 acc
                    results[network.run_type][model_name] = this_results

            except:
                self.logger.info('caught this error: ' + traceback.format_exc());
                failures.append(network)

        self.logger.info("\n\nDONE getting stored results from networks")
        self.logger.info("####################################################")

        return results, to_retrain, failures
    
    
    def exportResultsToExcel(self):
        path =self.results_path
        results = self.results

        storePath = path.replace(".pkl", ".xlsx")
        import xlsxwriter
        workbook = xlsxwriter.Workbook(storePath)
    
        for run_type in results.keys()[1:]:  # audio, lipreading, combined:
            worksheet = workbook.add_worksheet(run_type)  # one worksheet per run_type, but then everything is spread out...
            row = 0
    
            allNets = results[run_type]
    
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
                if run_type == 'audio':
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
    
        self.logger.info("Excel file stored in %s", storePath)
        self.fh.close()
        self.logger.removeHandler(self.fh)
    
    def exportResultsToExcelManyNoise(self, resultsList, path):
        storePath = path.replace(".pkl", ".xlsx")
        import xlsxwriter
        workbook = xlsxwriter.Workbook(storePath)
    
        storePath = path.replace(".pkl", ".xlsx")
        import xlsxwriter
        workbook = xlsxwriter.Workbook(storePath)
    
        row = 0
    
        if len(resultsList[0]['audio'].keys()) > 0: thisrun_type = 'audio'
        if len(resultsList[0]['lipreading'].keys()) > 0: thisrun_type = 'lipreading'
        if len(resultsList[0]['combined'].keys()) > 0: thisrun_type = 'combined'
        worksheetAudio = workbook.add_worksheet('audio');
        audioRow = 0
        worksheetLipreading = workbook.add_worksheet('lipreading');
        lipreadingRow = 0
        worksheetCombined = workbook.add_worksheet('combined');
        combinedRow = 0
    
        for r in range(len(resultsList)):
            results = resultsList[r]
    
            for run_type in results.keys()[1:]:
                if len(results[run_type]) == 0: continue
                if run_type == 'audio': worksheet = worksheetAudio; row = audioRow
                if run_type == 'lipreading': worksheet = worksheetLipreading; row = lipreadingRow
                if run_type == 'combined': worksheet = worksheetCombined; row = combinedRow
    
                allNets = results[run_type]
    
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
                    worksheet.write(row, 4, thisNet['results_type'])
    
                    # now write the values
                    vals = thisNet['values']  # vals is list of [test_cost, test_acc, test_top3_acc]
                    for i in range(len(vals)):
                        worksheet.write(row, startVals + i, vals[i])
    
                if run_type == 'audio': audioRow = row
                if run_type == 'lipreading': lipreadingRow = row
                if run_type == 'combined': combinedRow = row
    
            row += 1
    
        workbook.close()
    
        self.logger.info("Excel file stored in %s", storePath)

        self.fh.close()
        self.logger.removeHandler(self.fh)


if __name__ == "__main__":
    main()
