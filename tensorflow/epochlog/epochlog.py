import os

class BaseEpochLogger:
    """
    Base class that handles the actual storage of epoch number etc.
    Inheriting classes may handle loading and saving of weights differently,
    and may do things like storage of additional information.
    """
    def __init__(self, model, epochlog):
        """
        Initialize the EpochLogger with a model to save,
        and the logfile to store the epoch number in.

        model: a tensorflow.keras.Model that has a save_weights method,
            whose weights we want to save each epoch

        epochlog: a string path to the file where we want to record
            the number of the most recent log

        savepath: a string path to where we want the network weights saved
        """
        self.model = model
        self.logpath = epochlog
        self.weightspath = savepath
        # not currently excepting anything, should be catastrophic
        # in the future can create a new file if one does not exist
        try:
            self.epochlog = open(self.logpath, 'r+')
            self.num_epochs = int(self.epochlog.readline())
        except FileNotFoundError as e:
            if str(e) == "[Errno 2] No such file or directory: '" + self.logpath + "'":
                try:
                    os.makedirs(os.path.dirname(self.logpath))
                except:
                    pass
            else:
                os.makedirs(os.path.dirname(self.logpath))
            self.epochlog = open(self.logpath, 'w+')
            self.num_epochs = 0
            self.epochlog.write('0')
        
    def load_weights(self):
        # FIXME should get rid of this, decide whether to load weights upon function instantiation
        # (?) Decide on this!
        """
        load the weights
        """
        # if the number of epochs > 0, assume there are weights to load
        # if there are not weights to load, that means something has gone wrong
        # and we should fail fatally
        if self.num_epochs > 0:
            self.model.load_weights(self.weightspath)


    def epochs_done(self):
        """
        Returns the number of epochs done
        """
        return self.num_epochs

    def done_epoch(self):
        """
        Tell the EpochLogger that you finished training another epoch.
        NOTE: BEFORE calling this function, you should already have saved the weights
        """
        # save the weights themselves
#         self.model.save_weights(self.weightspath)
        # would have to do this in the inheriting method
    
    
        # iterate the number of epochs
        self.num_epochs += 1
        # go to the beginning of the epochlog, and delete everything
        self.epochlog.seek(0)
        #self.epochlog.truncate()
        # write the number of epochs that we've completed as a string
        self.epochlog.write(str(self.num_epochs))

    def __del__(self):
        """
        close the connection to the logger file
        """
        self.epochlog.close()
    
class EpochLogger:
    """
    Solves the problem of intermittent training.

    Say we want to train a network for N epochs,
    but something breaks after n epochs.

    The first thing we want to make sure we've done
    is saved weights at every epoch.
    Thus, we want to effortlessly check whether saved weights exist,
    and either load them or create a new file to save them in before
    we begin training.
    We then want an easy way to save after each epoch.

    But another problem arises if we've already trained for n epochs,
    and load the most recent weight: we now want to train for only
    the remaining (N-n) epochs. Thus, we want to keep track of
    the number n.

    This class is supposed to make it easy to do all this.
    After building and compiling the model, call (for example):

    epochlogger = EpochLogger(model, \"/path/to/epoch.log\", \"/path/to/weights\")
    epochlogger.load_weights()

    This should automatically load weights if they exist.

    Next, after each epoch of training, call:

    epochlogger.done_epoch()

    This will iterate the count of how many epochs have run, and save the weight.
    """
    def __init__(self, model, epochlog, savepath):
        """
        Initialize the EpochLogger with a model to save,
        and the logfile to store the epoch number in.

        model: a tensorflow.keras.Model that has a save_weights method,
            whose weights we want to save each epoch

        epochlog: a string path to the file where we want to record
            the number of the most recent log

        savepath: a string path to where we want the network weights saved
        """
        self.model = model
        self.logpath = epochlog
        self.weightspath = savepath
        # not currently excepting anything, should be catastrophic
        # in the future can create a new file if one does not exist
        try:
            self.epochlog = open(self.logpath, 'r+')
            self.num_epochs = int(self.epochlog.readline())
        except FileNotFoundError as e:
            if str(e) == "[Errno 2] No such file or directory: '" + self.logpath + "'":
                try:
                    os.makedirs(os.path.dirname(self.logpath))
                except:
                    pass
            else:
                os.makedirs(os.path.dirname(self.logpath))
            self.epochlog = open(self.logpath, 'w+')
            self.num_epochs = 0
            self.epochlog.write('0')
        
    def load_weights(self):
        # FIXME should get rid of this, decide whether to load weights upon function instantiation
        # (?) Decide on this!
        """
        load the weights
        """
        # if the number of epochs > 0, assume there are weights to load
        # if there are not weights to load, that means something has gone wrong
        # and we should fail fatally
        if self.num_epochs > 0:
            self.model.load_weights(self.weightspath)


    def epochs_done(self):
        """
        Returns the number of epochs done
        """
        return self.num_epochs

    def done_epoch(self):
        """
        Tell the EpochLogger that you finished training another epoch.
        """
        # save the weights themselves
        self.model.save_weights(self.weightspath)
        # iterate the number of epochs
        self.num_epochs += 1
        # go to the beginning of the epochlog, and delete everything
        self.epochlog.seek(0)
        #self.epochlog.truncate()
        # write the number of epochs that we've completed as a string
        self.epochlog.write(str(self.num_epochs))

    def __del__(self):
        """
        close the connection to the logger file
        """
        self.epochlog.close()

# class SimpleLogger:
#     """
#     For use with Keras, which continues to be surprisingly goofy.
#     Does not manage saving and loading weights, just writes to and reads from the log file.
#     Specifically, keeps track of last epoch number and most recent loss value.
#     """
#     def __init__(self, model, epochlog):
#         """
#         model: the model itself. Should already have had its weights loaded.
#         epochlog: the path to the file in which you want to store your log file.
#         """
#         self.model = model
#         self.logpath = epochlog
#         self.weightspath = savepath
#         # not currently excepting anything, should be catastrophic
#         # in the future can create a new file if one does not exist
#         try:
#             self.epochlog = open(self.logpath, 'r+')
#             self.num_epochs = int(self.epochlog.readline())
#         except FileNotFoundError as e:
#             if str(e) == "[Errno 2] No such file or directory: '" + self.logpath + "'":
#                 try:
#                     os.makedirs(os.path.dirname(self.logpath))
#                 except:
#                     pass
#             else:
#                 os.makedirs(os.path.dirname(self.logpath))
#             self.epochlog = open(self.logpath, 'w+')
#             self.num_epochs = 0
#             self.epochlog.write('0')
        
        