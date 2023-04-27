import os

class BaseEpochLogger:
    """
    Base class that handles the actual storage of epoch number etc.
    Inheriting classes may handle loading and saving of weights differently,
    and may do things like storage of additional information.
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
        
        self.log_vals = []
        try:
            self.epochlog = open(self.logpath, 'r+')
#             self.num_epochs = int(self.epochlog.readline())
            for line in self.epochlog:
                self.log_vals.append(line)
            self.num_epochs = int(self.log_vals[0])
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
            # apparently, this written '0' does not get flushed to the file
            # this means that if there is, e.g. an error while calculating the gradient,
            # on the next run we will load an existing, empty file from which we will be unable to read the epoch.
            # Thus, must flush what we've written
            self.epochlog.flush()
        
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
        Also, if you want to write extra stuff to the file, put put it into the log_vals list
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
        #self.epochlog.write(str(self.num_epochs))
        self.log_vals[0] = str(self.num_epochs)
        for value in self.log_vals:
            self.epochlog.write(value + '\n')

    def __del__(self):
        """
        close the connection to the logger file
        """
        self.epochlog.close()
    
class EpochLogger(BaseEpochLogger):
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

    def done_epoch(self):
        """
        Tell the EpochLogger that you finished training another epoch.
        """
        # save the weights themselves
        self.model.save_weights(self.weightspath)
        
        super().done_epoch()


class KerasEpochLogger(BaseEpochLogger):
    """
    An EpochLogger for use with keras-style training,
    i.e. fit().
    The requirement here is that this class should have a
    callback method that both saves the epoch and calls the
    appropriate ModelCheckpoint callback function
    """
    def __init__(self, model, epochlog, savepath, model_checkpoint_object):
        """
        model_checkpoint_object is of type tf.keras.callbacks.ModelCheckpoint.
        We need this because we later want to set the .best field to the value loaded from
        the file
        """
        super().__init__(model, epochlog, savepath)
        self.mco = model_checkpoint_object
        if len(self.log_vals) == 2:
            self.mco.best = float(self.log_vals[1])
    
    def done_epoch(self):
        """
        Should have called the mco's callback function by now
        This means that mco.best has been updated, and we can write that
        into the log file
        """
        if len(self.log_vals) > 1:
            self.log_vals[1] = str(mco.best)
        else:
            self.log_vals.append(str(mco.best))
        super().done_epoch()