# based on the example from the "training loops" section of
# https://www.tensorflow.org/guide/basics#graphs_and_tffunction
import tensorflow as tf
import epochlog

# PLAN:
# * Create a random variable around a quadratic
# * Create a custom model that we can train to approximate the quadratic
#   from the random variable points as training data
# * Use EpochLogger as it is intended to be used (initiate, etc)
# * Try to load weights before they exist and the number of epochs is 0
# ** This should pass quietly despite weights not existing
# * Go through the training loop (this example was selected because
#   it involves a training loop rather than model.fit())
# * kill the EpochLogger instance, create a new model and a new Epochlogger instance
# * load weights in the new model, and check the number of epochs. Should be as we trained
# * run the new model on the same test data. Results should be good.
# * destroy all weight files and reset epoch.log to 0 for repeatability

# initialize the x-values of our data points
x = tf.linspace(-2, 2, 201)
x = tf.cast(x, tf.float32)

# define the function we want to fit the model to
def f(x):
    y = x**2 + 2*x - 5
    return y

# the "noisy" training data
y = f(x) + tf.random.normal(shape=[201])

# create the Model subclass we will use for other stuff
class Model(tf.Module):

      def __init__(self):
          # Randomly generate weight and bias terms
          rand_init = tf.random.uniform(shape=[3], minval=0., maxval=5., seed=22)
          # Initialize model parameters
          self.w_q = tf.Variable(rand_init[0])
          self.w_l = tf.Variable(rand_init[1])
          self.b = tf.Variable(rand_init[2])

      @tf.function
      def __call__(self, x):
          # Quadratic Model : quadratic_weight * x^2 + linear_weight * x + bias
          return self.w_q * (x**2) + self.w_l * x + self.b

# instantiate the model in question
quad_model = Model()

# define the mean-squared-error loss function
def mse_loss(y_pred, y):
    return tf.reduce_mean(tf.square(y_pred - y))

# now use mse_loss to "diagnose" the initial fit
print("initial quality of quad_model: " + str(mse_loss(quad_model(x), y)))

# initialize the EpochLogger
quadEpochLogger = epochlog.EpochLogger(quad_model, "epoch.log", "quad_model")

# load weights, despite there being none. This should not err.
quadEpochLogger.load_weights()

# check that the number of epochs trained is 0
if quadEpochLogger.epochs_done() == 0:
    print("all good, trained 0 epochs before starting")
else:
    raise Exception("Oops! non-zero number of epochs before starting training!")

# train the model
batch_size = 32
dataset = tf.data.Dataset.from_tensor_slices((x, y))
dataset = dataset.shuffle(buffer_size=x.shape[0]).batch(batch_size)
# Set training parameters
epochs = 100
learning_rate = 0.01
losses = []

# Format training loop
for epoch in range(epochs):
    for x_batch, y_batch in dataset:
        with tf.GradientTape() as tape:
            batch_loss = mse_loss(quad_model(x_batch), y_batch)
        # Update parameters with respect to the gradient calculations
        grads = tape.gradient(batch_loss, quad_model.variables)
        for g,v in zip(grads, quad_model.variables):
            v.assign_sub(learning_rate*g)
    # Keep track of model loss per epoch
    quadEpochLogger.done_epoch()
    print("epochs done according to logger: " + str(quadEpochLogger.epochs_done()))
    loss = mse_loss(quad_model(x), y)
    losses.append(loss)
    if epoch % 10 == 0:
        print(f'Mean squared error for step {epoch}: {loss.numpy():0.3f}')

# get the performance after training:
post_training_mse = mse_loss(quad_model(x), y)

# kill the logger:
del quadEpochLogger

# create a new model
model2 = Model()
# create a new logger
new_logger = epochlog.EpochLogger(model2, "epoch.log", "quad_model")

# load weights:
new_logger.load_weights()
# check that the number of epochs trained is epochs
if new_logger.epochs_done() == epochs:
    print("all good, trained 100 epochs before starting")
else:
    raise Exception("Oops! Supposedly trained " +
            str(new_logger.epochs_done()) + " epochs already")

# check the accuracy of the new model
post_load_mse = mse_loss(model2(x),y)

if post_load_mse == post_training_mse:
    print("both accuracies are the same after loading!")
else:
    print("after training, had: " + str(post_training_mse) + ", but after loading had: " +
        str(post_load_mse))

# Destroy everything!
print("since Daniel doesn\'t know what he is doing, won't delete anything. Reset it manually!")
