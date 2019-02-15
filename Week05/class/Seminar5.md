# Seminar 5: Recurrent Neural Networks (RNN)

In this seminar exercise, our goal are
* to provide an introduction to how to implement, train, and evaluate recurrent neural networks in TensorFlow, 
* show how to use a high-level API Keras, and
* demonstrate how to implement, train, and evaluate a recurrent neural network for text generation.


## Keras + Eager execution
Keras is a high-level API to build and train deep learning models (see official documents [here](https://keras.io/why-use-keras/) and [here](https://www.tensorflow.org/guide/keras)).  There are two ways to construct a model using Keras:
- Sequential model
- Functional API

### Sequential model
The `Sequential` model is a simple stack of layers. It cannot represent arbitrary models. If your model is simply a fully-connected neural network,  the `Sequential` model is sufficient and fast.  
```
model = tf.keras.Sequential([  
  tf.keras.layers.Dense(2, activation='relu', input_shape=(2, )),  
  tf.keras.layers.Dense(1, activation='sigmoid')])  
  
model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='binary_crossentropy', metrics=['accuracy'])  
  
model.fit(x, y, epochs=100, batch_size=4)
```
The methods under `Sequential` class are documented [here](https://keras.io/models/sequential/). 

### Functional API
The Keras functional API is the way to go for defining complex models. You can build a fully-customizable model by subclassing `tf.keras.Model` class.  
```
class MyModel(tf.keras.Model):
  def __init__(self):    
    super(MyModel, self).__init__()     
    self.dense_1 = tf.keras.layers.Dense(32, activation='relu') 
    self.dense_2 = tf.keras.layers.Dense(1, activation='sigmoid')  

  def call(self, inputs):        
    h1 = self.dense_1(inputs)    
    return self.dense_2(h1)
```
After defining your own model,  you can create an instance of the class, compile and fit the dataset same as when using `Sequential`:
```
# Get an instance of the class
model = MyModel()

model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='binary_crossentropy', metrics=['accuracy'])  
model.fit(x, y, epochs=100, batch_size=4)
```
The Keras functional API also shares the methods available in `Sequential`. 

###  Eager execution
TensorFlow's [eager execution](https://www.tensorflow.org/guide/eager) is an imperative programming environment that evaluates operations immediately, without building graphs. This makes it easy to get started with TensorFlow and debug models. 
Enable eager execution :
```
tf.enable_eager_execution()
```
### Keras functional API  + eager execution
Combining Keras functional API  and eager execution
```
# Get an instance of the class
model = MyModel()

for iter in range(n_iterations):
   with tf.GradientTape() as tape: 
      output = model(inputs) 
      loss = loss_func(targets, output) 
 
   grads = tape.gradient(loss, model.variables) 
   optimiser.apply_gradients(zip(grads, model.variables)) 
   loss_list.append(loss)
```
Everything in the block of `tf.GradientTape()` are operations that are recorded and at least one of their inputs is being "watched".

## RNN layers in TensorFlow Keras
The existing classes to build an RNN models in Tensorflow include `class Embedding`, `class RNN` , `class GRU`, `class CuDNNGRU`, `class LSTM`,  `class CuDNNLSTM`,  and so on.  `class CuDNN...` is the GPU version of `class ...`.  
1. `tf.keras.layers.Embedding`: 
```
__init__(input_dim, output_dim, embeddings_initializer='uniform',    embeddings_regularizer=None, activity_regularizer=None, embeddings_constraint=None, mask_zero=False, input_length=None, **kwargs)
```
Embedding layer can only be used the first layer of an RNN.  Word embedding refers to a class of methods for representing words and documents using a dense vector representation.  In an embedding, words are represented by dense vectors where a vector represents the projection of the word into a continuous vector space. 

There are various algorithms available to learn  the vector representations, including bag of words, Word2Vec, or simply a neural net layer.  

**Example:** if we have a document with a vocabulary size of 30, we want to embed each word in a vector space of dimension 20, we define the embedding layer as following:
```
# create an instance of the embedding class
embedding_layer = tf.keras.layers.Embedding(30, 20)
words_representations = embedding_layer(inputs)
```
The output `words_representations` of this layer is a 30x20 matrix. 

2. `tf.keras.layers.GRU`

<img src="https://github.com/lse-st449/lectures/raw/master/Week05/class/graphs/gru_cell.png"  width="750" height="450">

An implementation of Gated Recurrent Unit (GRU) - Cho et al 2014.
```
__init__(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False, **kwargs)
```
3. `tf.keras.layers.LSTM`

<img src="https://github.com/lse-st449/lectures/raw/master/Week05/class/graphs/LSTM_cell.png"  width="750" height="450">

An implementation of `Long Short-Term Memory (LSTM) unit - Hochreiter and Schmidhuber 1997`.  The constructor of the class is, 
```
__init__(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False, **kwargs)
```

4. `tf.keras.layers.RNN`
```
tf.keras.layers.RNN(cell, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)
```
This class allows you to define the computation in a cell in the RNN.  As explained in the [official document](https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN): 
> **`cell`**: A RNN cell instance or a list of RNN cell instances. A RNN cell is a class that has: - a `call(input_at_t, states_at_t)` method, returning `(output_at_t, states_at_t_plus_1)`

**Example:**
In the lecture, we saw a common RNN architecture

<img src="https://github.com/lse-st449/lectures/raw/master/Week05/class/graphs/rnn_a.png"  width="650" height="350">

```
class RNNCell(keras.layers.Layer):        
   def __init__(self, units, **kwargs):            
      self.units = units            
      self.state_size = units            
      super(RNNCell, self).__init__(**kwargs)        
   def build(self, input_shape):            
      self.u = self.add_weight(shape=(input_shape, self.units), initializer='uniform')                 
      self.w = self.add_weight(shape=(self.units, self.units),initializer='uniform') 
      self.v = self.add_weight(shape=(self.units, self.units),initializer='uniform')              
      self.built = True        
   def call(self, inputs, prev_state):                      
      h = K.dot(inputs, self.u)            
      state = tf.nn.tanh(h + K.dot(prev_state, self.w))
      output = tf.nn.softmax(K.dot(state, self.v))             
      return output, state
```
After defining the computation within a cell, 
```
 cell = RNNCell(32)     
 layer = tf.keras.layers.RNN(cell)    
 y = layer(inputs)
```

-   **`return_sequences`**: Boolean. Whether to return the last output in the output sequence, or the full sequence.
-   **`return_state`**: Boolean. Whether to return the last state in addition to the output.
-  **`go_backwards`**: Boolean (default False). If True, process the input sequence backwards and return the reversed sequence.
-   **`stateful`**: Boolean (default False). If True, the last state for each sample at index i in a batch will be used as initial state for the sample of index i in the following batch.
-   **`unroll`**: Boolean (default False). If True, the network will be unrolled, else a symbolic loop will be used. Unrolling can speed-up a RNN, although it tends to be more memory-intensive. Unrolling is only suitable for short sequences.

## Text generation with RNN
We will train a RNN using transcripts of Donald Trump's speeches in 2016 using a character-level LSTM model. Ideally, our model will be able to learn the structure and grammar from the transcripts and generate something similar. 

### General idea
For a character-level RNN model,  the training set and corresponding targets are demonstrated as below:
![alt text](https://github.com/lse-st449/lectures/raw/master/Week05/class/graphs/text.png)

The RNN is therefore designed to predict the sequence:
![alt text](https://github.com/lse-st449/lectures/raw/master/Week05/class/graphs/rnnblock.png)

### Pre-process text data
#### Upload file to Google Colab 
```
import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
from google.colab import files
files.upload()
```
If successful, you will see
![alt text](https://github.com/lse-st449/lectures/raw/master/Week05/class/graphs/upload_file.png)

#### Read from the file and count unique characters
```
text = open('trump_speeches.txt').read()
#text = open(path_to_file).read()
vocab = sorted(set(text))
vocab_size = len(vocab)
print('Data has {} characters, {} unique characters'.format(len(text), len(vocab)))
print(text[:100])
```
![alt text](https://github.com/lse-st449/lectures/raw/master/Week05/class/graphs/len_text_unique.png)

#### Map unique characters to indices
```
char_to_idx = {u: i for i, u in enumerate(vocab)}
print(char_to_idx)
print('The index for lowercase letter Q is:', char_to_idx['Q'])
```
![alt text](https://github.com/lse-st449/lectures/raw/master/Week05/class/graphs/char_to_idx.png)

#### Convert the full text into integer representations
```
int_text = np.array([char_to_idx[i] for i in text])
```
Show the first 17 mapping from text to integers, 
```
print ('{} --- are mapped to --- > {}'.format(text[:17], int_text[:17]))
```
![alt text](https://github.com/lse-st449/lectures/raw/master/Week05/class/graphs/char_mapped_to_int.png)

To decode the output of the model, we also need 
```
idx_to_char = np.array(vocab)
```
which maps indices back to characters.  

#### Split text into training inputs and targets
If the text we want to learn is `Learning`,  the training inputs will be the sequence `Learnin` and the corresponding target is `earning`. 
```
def split_input_target(data):
    input_eg = data[:-1] # get all elements in data except for the last one
    target_eg = data[1:] # omitting the first element
    return input_eg, target_eg
```
Use `tf.data.Dataset` to slice the entire dataset into batches with size of `seq_len + 1`, and then 
```
seq_len = 100
sections = tf.data.Dataset.from_tensor_slices(int_text).batch(seq_len+1, drop_remainder=True)
dataset = sections.map(split_input_target)
```

### Build the model
![alt text](https://github.com/lse-st449/lectures/raw/master/Week05/class/graphs/rnnfortext.png)
Typically, we call the constructor of a class by
```
tf.keras.layers.Embedding(input_dim, output_dim,...)
```
to make an instance.  
Normally, the model starts with an embedding layer followed by an LSTM layer or GRU layer.   
```
class Model(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(Model, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.CuDNNLSTM(self.hidden_size, return_sequences=True, recurrent_initializer='glorot_uniform', stateful=True)

        self.output_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, x):
        embedding = self.embedding(x)
        h = self.lstm(embedding)
        prediction = self.output_layer(h)
        return prediction
```
### Train the model
We choose batch size to be 64 and buffer size to be 10000, which is the number of elements from this dataset from which the new dataset will sample.
```
batch_size = 64
buffer_size = 10000
# hyperparameters
hidden_size = 1024
embedding_dim = 256

# build an instance of class Model
rnn_lstm = Model(vocab_size, embedding_dim, hidden_size)
# specify optimiser
optimiser = tf.train.AdamOptimizer()
# build the model
rnn_lstm.build(tf.TensorShape([batch_size, seq_len]))
```
We can print out the summary of the model using `rnn_lstm.summary()`, 

![alt text](https://github.com/lse-st449/lectures/raw/master/Week05/class/graphs/Param_count.png)

Then, we use `tf.train.Checkpoint` to save the parameters of the model after training,
```
# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# Checkpoint instance
checkpoint = tf.train.Checkpoint(optimizer=optimiser, model=rnn_lstm)
```
Train for 50 epochs and save the parameters of the model every 5 epochs, 
```
epochs = 50
loss_list = []
for epoch in range(epochs):
    data = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
    rnn_lstm.reset_states()
    
    for (batch, (inputs, targets)) in enumerate(data):
        with tf.GradientTape() as tape:
            output = rnn_lstm(inputs)
            loss = loss_func(targets, output)
        grads = tape.gradient(loss, rnn_lstm.variables)
        optimiser.apply_gradients(zip(grads, rnn_lstm.variables))
        loss_list.append(loss)

        if batch % 100 == 0 and batch != 0:
           print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, loss))
           
    if (epoch+1) % 5 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)
```
### Generate new text
![alt text](https://github.com/lse-st449/lectures/raw/master/Week05/class/graphs/rnnprediction.png)
Create a new instance, restore the parameter values saved at the latest checkpoint, and build a model with sequence length 1, 
```
model = Model(vocab_size, embedding_dim, hidden_size)
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))
# Generate text from trained model
print('Generated text after {} epoch training:'.format(epoch + 1))
text_generated = []
start_char = 'I'
generating_len = 1000
# convert the start string to an integer
start_int = [char_to_idx[i] for i in start_char]
start_int = tf.expand_dims(start_int, 0)
# Low temperatures results in more predictable text.
# Higher temperatures results in more surprising text.
# Experiment to find the best setting.
temperature = 0.8

rnn_lstm.reset_states()
for i in range(generating_len):
   predictions = model(start_int)
   predictions = tf.squeeze(predictions, 0)
   predictions = predictions / temperature
   pred_id = tf.multinomial(predictions, num_samples=1)[-1, 0].numpy()
   start_int = tf.expand_dims([pred_id], 0)
   text_generated.append(idx_to_char[pred_id])
print(start_char + ''.join(text_generated))
```
### Results

#### The evolution of text generated
After 5 epochs,
```
I
mespany to support them in from we do it. It’s not so a spent ngorigration with a said, "Are yought deally have a problem. Midicalization jobs. And the world, we have a was good for months. And we don’t know he said, "Tromployed is the best bat the money. I believe it. We are groups. I’ll tell you that we have things. And we’re so night? They said "What just made the things. They want to be probyme to all the group for years. I go all the look and I getter very much before. 
I send the Semorricr most of this mants. Geer me and we can do an unnel gavel. They don’t re-people you. Thank you. I know, how those people are try turner of many at the great establition in 111. Trade years. I don’t wanted to make us and we will sap from quess. 
…2....That’s the first time. But is a report in from – we would have all take too. You know what here’s the world." Ay the esenty, was going to muth poll. And we don’t know if we did. We would have terrifrout their trillers, and some pents are the country
```
After 20 epochs, 
```
I’d look at a crilitary community. Well, Iran leaderstands. They’re planus. 
I don’t want to stoppened to this small good for the votess. There’s not going to have a screling time to killed and they have a disantre of tive this stated by the way, and we need some pleated un tough case money, person countries and that’s going to win anymore. They’re not protected. You look at the time. So, I said, "You’re going to do it companies with the reason and I watched, and then you could have months can says "9 may group, a lot of paying with protems in terrorist. I’m saying this was going to get a total have and I would dakn a cromple are grough and lets. So Iran Masicit with Chinas. 
They don’t have to renow, look at the biggest in bad and we will fined out. That’s going to tell you that’s going to be great support of America from Brind. So the reasons he said, "He’ll get an the biggest conrate. Don’t want to have the people to the torsell. It’re totally to sebated to the wistors. We have thousan
```
After 50 epochs, 
```
I’ve got the country as a good ration. I said, "Many people are not going to deals the credit. 
And I’ve had an amazing because they don’t want to do this rune. 
I spent down that for less – I’m going to believe it. I wouldn’t want to take the leadersace. And it’s not all terrially destly right. 
Now how to know I live to be our so the repabling it. Whis is amazing troes that for dollars. And make our jobs. They don’t want to askay. 
And to years are wering proteclus. It’s here bord, that’s what I don’t have 10. Belicars, the people ching is he gons. It was also. They don’t want to tell you about this? But they were grauthing you bad 9D and we sent – I got it amazing – good speaking as an enample, and you look at this business assend how was tremendous campaign.". "I said, "Is if I said, "Look, "The and they don’t take a couple. Think they’re believe me and I mean, take a friend of the Srade Poter drove and I siement and what bring he stople and they are money to me termight, he said, "Dona
```
#### Loss plot
![alt text](https://gitlab.com/milanv/AI-and-Deep-Learning/raw/master/Seminars/Seminar5/graphs/loss.png)

