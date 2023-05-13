# Sequence to Sequence Models for Language Processing

Before exploring the specific mathematical concepts of sequence to sequence models, let's begin with high-level understanding of what goes on behind the scenes of language translation. Consider the early attempts in the past, when dictionaries or grammar books were relied upon for word translations, and contrast that with modern machine translation systems like Google Translate. Sequence to sequence models for machine translation essentially operate on a similar principle but with the use of advanced algorithms to perform translations. These evolutionary translations systems such as machine language translators have fascinated humans for centuries. Machine translation technology came after the advent of digital computing and the advancements in technology have significantly improved our ability to overcome language barriers, which has long been a fundamental challenge for linguists, computer scientists, and language learners alike. They have been adapted in technology and are used to translate between human languages such as the online translation websites that people use to assimilate content that is not in their native language (Neubig 2017). The machine translation system as part of the sequence to sequence models operates on the same principle of Google's translation system, where we have an input and an output. The input corresponds to the source language while the output to the target language. Machine translation can be described as the process of transforming a sequence of words in the source language into an effective model that enables accurate conversion across a wide range of languages (Neubig 2017). Machine translation is a well known practical example of  sequence to sequence models.

One aspect that is particularly motivating about sequence to sequence models for language translation is their ability to bridge communication gaps and foster understanding across different languages and cultures. By leveraging sequence to sequence models, we can facilitate accurate translations between languages, enabling people to connect, share ideas and collaborate on a global scale.

## What is sequence to sequence models and its Application to Natural language translation?

Sequence to sequence models which are also referred to as Seq2Seq models are a type of neural network architecture that are designed to map sequences and are applicable for tasks where the input and output are both sequences of variable length. Seq2Seq models are particularly popular and effective in a field with language translation tasks, which includes text summarization, dialogue generation, conversation modeling and machine translation, where the goal is to convert text sequence from one form of sequences to another. They have also been extended to handle more complex input and output types, such as image captioning and speech recognition (Cho 2014). Seq2Seq models encompass a wider range of models that involve mapping one sequence to another as mentioned earlier. While machine translation falls within this category, it also encompasses a diverse range of other methods utilized for various tasks (Neubig 2017). Consider a computer program as a system that receives a sequence of input bits and produces a sequence of output bits. Examples of this sequence conversion are shown below.

![](https://github.com/wanyic/seq2seq/tree/main/img/image4.png)

# Application

Now that we have gained some mathematical knowledge about sequence to sequence models in machine translation, we can apply them using existing data. For this part, we will be building a charachter level seq2seq model for language translation. The [data](https://www.manythings.org/anki/) we are going to use is a data frame with sentence pairs in which the first column is the input texts (English texts) and the second column is the target texts (French texts). Small part of the data is also shown below. Notice that sometimes the same English phrase can have different translations in French. The dimension of our data set is 10000 rows of sentence pairs and 2 columns.

| Input   | Target                       | 
|:--------|:-----------------------------|
|  Go.    |                       Va !   |
|  Go.    |                    Marche.   |
|  Hi.    |                    Salut !   |
|  Hi.    |                     Salut.   |
| Run!    |                    Cours !   |
| Run!    |                   Courez !   |
| Begin.  |                   Commencez. |
| Begin.  |                     Commence.|

The R packages we will be using for the model implementation are shown below. The **keras** package is a high-level neural networks API developed with a focus on enabling fast experimentation which is very useful in Seq2Seq modeling. The **data.table** and **stringr** packages are used to help us read in the text data file and clean or modify the data to our desired form.

```r
library(keras)
library(data.table)
library(stringr)
```

## Data Vectorization

Before we start modeling, it is important that we have the data vectorized so that the encoder-decoder part of the model can process our input texts and target texts. Since our focus is a character level seq2seq model, we need to split the texts into letters and maps them by creating vectors of the unqiue letters (for word level seq2seq models, the texts will be spilted into unique words instead of letters). For the target texts, we also add string "\t" denoting begin of sentence and string "\n" denoting end of sentence, so that LSTM starts making predictions when the starting symbol "\t" is encountered and stops predicting when ending symbol "\n" is encountered.

```r
# vectorize the data.
input_texts  <- text[[1]]
input_texts  <- lapply( input_texts, function(s) strsplit(s, split="")[[1]])
# add bos and eos to the target texts
target_texts <- paste0("\t", text[[2]], "\n")
target_texts <- lapply( target_texts, function(s) strsplit(s, split="")[[1]])

input_characters  <- sort(unique(unlist(input_texts)))
target_characters <- sort(unique(unlist(target_texts)))
# based on the unique characters, we create tokens for the texts
num_encoder_tokens <- length(input_characters)
num_decoder_tokens <- length(target_characters)
max_encoder_seq_length <- max(sapply(input_texts,length))
max_decoder_seq_length <- max(sapply(target_texts,length))
```

> Number of samples: 10000 
> 
> Number of unique input tokens: 70
> 
> Number of unique output tokens: 93
> 
> Max sequence length for inputs: 14
> 
> Max sequence length for outputs: 59

## Encoder-Decoder Modeling

For model tuning, there are several parameters that we may want to vary based on the purpose of the model in order to get more accurate predictions. The parameters are the batch size, epochs, and latent dimensionality of the encoding space. When modeling the encoder-decoder part with the LSTM layers, it is very important that we use a suitable "units" input. The units parameter in the LSTM layer is the latent dimensionality of the encoding space, which gives the model its capacity to learn the data and like hidden layers/neurons in a feed-forward algorithm, we don't want it to be too small (under-fit) or too large (over-fit). Given the number of samples that we have (10000 sentence pairs), we can use 350 units for the latent dimension. In the case that we chose a unit value that is too small, the model usually fails to learn the targeted sequences and produced very interesting and funny outputs that does not make sense. Two examples of funny outputs are shown below:

> Input sentence  :  I'm no rebel. 
> 
> Target sentence :  Je ne suis pas un rebelle. 
> 
> Decoded sentence:  Je se sais ais ss ais ss aiss ss aiss sss ss ais ss ais ss 

> Input sentence  :  It's possible. 
> 
> Target sentence :  C'est possible. 
> 
> Decoded sentence:  C'es s ais ais ss aiss ss ss ais ss ais ss aiss ss aiss ss 

Below we have the encoder and decoder to process the input sequence with a lstm layer unit of 350. Note that the overall structure of our model is implemented based on the character level Seq2Seq model example from the [Keras page](https://keras.io/examples/nlp/lstm_seq2seq/) and R codes from Keras' [Github repository](https://github.com/rstudio/keras/blob/main/vignettes/examples/lstm_seq2seq.R).

```r
# lstm layer units/latent dimension
latent_dim <- 350

# define an input sequence and process it.
encoder_inputs  <- layer_input(shape=list(NULL, num_encoder_tokens))
encoder <- layer_lstm(units=latent_dim, return_state=TRUE)
encoder_results <- encoder_inputs %>% encoder
# discard encoder_outputs and keep the states for decoder.
encoder_states  <- encoder_results[2:3]

# use encoder_states as initial state.
decoder_inputs  <- layer_input(shape=list(NULL, num_decoder_tokens))
# return full output sequences from decoder and return internal states for inferences
decoder_lstm <- layer_lstm(units=latent_dim, return_sequences=TRUE,
                           return_state=TRUE)
decoder_results <- decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense <- layer_dense(units=num_decoder_tokens, activation='softmax')
decoder_outputs <- decoder_dense(decoder_results[[1]])
```

Then, using the processed input data from the encoder and decoder we can define the main model. The optimizer for our model is rmsprop which is similar to the gradient descent algorithm with momentum and usually works well with neuron networks. We also have categorical cross-entrophy as the loss function since we are working with text data. Additionally, the batch size for our model is 128 since we have a large dataset, we want a larger batch size that can give faster progress in training.
### Header 3

```js
// Javascript code with syntax highlighting.
var fun = function lang(l) {
  dateformat.i18n = require('./lang/' + l)
  return true;
}
```

```ruby
# Ruby code with syntax highlighting
GitHubPages::Dependencies.gems.each do |gem, version|
  s.add_dependency(gem, "= #{version}")
end
```

#### Header 4

*   This is an unordered list following a header.
*   This is an unordered list following a header.
*   This is an unordered list following a header.

##### Header 5

1.  This is an ordered list following a header.
2.  This is an ordered list following a header.
3.  This is an ordered list following a header.

###### Header 6

| head1        | head two          | three |
|:-------------|:------------------|:------|
| ok           | good swedish fish | nice  |
| out of stock | good and plenty   | nice  |
| ok           | good `oreos`      | hmm   |
| ok           | good `zoute` drop | yumm  |

### There's a horizontal rule below this.

* * *

### Here is an unordered list:

*   Item foo
*   Item bar
*   Item baz
*   Item zip

### And an ordered list:

1.  Item one
1.  Item two
1.  Item three
1.  Item four

### And a nested list:

- level 1 item
  - level 2 item
  - level 2 item
    - level 3 item
    - level 3 item
- level 1 item
  - level 2 item
  - level 2 item
  - level 2 item
- level 1 item
  - level 2 item
  - level 2 item
- level 1 item

### Small image

![Octocat](https://github.githubassets.com/images/icons/emoji/octocat.png)

### Large image

![Branching](https://guides.github.com/activities/hello-world/branching.png)


### Definition lists can be used with HTML syntax.

<dl>
<dt>Name</dt>
<dd>Godzilla</dd>
<dt>Born</dt>
<dd>1952</dd>
<dt>Birthplace</dt>
<dd>Japan</dd>
<dt>Color</dt>
<dd>Green</dd>
</dl>

```
Long, single-line code blocks should not wrap. They should horizontally scroll if they are too long. This line should be long enough to demonstrate this.
```

```
The final element.
```
