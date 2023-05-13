

Text can be **bold**, _italic_, or ~~strikethrough~~.

[Link to another page](./another-page.html).

There should be whitespace between paragraphs.

There should be whitespace between paragraphs. We recommend including a README, or a file with information about your project.

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

cat('Number of samples:', length(input_texts),'\n')
cat('Number of unique input tokens:', num_encoder_tokens,'\n')
cat('Number of unique output tokens:', num_decoder_tokens,'\n')
cat('Max sequence length for inputs:', max_encoder_seq_length,'\n')
cat('Max sequence length for outputs:', max_decoder_seq_length,'\n')
```

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
