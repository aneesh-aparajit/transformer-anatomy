# Attention is All you Need

<p align="center">
  <img src="https://hub.packtpub.com/wp-content/uploads/2018/04/Attention.png" />
</p>

## The Encoder
Each embedding layer receives a sequence of embeddings and feeds them through the following through the following sublayers:
- A multi-head self-attention layer
- A fully connected FFN that is applied to each input embedding

The output embeddings of each encoder layer have the same size as the inputs, and we'll soon see that the main role of the encoder stack is to "update" the input embeddings to produce representations that encode some contextual information in the sequence.

Each of these sublayers use skip connections and layer normalization, which are standard tricks to train deep neural networks effectively. But to truly understand what makes a transformer work, we need to go deeper.

### Self-Attention
- __Attention__ is a mechanism that allows neural networks to assign a different amount of weight or "attention" to each element in the sequence. 
- The "self" part in the "self-attention" refers to the fact that these weights are computed for all hidden states in the same set $-$ for example, all the hidden states in the encoder. By contrast, the attention mechanism associated with recurrent models involves computing the relevance of each encoder hidden state to the decoder hidden state at a given decoder timestep.

The main idea behind self-attention is that instead of using fixed embedding for each token, we can use the whole sequence to compute a _weighted average_ of each embedding. Another way to formulate this is to say that given a sequence of token embeddings $(x_1, x_2, ..., x_n)$, self-attention produces a new embedding $(x_1', x_2', ..., x_n')$ where each $x_j'$ is a linear combination of $x_j$.

$$x_i' = \sum_{j=1}^nw_{ji}x_j$$

The coefficients $w_{ji}$ are called _attention weights_ and are normalized so that $\sum_j{w_ji} = 1$.

### Scaled Dot Product Attention
1. Project each token embedding into three vectors called __query__, __key__ and __value__.

$$\text{Attention}(Q, K, V) = \text{Softmax}(\frac{Q\cdot K^T}{\sqrt{d_h}})\cdot V$$

### Multi-Head Attention
Till now, we used the embeddings as such, but in reality, we apply three different transformations on the embeddings to generate the query, key and the value vectors. These transformations project the embeddings and each projection carries its own set of learnable parametera, which allows the self-attention layer to focus on different semantic aspects of the sequence.

It also turned out to be beneficial to have multiple sets of linear projections, each one representing a so-called attention head. 

![mha](https://i.ytimg.com/vi/A1eUVxscNq8/maxresdefault.jpg)

> __But, why do we need more than one attention head?__ <br>
The reason is that softmaxof the one head tends to focus more on mostly one aspect of the similarity. Having several head allow the model to focus on several aspects at once. 