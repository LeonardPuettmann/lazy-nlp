# 🦥 LazyNLP - label data and train models with low effort

This library allows you to label data using zeroshot and train simple classifiers - without the need to do anything! 

This library is for : 
- ✅ lazy people
- ✅ fast prototyping or quick experiments
- ❌ super accurate results 
- ❌ production-grade models
- ❌ every use-case 

## What you need 

To use this library, all you need to do is: 
- Have a list containing texts
- Have a second list containing all the possible labels you want to assign
- (When using from GitHub)`pip install -r requirements.txt`

## Quickstart

First, install LazyNLP with `pip install lazy-nlp`. 

Then you could use LazyNLP like this:

```python
# import LazyNLP
from lazy_nlp import LazyNLP

# load in some data set
df = pd.read_csv("some_text_dataset.csv")

# convert the text column to a list
sentences = df["text"].to_list()

# write a list with all possible labels
labels = ["positive", "neutral", "negative"]

# LazyNLP will handle the rest for you
lnlp = LazyNLP()
model, encoder = lnlp.run(sentences, labels)
```
### Save a model
The result of `.run` will be a `pytorch` model and a labelencoder. You can save these by using `lnlp.save(model, encoder)`

### Using the model
After you saved the model and encoder, you can simply predict on new data like this: 
```python
# pass a list to .predict with a list of new data. You need to .save the model and encoder before!
preds = lnlp.predict(["This is a new sentence"])
```

## LazyNLP steps

LazyNLP consists of three steps: zeroshot labeling, embedding and model training. With `.run` you trigger all of these steps at once, but you can also use LazyNLP only for the zeroshot, embedding or model training component idividually. 

### Zeroshot

```python
# provide a list of texts you want to label as well as a list of all potential labels
your_texts = ["This is a bad sentence.", "This is another sentence.", "More sentences!", "I, too, am a sentence", "This is a good sentence."]
your_labels = ["negative", "neutral", "positive"]

# returns a list of zeroshot labels
zeroshot_labels = lnlp.zeroshot(your_texts, your_labels)
```

### Embedding

```python
your_texts = ["This is a bad sentence.", "This is another sentence.", "More sentences!", "I, too, am a sentence", "This is a good sentence."]

# returns the embedded texts
embeddings = lnlp.embed(your_texts)
```

### Model training

If you have embedded texts as well as some labels, you can then train a model like this:

```python
# returns a pytorch model and a label encoder
model, encoder = lnlp.classify(embeddings, your_labels)
```

The model is extremly simple: is a MLP with only one hidden layer. Nothing fancy at all, but usually enough for working with high quality embeddings.