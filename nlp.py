from transformers import AutoTokenizer, AutoModel
from torch.nn import functional as F

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from sentence_transformers import SentenceTransformer

from tqdm import tqdm

import torch 
import torch.nn as nn

import pickle

import sys

import numpy as np

class LazyNLP:
    def zeroshot(self, sentences, label_list):
        # Load sentence bert both as the toekenizer and the model
        tokenizer = AutoTokenizer.from_pretrained("deepset/sentence_bert") 
        model = AutoModel.from_pretrained("deepset/sentence_bert")

        print("Getting zeroshot labels...")
        zeroshot_labels = []
        for sent in tqdm(sentences):
            # Tokenize the sentence and the labels
            inputs = tokenizer.batch_encode_plus([sent] + label_list, return_tensors="pt", padding=True)
            
            # Get the sentence and the attention mask from the inputs
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            # Pass the ids and the mask through the model
            outputs = model(input_ids, attention_mask=attention_mask)[0]

            # Get the sentence and the labels representations
            sentence_rep = outputs[:1].mean(dim=1)
            labels_rep = outputs[1:].mean(dim=1)

            # Use cosine similarity to find the most similar label
            similarity = F.cosine_similarity(sentence_rep, labels_rep, dim=1)
            closest = similarity.argsort(descending=True)

            # get the zeroshot label
            zeroshot_labels.append(label_list[closest[0]])

        print("- - - - - - - -")
        return zeroshot_labels 

    def embed(self, sentences):
        print("Embedding sentences...")

        retriever = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = retriever.encode(sentences, show_progress_bar=True)

        print("- - - - - - - -")
        print()
        
        return embeddings

    def classify(self, embeddings, zeroshot_labels):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Convert emebddings to tensor
        tensor_embeddings =  torch.FloatTensor(embeddings).to(device)

        # Encode labels
        encoder = LabelEncoder()
        zeroshot_labels = encoder.fit_transform(zeroshot_labels)
        tensor_target = torch.LongTensor(zeroshot_labels).to(device)

        num_classes = len(set(zeroshot_labels))

        # Setting up a dense neural network with pytorch
        model = nn.Sequential(
            nn.Linear(embeddings.shape[1], 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 24),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(24, num_classes)
        ).to(device)

        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())

        print("Training model...")
        print()

        # Training loop of the model
        losses = []
        num_epochs = int(round(len(zeroshot_labels) / 2))
        for epoch in tqdm(range(num_epochs)):

            # Create prediction of the model
            y_hat = model(tensor_embeddings)

            # Calculate loss
            loss = loss_function(y_hat, tensor_target)
            losses.append(loss.item())

            # Backpropagate to update weights
            model.zero_grad()
            loss.backward()

            sys.stdout.write(f'\rLoss of {round(float(loss), 3)}')
            sys.stdout.flush()

            optimizer.step()

        try:
            # Save model
            torch.save(model, "ml/model.pt")

            # Save encoder
            with open("ml/encoder.pkl", "wb") as handle:
                pickle.dump(encoder, handle)

        except:
            pass

        print("- - - - - - - -")
        print("Done!")
        return model

    def predict(self, sentences):
        # Load model
        model = torch.load("ml/model.pt")

        # Load encoder
        with open("ml/encoder.pkl", "rb") as handle:
            encoder = pickle.load(handle)

        # embed new sentences
        new_embeddings = self.embed(sentences)

        # Convert emebddings to tensor
        tensor_embeddings =  torch.FloatTensor(new_embeddings)

        # Create prediction of the model
        y_hat = model(tensor_embeddings)

        # Convert to numpy
        y_hat_numpy = [loss.detach().numpy() for loss in y_hat]

        # Get the argmax of the prediction
        y_hat_argmax = np.argmax(y_hat_numpy)

        # Decode the labels
        y_hat_decoded = encoder.inverse_transform(y_hat_argmax.ravel()) # .reshape(-1, 1)

        return str(y_hat_decoded[0])

    def run(self, sentences, labels):
        # Get the zeroshot labels
        zeroshot_labels = self.zeroshot(sentences, labels)

        # Get the embeddings
        embeddings = self.embed(sentences)

        # Train the model
        model = self.classify(embeddings, zeroshot_labels)

        return model




