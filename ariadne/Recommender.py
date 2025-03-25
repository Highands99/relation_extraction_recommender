import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow.keras.layers import Input, Flatten, Dense, Lambda
from sklearn.neighbors import NearestNeighbors

class RelationExtraction:

    def __init__(self, relation_dict, weights, centroids, temperature, distance):
        # Il costruttore __init__ viene chiamato quando si crea un'istanza della classe
        self.relation_dict = relation_dict
        self.model = None,
        self.weights = weights
        self.centroids = centroids
        self.temperature = temperature
        self.distance = distance
        self.bert, self.preprocessor, self.tokenize = None, None, None

    def initialize_NN(self):
        self.bert, self.preprocessor, self.tokenize = self.init_BERT()
        self.model = self.CreateModelSkeleton(self.weights)

    #Inizializzo il modello dei centroidi 
    def CreateModelSkeleton(self,weights):    
        
        input_shape=1536
        activation='swish'
        depth=5
        pert=0.01
        output_dimensions=15
        neurons_per_layer = round(depth/pert)
        layers=[
                Input(shape=(input_shape)),
                Flatten(),
                *[Dense(neurons_per_layer, activation=activation) for i in range(depth)]]
        
        if output_dimensions:
            layers.append(Dense(output_dimensions, activation=activation))
            
        layers.append(Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))
        model=tf.keras.models.Sequential(layers)

        model.compile() 
        model.build(input_shape=(None, 1))
        model.set_weights(weights)
        
        return model
    


    def init_BERT(only_real_token=False, extract_CLS_only=False):
        
        preprocessor = hub.load("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
        
        # Step 1: tokenize batches of text inputs.
        text_inputs = [tf.keras.layers.Input(shape=(), dtype=tf.string)] # 2 input layers for 2 text inputs
        tokenize = hub.KerasLayer(preprocessor.tokenize)
        tokenized_inputs = [tokenize(segment) for segment in text_inputs]
        
        # Step 3: pack input sequences for the Transformer encoder.
        seq_length = 512  # Your choice here.
        bert_pack_inputs = hub.KerasLayer(
            preprocessor.bert_pack_inputs,
            arguments=dict(seq_length=seq_length))  # Optional argument.
        encoder_inputs = bert_pack_inputs(tokenized_inputs)
        
        encoder = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4",
            trainable=False)
        outputs = encoder(encoder_inputs)
        # Get the output of the [CLS] token, which represents the sentence embedding.
        pooled_output = outputs["pooled_output"]      # [batch_size, 768].
        sequence_output = outputs["sequence_output"]  # [batch_size, seq_length, 768].
        
        # Obtain the token mask indicating which tokens belong to the actual input sentences.
        input_mask = encoder_inputs["input_mask"]
        
        # Cast input mask to float32 to match the data type of sequence output.
        input_mask = tf.cast(input_mask, dtype=tf.float32)
        
        # Apply the input mask to filter out padding tokens.
        filtered_sequence_output = sequence_output * tf.expand_dims(input_mask, axis=-1)
        
        if only_real_token:
            # Keep only tokens from the original input sentence (excluding [CLS] and [SEP])
            start_token_index = 1  # Start after [CLS]
            
            # Compute the end token index based on the input_mask
            end_token_index = tf.reduce_sum(input_mask, axis=-1) - 1
            
            # Squeeze the end_token_index tensor to remove extra dimensions
            end_token_index = tf.squeeze(end_token_index, axis=-1)
            
            # Convert end_token_index to a scalar tensor
            end_token_index = tf.cast(end_token_index, dtype=tf.int32)
            
            # Create a range of indices from start_token_index to end_token_index
            indices = tf.range(start_token_index, end_token_index)
            filtered_sequence_output = filtered_sequence_output[:, start_token_index:end_token_index]
        
        if extract_CLS_only:
            embedding_model = tf.keras.Model(text_inputs, pooled_output)  # Extract [CLS] token embedding if you put pooled_output.
        else:
            embedding_model = tf.keras.Model(text_inputs, filtered_sequence_output) # Extract tokens masked embedding if you put filtered_sequence_output.
        
        #sentences = tf.constant([sentence])
        return embedding_model, preprocessor,tokenize



    def check_if_head_tail_are_included_in_sentence(self, sentence, head_start, head_end, tail_start, tail_end):
        head_included = False
        tail_included = False
        if (head_start >= 0 and head_start<len(sentence)) and (head_end>0 and head_end < len(sentence)):
            head_included = True
        if (tail_start >= 0 and tail_start<len(sentence)) and (tail_end>0 and tail_end < len(sentence)):
            tail_included = True
        return head_included, tail_included



    def find_entity_tensor_slice(self, frase, entity_start_pos, entity_end_pos):
        tokenized_to_entity_end = self.preprocessor.tokenize(tf.constant([frase[:entity_end_pos]])).to_list()[0]
        entity_tokenized=self.preprocessor.tokenize(tf.constant([frase[entity_start_pos:entity_end_pos]])).to_list()[0]
        all_token_slice=[]#Token dell'entità
        for tok in entity_tokenized:
            for i in tok:
                all_token_slice.append(i)
        entity_lenght=len(all_token_slice)
        
        all_token_sentence=[]
        for element in tokenized_to_entity_end:
            for i in element:
                all_token_sentence.append(i)

        # +1 because first token is CLS token and idex must be adjusted
        bound_end=len(all_token_sentence)+1
        bound_start=(bound_end-entity_lenght)
        
        return bound_start,bound_end


    def check_if_head_tail_are_included_in_Tokens(self, sentence, h_s, h_e, t_s, t_e, bert_token_window=512):
        bert_token_window=bert_token_window-2 #Remove CLS and SEP token from token window
        max_index = max(h_s, h_e, t_s, t_e)
        sentence_to_index = sentence[:max_index]
        sub_sentence = tf.constant([sentence_to_index])
        tokenized_sub_text = self.preprocessor.tokenize(sub_sentence)
        tokens_of_sub_text=[]

        for row in tokenized_sub_text:
            # Iterate over elements in each row
            for element in row:
                for i in element.numpy():
                    tokens_of_sub_text.append(i)

        if len(tokens_of_sub_text)>bert_token_window:
            return False,tokens_of_sub_text
        else:
            return True,tokens_of_sub_text

        


    ####Function to generate sentence embedding for each row of a dataframe####
    ####Returns a list containing a tensor for CLS, tensors for head tokens and tail tokens#####
    def embedd_row(self, row):
        emt=self.bert(tf.constant([row['sentence']]))
        included,_=self.check_if_head_tail_are_included_in_Tokens(row['sentence'],  row['head_start'], row['head_end'],
                                                            row['tail_start'], row['tail_end'], bert_token_window=512)
        head_included, tail_included = self.check_if_head_tail_are_included_in_sentence(row['sentence'], row['head_start'], row['head_end'], row['tail_start'], row['tail_end'])
        if not included or not head_included or not tail_included:
            return np.nan


        tensors_cls_head_tail=[]
        tensors_head_start_index, tensors_head_end_index=self.find_entity_tensor_slice(row['sentence'],row['head_start'], row['head_end'])
        tensors_tail_start_index, tensors_tail_end_index=self.find_entity_tensor_slice(row['sentence'], row['tail_start'], row['tail_end'])
        tensors_cls_head_tail.append(emt[0][0]) #CLS
        tensors_cls_head_tail.append(emt[0][tensors_head_start_index:tensors_head_end_index]) #head
        tensors_cls_head_tail.append(emt[0][tensors_tail_start_index:tensors_tail_end_index]) #tail

        return tensors_cls_head_tail
    
    def concat_tensors(self,row):
        mean_head = tf.reduce_mean(row['emb_head'], axis=0, keepdims=True)
        mean_tail = tf.reduce_mean(row['emb_tail'], axis=0, keepdims=True)
        return tf.concat([ mean_head, mean_tail], axis=1)


    def from_sentence_to_headtail(self, sentence_row):
        '''Colonne che si aspetta: 
        columns=['sentence', 'head_start', 'head_end', 'tail_start', 'tail_end']
         - sentence è la stringa contenente la frase
         - head_start posizione di inizio della head
         - head_end posizione di fine della head
         - tail_start posizione di inizio della tail
         - tail_end posizione di fine della tail
        '''

        sentence_row['embeddings'] = sentence_row.apply(lambda row: self.embedd_row(row), axis=1)
        # Filter out rows with NaN embeddings
        sentence_row = sentence_row[~sentence_row['embeddings'].isna()]

        assert len(sentence_row)>0, "head or tail are not included in the sentence or excede the maximum numebr of tokens allowed by Bert." 


        # Splitting the list into separate columns
        sentence_row[['emb_cls', 'emb_head', 'emb_tail']] = sentence_row['embeddings'].apply(pd.Series)

        #print(sentence_row['emb_head'][0])

        sentence_row = sentence_row[['emb_head', 'emb_tail']]

        #print(sentence_row)

        sentence_row['K'] = sentence_row.apply(self.concat_tensors, axis=1)

        #print(sentence_row['K'])

        headtail = sentence_row['K'][0]

        print(headtail)


        return headtail


    def compute_predictions_and_scores(self, 
                                       sentence_row,
                                       cutoff=3):   
 
        headtail=self.from_sentence_to_headtail(sentence_row)
        if self.distance == 'cosine_triangle':
            from scipy.spatial.distance import cosine
            def cosine_triangle(x,y):
                return np.sqrt(np.maximum(2* cosine(x,y),  np.finfo(float).eps))
            
            self.distance=cosine_triangle
            algo='brute'

        elif self.distance == 'euclidean_squared':
            from scipy.spatial.distance import sqeuclidean
            self.distance=sqeuclidean
            algo='brute'
            
        elif self.distance == 'euclidean':
            algo='ball_tree'
            
        elif self.distance == 'cosine':
            algo='brute'

        kNN = self.centroids.shape[1]
        centroids_y=np.diag([1]*kNN)

        y_pred_test=self.model(headtail)

        nbrs = NearestNeighbors(n_neighbors=kNN, 
                                algorithm=algo, 
                                metric=self.distance, 
                                n_jobs=-1).fit(self.centroids.T)
        distances_embedding, indices = nbrs.kneighbors(y_pred_test)

        weights = np.exp(-distances_embedding / self.temperature)
        weightsP = np.sum(np.expand_dims(weights, -1) * centroids_y[indices, :], 1)

        weightsN=np.sum(np.expand_dims(weights, -1) * (1-centroids_y)[indices, :], 1)
        weightsPN=np.concatenate([weightsP,weightsN],0)

        norm_w_batch = (weightsPN/(weightsPN.sum(0,keepdims=True)))[0]

        pred_labels=(norm_w_batch.round(2) > cutoff * 1/kNN).astype(int)
        labels_score=norm_w_batch*(kNN)
        inverted_dict = {v: k for k, v in self.relation_dict.items()}

        predictions_and_scores=dict([(inverted_dict[i],round(labels_score[i],2)) for i, pred in enumerate(pred_labels) if pred==1])
        sorted_predictions_and_scores=dict(sorted(predictions_and_scores.items(), key=lambda key_val: key_val[1], reverse=True))
        return sorted_predictions_and_scores
    
  