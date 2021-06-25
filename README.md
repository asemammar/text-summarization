# Pointer-Generator Networks with Different Word Embeddings forAbstractive Summarization

**Burak Suyunu, Muhammed Emin Güre**  
Department of Computer Engineering, Boğaziçi University  
CmpE 58T - Adv. Natural Language Processing (Spring 2021)

-----

## Requirements
* python >= 3.6
* TensorFlow >= 2

Package dependencies are in `requirements.txt` file.  
To install all run `pip install -r requirements.txt`.

## Turkish Dataset (MLSUM)

### Option 1 - Prepare Data Yourself
```bash
cd src
python mlsum_data_prep.py
```

### Option 2 - Download Preprocessed Data
You can directly download the preprocessed Turkish dataset using the following link: https://drive.google.com/drive/folders/1f3Q3OGWX3BIgHu_8PuO8LjszzFwqi_oG?usp=sharing


## English Dataset (CNN/Daily Mail)
The CNN/Daily Mail Dataset available online is stored in binary files. However the pipeline to use binary files is inefficient. In this project we use TensorFlows's tfrecords files to store datasets.

### Option 1 - Prepare Data Yourself
Or you can obtain the raw and preprocessed dataset from this link: https://github.com/steph1793/CNN-DailyMail-Bin-To-TFRecords

### Option 2 - Download Preprocessed Data
You can directly download the preprocessed dataset using with this link: https://drive.google.com/drive/folders/1_-GvNvL1DB8t0tjZgjHfm6JbChgw6jG4?usp=sharing


## Embedding Matrices

To be able to run the model with pretrained embeddings, you need to generate the embedding matrix of the corresponding pre-trained word embedding method. 

### Option 1 - Prepare Data Yourself
To generate the embedding matrices yourself:

1. Go to <code>src/embeddings/</code>
2. Choose the method that you want to generate embedding matrix and open the corresponding file (<code>glove.py</code> for pre-trained English GloVe embeddings)
3. Take required actions and make path changes written as comments if there is any.
  3.1. For <code>nnlm.py | use.py | use-tr.py</code>, you don't need to make any adjustments.
4. Run the code <code>python embedding_name.py</code> (<code>python glove.py</code> for English GloVe)

### Option 2 - Download Preprocessed Data
You can directly download the preprocessed Embedding Matirces with this link: https://drive.google.com/drive/folders/18a-keUl5GAAQmZf-i3lUwYv2z8YPh888?usp=sharing


## Training Models

### Option 1 - Train the Model Yourself

For English Models:

```bash
python src/main.py --mode=train --data_dir=/path/to/tfrecords_finished_files/chunked_train --vocab_path=/path/to/tfrecords_finished_files/vocab --checkpoint_dir=/path/to/Checkpoints/embedding_name
```

For English Models with Embedding:

```bash
python src/main.py --mode=train --data_dir=/path/to/tfrecords_finished_files/chunked_train --vocab_path=/path/to/tfrecords_finished_files/vocab --checkpoint_dir=/path/to/Checkpoints/embedding-name --pt_embedding=/path/to/embeddings/embedding-name_embedding_matrix.pk --embed_size=embedding-dimension
```

For Turkish Models:

```bash
python src/main.py --mode=train --data_dir=/path/to/mlsum/train.tfrecords --vocab_path=/path/to/mlsum/vocab --checkpoint_dir=/path/to/Checkpoints/embedding_name
```

For Turkish Models with Embedding:

```bash
python src/main.py --mode=train --data_dir=/path/to/mlsum/train.tfrecords --vocab_path=/path/to/mlsum/vocab --checkpoint_dir=/path/to/Checkpoints/embedding-name --pt_embedding=/path/to/embeddings/embedding-name_embedding_matrix.pk --embed_size=embedding-dimension
```

### Option 2 - Download Pretrained Models

We have trained each model for 5000 iterations and saved as checkpoints. You can directly download the pretrained models with this link: https://drive.google.com/drive/folders/1-aIVMb4jCJF515KcSpH2uLSfV88ZqxlK?usp=sharing

Give the pretrained model directory as --checkpoint_dir parameter to the model to continue training or evaluate/test with it.


## Testing/Evaluating Models

While testing and evaluating results, batch size and beam size must be equal.

When mode is set to eval, the code outputs the ROUGE score according to the files found in --data_dir. Default number to evaluate ROUGE score is 5.

When mode is set to test, the code outputs some translation examples to directory given as parameter to test_save_dir.

For English Models with Embedding:

```bash
python src/main.py --mode=eval --data_dir=/path/to/tfrecords_finished_files/chunked_val --vocab_path=/path/to/tfrecords_finished_files/vocab --checkpoint_dir=/path/to/Checkpoints/embedding-name --pt_embedding=/path/to/embeddings/embedding-name_embedding_matrix.pk --embed_size=embedding-dimension --batch_size=4 --beam_size=4
```

For Turkish Models with Embedding:

```bash
python src/main.py --mode=eval --data_dir=/path/to/mlsum/val.tfrecords --vocab_path=/path/to/mlsum/vocab --checkpoint_dir=/path/to/Checkpoints/embedding-name --pt_embedding=/path/to/embeddings/embedding-name_embedding_matrix.pk --embed_size=embedding-dimension --batch_size=4 --beam_size=4
```

## Demo
```bash
cd src
streamlit run app.py
```

![Demo](https://github.com/emingure/text-summarization/blob/main/assets/demo.png?raw=true)


## Reference

Code base is taken from https://github.com/steph1793/Pointer_Generator_Summarizer