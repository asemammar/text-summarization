# Pointer-Generator Networks with Different Word Embeddings forAbstractive Summarization

**Burak Suyunu, Muhammed Emin Güre**
Department of Computer Engineering, Boğaziçi University
CmpE 58T - Adv. Natural Language Processing (Spring 2021)

-----

### Requirements
* python > 3.6

Package dependencies are in `requirements.txt` file. To install all run `pip install -r requirements.txt`.

### Turkish Dataset Preparation
```bash
cd src
python mlsum_data_prep.py
```

### Demo
```bash
cd src
streamlit run app.py
```

![Demo](https://github.com/emingure/text-summarization/blob/main/assets/demo.png?raw=true)