import tensorflow as tf
from model import PGN
from training_helper import train_model
from test_helper import beam_decode
from batcher import batcher, Vocab, Data_Helper
from tqdm import tqdm
from rouge import Rouge
import pprint
import pandas as pd
import pickle

def train(params):
	assert params["mode"].lower() == "train", "change training mode to 'train'"

	tf.compat.v1.logging.info("Building the model ...")
	model = PGN(params)

	print("Creating the vocab ...")
	vocab = Vocab(params["vocab_path"], params["vocab_size"])

	print("Creating the batcher ...")
	b = batcher(params["data_dir"], vocab, params)
	

	print("Creating the checkpoint manager")
	checkpoint_dir = "{}".format(params["checkpoint_dir"])
	ckpt = tf.train.Checkpoint(step=tf.Variable(0), PGN=model)
	ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=11)

	ckpt.restore(ckpt_manager.latest_checkpoint)
	if ckpt_manager.latest_checkpoint:
		print("Restored from {}".format(ckpt_manager.latest_checkpoint))
	else:
		print("Initializing from scratch.")

	tf.compat.v1.logging.info("Starting the training ...")
	train_model(model, b, params, ckpt, ckpt_manager, "output.txt")
 

def test(params):
	assert params["mode"].lower() in ["test","eval", "test_request"], "change training mode to 'test' or 'eval'"
	assert params["beam_size"] == params["batch_size"], "Beam size must be equal to batch_size, change the params"

	tf.compat.v1.logging.info("Building the model ...")
	model = PGN(params)
	print("Creating the vocab ...")
	vocab = Vocab(params["vocab_path"], params["vocab_size"])

	print("Creating the batcher ...")
	b = batcher(params["data_dir"], vocab, params)

	print("Creating the checkpoint manager")
	checkpoint_dir = "{}".format(params["checkpoint_dir"])
	ckpt = tf.train.Checkpoint(step=tf.Variable(0), PGN=model)
	ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=11)

	path = params["model_path"] if params["model_path"] else ckpt_manager.latest_checkpoint
	ckpt.restore(path)
	print("Model restored")



	for batch in b:
		# try:
		# 	pprint.pprint(model.summary())
		# except:
		# 	print('hhhhhhhhhhhhhhhhhhhhhhhhhhhhh')
		yield  beam_decode(model, batch, vocab, params)


def test_and_save(params):
	assert params["test_save_dir"], "provide a dir where to save the results"
	gen = test(params)

	def getRow(x):

		temp = {


			'PRED': x.abstract,

			'GOLD': x.real_abstract,

			'SOURCE': x.text,

		}
		return temp
	with tqdm(total=params["num_to_test"],position=0, leave=True) as pbar:
		r = pd.DataFrame()
		for i in range(params["num_to_test"]):
			trial = next(gen)

			r=r.append(getRow(trial),ignore_index=True)
			# with open(params["test_save_dir"]+"/article_"+str(i)+".txt", "w",encoding='utf-8') as f:
			# 	f.write("article:\n")
			# 	f.write(trial.text)
			# 	f.write("\n\nabstract:\n")
			# 	f.write(trial.abstract)
			pbar.update(1)
		r.to_csv(params["test_save_dir"]+"/results.txt", encoding='utf-8')
	ro = Rouge()
	scores = ro.get_scores(r['PRED'], r['GOLD'], avg=True)
	print("\n\n")
	pprint.pprint(scores)

def test_and_serve(params):
	res = []
	gen = test(params)
	trial = next(gen)
	return trial.abstract

def evaluate(params):
	gen = test(params)
	reals = []
	preds = []
	with tqdm(total=params["max_num_to_eval"],position=0, leave=True) as pbar:
		for i in range(params["max_num_to_eval"]):
			trial = next(gen)
			reals.append(trial.real_abstract)
			preds.append(trial.abstract)
			pbar.update(1)
	r=Rouge()
	scores = r.get_scores(preds, reals, avg=True)
	print("\n\n")
	pprint.pprint(scores)