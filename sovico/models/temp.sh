jupyter nbconvert --to script rnn.ipynb
python rnn.py --data_dir=../data/1234people_160000frame_simple --window_size=40 --after=80 --save_dir=../checkpoint --rnn_hiddens 16 16 16
