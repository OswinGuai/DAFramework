#python sample_for_few-shot.py /data/office/amazon_list.txt src 20 1
#python sample_for_few-shot.py /data/office/webcam_list.txt src 8 1
python sample_for_few-shot.py /data/office/dslr_list.txt src 8 1
python sample_for_few-shot.py /data/office/webcam_list.txt tgt 3 1
python sample_for_few-shot.py /data/office/dslr_list.txt tgt 3 1
python sample_for_few-shot.py /data/office/amazon_list.txt tgt 3 1

DOC='''
Output like this:

Generate samples for target domain in few-shot DA.
The number of target samples for per category: 3.
The number of splits: 5.
The tag of output file: /data/office-home/Art.
The output files name like /data/office-home/Art_train_1.txt and /data/office-home/Art_test_1.txt.

Sampling is finished.
'''
