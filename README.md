# VandyHacks_Heartbeat_Classification

Authors: Rastko Stojsin, Teddy Weaver

2020 has been a stressful year for just about everyone. How's your heart doing?...... Are you sure?

To help you find out, we've developed a kinda-sorta accurate heartbeat classifier to determine if your heartbeat is regular (yay!), has a murmur, exhtole, or just too much background noise to tell the difference. All you need is your smartphone!

Armed with ~500 heartbeat audio files of questionale quality on [Kaggle](https://www.kaggle.com/kinguistics/heartbeat-sounds), we transformed the audio files into different visual representations of frequency and pitch.

To create our mediocre multi-class classifier, we used transfer learning of a Convolutional Neural Network (CNN) -- a class of deep neural networks that while orginially desigend for image classifcation. The base archietcture we used for each of our ensemble models was a [ResNeXt-101-32x8d](https://github.com/facebookresearch/ResNeXt).

The result -- Around 72% accuracy in predicting a heartbeat!


--- Next Steps ----

1. Learn better memory management in PyTorch! We ran out of RAM very quickly with higher DPI images
2. Research audio signals to create better features or images
3. Try other models! Would a RNN work better?
4. Host models with a web service. Currently being run as part of the python scripts
5. Take a nap
